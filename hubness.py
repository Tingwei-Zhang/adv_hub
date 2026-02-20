import sys
from tqdm import tqdm
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import extract_args, jpeg, pgd_step, threshold,\
                  load_model_data_and_dataset, criterion,\
                  evaluate_retrieval_and_save, cosine_similarity_with_fallback,\
                  generate_or_load_centroids


# Validate and configure experiment
cfg = extract_args(sys.argv[1])
if cfg.save_training:
    print('Saving training data...')

# Instantiate models, devices, and datasets
print('Loading models and data...')
model_data, dataset = load_model_data_and_dataset(cfg.dataset_flag, cfg.model_flags,
                                                  cfg.gpu_nums, cfg.seed)

data_device = model_data[0][2]              # Assign initial device to hold data
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

# Create Empty Lists for Logging
X_advs = []
theoretical_adv_loss,adv_loss, gt_loss = [], [], []                  # Ground truth and adversarial distances
adv_embeds_list =  []

# image x text
print('Generating Illusions...')
torch.manual_seed(cfg.seed)

centroids = generate_or_load_centroids(cfg)

if centroids.shape[0] < cfg.number_images:
    raise ValueError(
        f"Insufficient centroid count ({centroids.shape[0]}) for number_images={cfg.number_images}."
    )


for i, (X, gt, y_id, y_orig) in enumerate(dataloader):
    if i >= (cfg.number_images // cfg.batch_size):
        break

    # Compute real image indices for this batch
    real_img_indices = torch.arange(cfg.batch_size) + (i * cfg.batch_size)

    print(f"Processing Batch {i}, Image Indices: {real_img_indices.tolist()}")

    X_max, X_min = threshold(X, cfg.epsilon, cfg.modality, data_device)

    for m, l, d in model_data:
        with torch.no_grad():
            Y = l.to(d)  # Move ground truth labels to device
            centroids_batch = centroids[real_img_indices].to(d)  # Select centroids for batch

    pbar = tqdm(range(cfg.max_epochs))
    lr = cfg.lr

    # Initialize adversarial batch
    X_adv = X.clone().detach().to(data_device).requires_grad_(True)

    for j in pbar:
        with torch.no_grad(): 
            if cfg.jpeg:
                X_adv = jpeg(X_adv.cpu()).to(data_device)

        for m, l, d in model_data:
            X_adv, _, loss = pgd_step(
                model=m,
                X=X_adv.to(d).requires_grad_(True),
                Y=centroids_batch,
                X_min=X_min,
                X_max=X_max,
                lr=lr,
                modality=cfg.modality,
                device=data_device,
            )
            X_adv = X_adv.detach().requires_grad_(True)
            torch.cuda.empty_cache()

        pbar.set_postfix({'loss': loss.mean().item(), 'lr': lr})

        if ((j + 1) % cfg.gamma_epochs) == 0:
            lr *= 0.9  # Decay learning rate

    # Compute embeddings and losses for the entire batch
    with torch.no_grad():
        adv_embeds = m.forward(X_adv.to(data_device), cfg.modality, normalize=False)
        clean_embeds = m.forward(X.to(data_device), cfg.modality, normalize=False)

    # Compute batch-wise losses
    theoretical_adv_loss_value = (1 - criterion(adv_embeds, centroids_batch, dim=1))
    adv_loss_value = (1 - criterion(adv_embeds.unsqueeze(1), Y.unsqueeze(0), dim=2))
    gt_loss_value = (1 - criterion(clean_embeds.unsqueeze(1), Y.unsqueeze(0), dim=2))

    # Store batch-wise results
    theoretical_adv_loss.append(theoretical_adv_loss_value)
    adv_loss.append(adv_loss_value)
    gt_loss.append(gt_loss_value)
    
    X_advs.append(X_adv.detach().cpu().clone())
    adv_embeds_list.append(adv_embeds.detach().cpu().clone())

    # print('Adversarial loss:', adv_loss_value)
    # print('Clean loss:', gt_loss_value)


# Save data
if not (cfg.model_flag == 'audioclip' and cfg.modality == 'audio'):
    X_advs_tensor = torch.cat(X_advs, dim=0)  # Shape: (N, C, H, W)
    np.save(cfg.output_dir + 'x_advs', X_advs_tensor.numpy())

theoretical_adv_loss = torch.cat(theoretical_adv_loss, dim=0).cpu().numpy()
adv_loss = torch.cat(adv_loss, dim=0).cpu().numpy()
gt_loss = torch.cat(gt_loss, dim=0).cpu().numpy()

np.save(cfg.output_dir + 'adv_loss', adv_loss)
np.save(cfg.output_dir + 'gt_loss', gt_loss)
np.save(cfg.output_dir + 'theoretical_adv_loss',theoretical_adv_loss)

# Computing adv_test_similarity_matrix 
adv_embeds_tensor = torch.cat(adv_embeds_list, dim=0).cuda()  # Shape: (N, 1024)

Y = Y.to(data_device)
adv_test_similarity_matrix = cosine_similarity_with_fallback(
    reference_embeds=Y,
    query_embeds=adv_embeds_tensor,
    device=data_device,
)

adv_test_similarity_matrix = adv_test_similarity_matrix.detach().cpu().numpy()
with open(cfg.output_dir + 'adv_test_similarity_matrix.pkl', 'wb') as f:
    pickle.dump(adv_test_similarity_matrix, f)

# Run retrieval evaluation and save report
_, eval_output_path = evaluate_retrieval_and_save(
    dataset_flag=cfg.dataset_flag,
    model_flag=cfg.model_flag,
    output_dir=cfg.output_dir,
    adv_test_similarity_matrix=adv_test_similarity_matrix,
)

print("Files have been saved to:", cfg.output_dir)
print("Evaluation results have been saved to:", eval_output_path)

if cfg.save_training:
    # Computing adv_train_similarity_matrix 
    training_embeds_path = cfg.training_embeds_path
    training_embeds = torch.tensor(np.load(training_embeds_path)).to('cuda')
    adv_train_similarity_matrix = torch.nn.functional.cosine_similarity(
        training_embeds.unsqueeze(0), 
        adv_embeds_tensor.unsqueeze(1),
        dim=2
    )
    adv_train_similarity_matrix = adv_train_similarity_matrix.detach().cpu().numpy()
    with open(cfg.output_dir + 'adv_train_similarity_matrix.pkl', 'wb') as f:
        pickle.dump(adv_train_similarity_matrix, f)

    # Computing test_similarity_matrix and train_similarity_matrix
    test_similarity_list = []
    train_similarity_list=[]
    with torch.no_grad():
        for i, (X_batch, gt, y_id, y_orig) in enumerate(tqdm(dataloader)):
            X_batch = jpeg(X_batch.cpu()).to(data_device) if cfg.jpeg else X_batch.to(data_device)
            # Compute image embeddings
            clean_embeds = m.forward(X_batch, cfg.modality, normalize=False)
            # Compute cosine similarity
            train_similarity = torch.nn.functional.cosine_similarity(
                training_embeds.unsqueeze(0), 
                clean_embeds.unsqueeze(1),             
                dim=2                       
            )
        
            train_similarity_list.append(train_similarity.cpu().numpy())
            test_similarity = torch.nn.functional.cosine_similarity(
                Y.unsqueeze(0), 
                clean_embeds.unsqueeze(1),             
                dim=2                       
            )    
            test_similarity_list.append(test_similarity.cpu().numpy())
    # Concatenate all similarity matrices
    test_similarity_matrix = np.concatenate(test_similarity_list, axis=0)
    test_similarity_matrix = test_similarity_matrix.T
    train_similarity_matrix = np.concatenate(train_similarity_list, axis=0)
    train_similarity_matrix = train_similarity_matrix.T

    with open(cfg.output_dir + 'test_similarity_matrix.pkl', 'wb') as f:
        pickle.dump(test_similarity_matrix, f)

    with open(cfg.output_dir + 'train_similarity_matrix.pkl', 'wb') as f:
        pickle.dump(train_similarity_matrix, f)
