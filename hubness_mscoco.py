import sys
from tqdm import tqdm
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from utils import extract_args, jpeg, pgd_step, print_results, threshold,\
                  load_model_data_and_dataset, criterion


# Validate and configure experiment
cfg = extract_args(sys.argv[1])
# cfg = extract_args("mscoco/imagebind")
# cfg = extract_args("mscoco/imagebind_evasion_50")
# cfg = extract_args("mscoco/openclip")


train_test_flag = cfg.train_test_flag.lower() == "true"
print("Train Test Flag: ", train_test_flag)
if train_test_flag == True:
    print("Train Ratio: ", cfg.train_ratio)

# Instantiate models, devices, and datasets
print('Loading models and data...')
model_data, dataset = load_model_data_and_dataset(cfg.dataset_flag, cfg.model_flags,
                                                  cfg.gpu_nums, cfg.seed)
if cfg.target_model_flag is not None:
    if cfg.target_model_flag in cfg.model_flags:
        print('Using (one of) the same model(s) for target and input...')
        target_tup = model_data[cfg.model_flags.index(cfg.target_model_flag)]
    else:
        target_tup, _ =\
            load_model_data_and_dataset(cfg.dataset_flag, [cfg.target_model_flag],
                                        [cfg.target_gpu_num], cfg.seed)[0]

data_device = model_data[0][2]              # Assign initial device to hold data
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

# Create Empty Lists for Logging
X_advs = []
X_inits, gts = [], []                       # Initial images and ground truths
adv_loss, gt_loss = [], []                  # Ground truth and adversarial distances
adv_embeds_list =  []

# image x text
print('Generating Illusions...')
torch.manual_seed(cfg.seed)

for i, (X, gt, y_id, y_orig) in enumerate(dataloader):
    if i >= (cfg.number_images // cfg.batch_size):
        break
    X_inits.append(X.clone().detach().cpu())
    np.save(cfg.output_dir + 'x_inits', np.concatenate(X_inits))

    for img_idx in range(cfg.batch_size):
        print("-----Image_"+str(cfg.batch_size * i +img_idx)+"-----")
        selected_X = X[img_idx].unsqueeze(0)
        X_init = X.clone().detach().cpu()  # Detach and move to CPU
        X_max, X_min = threshold(selected_X, cfg.epsilon, cfg.modality, data_device)

        for m, l, d in model_data:
            with torch.no_grad():
                Y = l.to(d)

        if train_test_flag == True:
            train_size = int(cfg.train_ratio * len(Y))
            val_size = len(Y) - train_size
            Y_train, Y_val = random_split(Y, [train_size, val_size])
            Y_train_tensor = torch.stack([Y_train[i] for i in range(len(Y_train))])
            Y_val_tensor = torch.stack([Y_val[i] for i in range(len(Y_val))])
        else:
            Y_train_tensor = Y
            Y_val_tensor = Y

        pbar = tqdm(range(cfg.max_epochs))
        lr = cfg.lr
        for idx, j in enumerate(pbar):
            if idx == 0:
                X_adv = selected_X
            with torch.no_grad(): 
                X_adv = jpeg(X_adv.cpu()).to(data_device) if cfg.jpeg else X_adv
            for m, l, d in model_data:
                X_adv= X_adv.to(d).requires_grad_(True)
                embeds = m.forward(X_adv, cfg.modality, normalize=False)
                loss = 1 - criterion(embeds, Y_train_tensor, dim=1)
                update = lr * torch.autograd.grad(outputs=loss.mean(), inputs=X_adv)[0].sign()
                X_adv = (X_adv.detach().cpu() - update.detach().cpu()).to(data_device)
                X_adv = torch.clamp(X_adv, min=X_min, max=X_max).detach().requires_grad_(True)
                X_adv.grad = None
                torch.cuda.empty_cache()
                loss = loss.clone().detach().cpu()

            pbar.set_postfix({'loss': loss.mean().item(), 'lr': lr})
            if ((j + 1) % cfg.gamma_epochs) == 0:
                lr *= 0.9

        # Compute embeddings and losses
        with torch.no_grad():
            adv_embeds = m.forward(X_adv.to(data_device), cfg.modality, normalize=False)
            clean_embeds = m.forward(selected_X.to(data_device), cfg.modality, normalize=False)

        # Compute losses
        adv_loss_value = (1 - criterion(adv_embeds, Y_val_tensor, dim=1)).mean().item()
        gt_loss_value = (1 - criterion(clean_embeds, Y_val_tensor, dim=1)).mean().item()

        adv_loss.append(adv_loss_value)
        gt_loss.append(gt_loss_value)
        X_advs.append(X_adv.detach().cpu().clone())
        adv_embeds_list.append(adv_embeds.detach().cpu().clone())


        print('Adversarial loss:', adv_loss_value)
        print('Clean loss:', gt_loss_value)

# Save data
X_advs_tensor = torch.cat(X_advs, dim=0)  # Shape: (N, C, H, W)
np.save(cfg.output_dir + 'x_advs', X_advs_tensor.numpy())
np.save(cfg.output_dir + 'adv_loss', np.array(adv_loss))
np.save(cfg.output_dir + 'gt_loss', np.array(gt_loss))

# Computing adv_test_similarity_matrix 
adv_embeds_tensor = torch.cat(adv_embeds_list, dim=0).cuda()  # Shape: (N, 1024)

Y = Y.to(data_device)
adv_test_similarity_matrix = torch.nn.functional.cosine_similarity(
    Y.unsqueeze(0), adv_embeds_tensor.unsqueeze(1), dim=2
)
adv_test_similarity_matrix = np.array(adv_test_similarity_matrix.detach().cpu())
with open(cfg.output_dir + 'adv_test_similarity_matrix.pkl', 'wb') as f:
    pickle.dump(adv_test_similarity_matrix, f)


# Computing adv_train_similarity_matrix 
training_embeds_path = cfg.training_embeds_path
training_embeds = torch.tensor(np.load(training_embeds_path)).to('cuda')
adv_train_similarity_matrix = torch.nn.functional.cosine_similarity(
    training_embeds.unsqueeze(0), 
    adv_embeds_tensor.unsqueeze(1),
    dim=2
)
adv_train_similarity_matrix = np.array(adv_train_similarity_matrix.detach().cpu())
with open(cfg.output_dir + 'adv_train_similarity_matrix.pkl', 'wb') as f:
    pickle.dump(adv_train_similarity_matrix, f)


# Computing test_similarity_matrix and train_similarity_matrix
test_similarity_list = []
train_similarity_list=[]
with torch.no_grad():
    for i, (X_batch, gt, y_id, y_orig) in enumerate(tqdm(dataloader)):
        X_batch = X_batch.to(data_device) 
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
