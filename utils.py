from argparse import Namespace
from pathlib import Path
import os
import toml
from dataset_utils import create_dataset, get_embeddings
from models import load_model
import torch
import numpy as np
from tqdm import tqdm
from DiffJPEG.compression import compress_jpeg
from DiffJPEG.decompression import decompress_jpeg
from DiffJPEG.jpeg_utils import diff_round, quality_to_factor


criterion = torch.nn.functional.cosine_similarity
EMBEDDING_FNM = 'outputs/embeddings/{}/{}_{}_query_embeddings.npy'

IMG_MEAN=(0.48145466, 0.4578275, 0.40821073)
IMG_STD=(0.26862954, 0.26130258, 0.27577711)
THERMAL_MEAN=(0.2989 * IMG_MEAN[0]) + (0.5870 * IMG_MEAN[1]) + (0.1140 * IMG_MEAN[2])
THERMAL_STD=(0.2989 * IMG_STD[0]) + (0.5870 * IMG_STD[1]) + (0.1140 * IMG_STD[2])

def unnorm(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    if type(mean) != float:
        mean = torch.tensor(mean)[None, :, None, None]
        std = torch.tensor(std)[None, :, None, None]
    return ((tensor.clone().cpu() * std) + mean).to(device)

def norm(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    if type(mean) != float:
        mean = torch.tensor(mean)[None, :, None, None]
        std = torch.tensor(std)[None, :, None, None]
    return ((tensor.clone().cpu() - mean) / std).to(device)


def unnorm_audio(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    m = torch.tensor(-4.268)
    s = torch.tensor(9.138)
    return ((tensor.clone().cpu() * s) + m).to(device)

def norm_audio(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    m = torch.tensor(-4.268)
    s = torch.tensor(9.138)
    return ((tensor.clone().cpu() - m) / s).to(device)

def threshold(X, eps, modality, device, do_norm=True):
    if modality == 'vision':
        X_unnorm = unnorm(X.data)
        X_max, X_min = norm(torch.clamp(X_unnorm+eps, min=0, max=1)), norm(torch.clamp(X_unnorm-eps, min=0, max=1))
    elif modality == 'thermal':
        X_max, X_min = torch.clamp(X+eps, min=0, max=1), torch.clamp(X-eps, min=0, max=1)
    elif modality == 'audio':
        X_max, X_min = X + eps, X - eps
    if do_norm==False:
        X_max, X_min = torch.clamp(X+eps, min=0, max=1), torch.clamp(X-eps, min=0, max=1)
    return X_max.to(device), X_min.to(device)

def extract_args(exp_name):
    fnm = f'configs/{exp_name}.toml'
    print(f'Loading config from {fnm}...')

    cfg_dict = toml.load(fnm)['general']

    Path(cfg_dict['output_dir']).mkdir(parents=True, exist_ok=True)
    if 'model_flag' in cfg_dict:
        cfg_dict['model_flags'] = [cfg_dict['model_flag']]
        cfg_dict['target_model_flag'] = cfg_dict['model_flag']
    cfg_dict['target_model_flag'] = cfg_dict.get('target_model_flag', None)

    if 'gpu_num' in cfg_dict:
        cfg_dict['gpu_nums'] = [cfg_dict['gpu_num']]

    cfg_dict['jpeg'] = ('jpeg' in cfg_dict) and cfg_dict['jpeg']
    assert (not cfg_dict['jpeg']) or (cfg_dict['modality'] == 'vision')

    if cfg_dict['modality'] == 'vision':
        cfg_dict['epsilon'] = cfg_dict['epsilon'] / 255

    if type(cfg_dict['epochs']) == list:
        cfg_dict['max_epochs'] = max(cfg_dict['epochs'])
    else:
        cfg_dict['max_epochs'] = cfg_dict['epochs']
        cfg_dict['epochs'] = [cfg_dict['epochs']]

    if 'save_training' in cfg_dict:
        cfg_dict['save_training'] = True
    else:
        cfg_dict['save_training'] = False

    # assert cfg_dict['number_images'] % cfg_dict['batch_size'] == 0
    return Namespace(**cfg_dict)

def extract_eval_args(exp_name):
    fnm = f'configs/{exp_name}.toml'
    print(f'Loading config from {fnm}...')

    cfg_dict = toml.load(fnm)['general']

    if 'model_flag' in cfg_dict:
        cfg_dict['model_flags'] = [cfg_dict['model_flag']]
        cfg_dict['target_model_flag'] = cfg_dict['model_flag']
    cfg_dict['target_model_flag'] = cfg_dict.get('target_model_flag', None)

    if 'gpu_num' in cfg_dict:
        cfg_dict['gpu_nums'] = [cfg_dict['gpu_num']]

    assert cfg_dict['eval_type'] in ['adversarial', 'organic', 'transfer']
    if cfg_dict['eval_type'] == 'transfer':
        assert 'adv_file' in cfg_dict

    assert cfg_dict['number_images'] % cfg_dict['batch_size'] == 0
    return Namespace(**cfg_dict)

def jpeg(x, height=224, width=224, rounding=diff_round, quality=80):
    img_tensor = unnorm(x).squeeze(0)
    factor = quality_to_factor(quality)
    y, cb, cr = compress_jpeg(img_tensor, rounding=rounding, factor=factor)
    img_tensor = decompress_jpeg(y, cb, cr, height, width, rounding=rounding, factor=factor)
    return norm(img_tensor)

def pgd_step(model, X, Y, X_min, X_max, lr, modality, device):
    embeds = model.forward(X, modality, normalize=False)
    loss = 1 - criterion(embeds, Y, dim=1)
    update = lr * torch.autograd.grad(outputs=loss.mean(), inputs=X)[0].sign()
    X = (X.detach().cpu() - update.detach().cpu()).to(device)
    X = torch.clamp(X, min=X_min, max=X_max).requires_grad_(True)
    return X, embeds, loss.clone().detach().cpu()


def cosine_similarity_with_fallback(reference_embeds, query_embeds, device, chunk_size=100):
    """
    Compute cosine-similarity matrix with optional chunked fallback.
    Returns a matrix with shape: (num_query, num_reference).
    """
    reference_embeds = reference_embeds.to(device)
    query_embeds = query_embeds.to(device)
    try:
        return torch.nn.functional.cosine_similarity(
            reference_embeds.unsqueeze(0),
            query_embeds.unsqueeze(1),
            dim=2,
        )
    except RuntimeError:
        print("Memory error occurred. Computing similarity matrix in chunks...")
        num_reference = reference_embeds.shape[0]
        num_query = query_embeds.shape[0]
        similarity = torch.zeros((num_query, num_reference), device='cpu')

        for i in range(0, num_reference, chunk_size):
            end_i = min(i + chunk_size, num_reference)
            ref_chunk = reference_embeds[i:end_i]

            for j in range(0, num_query, chunk_size):
                end_j = min(j + chunk_size, num_query)
                query_chunk = query_embeds[j:end_j]
                chunk_sim = torch.nn.functional.cosine_similarity(
                    ref_chunk.unsqueeze(0),
                    query_chunk.unsqueeze(1),
                    dim=2,
                )
                similarity[j:end_j, i:end_i] = chunk_sim.cpu()
                torch.cuda.empty_cache()

        return similarity.to(device)

def gpu_num_to_device(gpu_num):
    return f'cuda:{gpu_num}' if torch.cuda.is_available() and gpu_num >= 0 else 'cpu'

def load_model_data_and_dataset(dataset_flag, model_flags, gpus, seed):
    devices = [gpu_num_to_device(g) for g in gpus]  
    models = [load_model(f, devices[i % len(devices)]) for i, f in enumerate(model_flags)]
    dataset = create_dataset(dataset_flag, model=models[0], device=devices[0], seed=seed)
    embeddings = []
    for i, f in enumerate(model_flags):
        fnm = EMBEDDING_FNM.format(f, dataset_flag, f)
        embeddings.append(get_embeddings(fnm, dataset.label_texts, devices[i % len(devices)],
                                         dataset_flag, models[i]))
    model_data = [
        (m, e, devices[i % len(devices)]) for i, (m, e) in enumerate(zip(models, embeddings))
    ]
    return model_data, dataset

def print_results(ranks, losses, model_flag):
    if type(ranks) == list:
        ranks = np.concatenate(ranks)
    if type(losses) == list:
        losses = np.concatenate(losses)

    top1 = f'{(ranks < 1).mean():.2f}'
    top5 = f'{(ranks < 5).mean():.2f}'
    mean = f'{np.mean(losses):.4f}'
    stddev = f'{np.std(losses):.4f}'
    print(f'{model_flag},{top1},{top5},{mean},{stddev}')

def save_batch_images(input_path,output_dir):
    image_tensor=torch.tensor(np.load(input_path))
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(tqdm(image_tensor)):
        save_image(torch.squeeze(unnorm(img)), os.path.join(output_dir, f'image_{i}.png'))

def _normalize_rows(x, eps=1e-12):
    norms = np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)
    return x / norms


def cal_centroid(embedding, eps=1e-12):
    normalized_embeddings = _normalize_rows(embedding, eps=eps)
    return np.mean(normalized_embeddings, axis=0)


def cal_geometric_median(embedding, max_iter=200, tol=1e-6, eps=1e-12):
    x = _normalize_rows(embedding, eps=eps)
    y = np.mean(x, axis=0)
    y = y / np.clip(np.linalg.norm(y), eps, None)

    for _ in range(max_iter):
        diff = x - y
        dist = np.linalg.norm(diff, axis=1)
        if np.any(dist < eps):
            y_new = x[np.argmin(dist)]
        else:
            w = 1.0 / np.clip(dist, eps, None)
            y_new = (w[:, None] * x).sum(axis=0) / w.sum()

        if np.linalg.norm(y_new - y) < tol:
            y = y_new
            break
        y = y_new

    return y / np.clip(np.linalg.norm(y), eps, None)


def cal_trimmed_centroid(embedding, trim_ratio=0.1, eps=1e-12):
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must be in [0.0, 0.5).")

    x = _normalize_rows(embedding, eps=eps)
    c0 = np.mean(x, axis=0)
    c0 = c0 / np.clip(np.linalg.norm(c0), eps, None)
    dist = 1.0 - (x @ c0)

    n = x.shape[0]
    k_keep = max(1, int(round((1.0 - trim_ratio) * n)))
    keep_idx = np.argpartition(dist, k_keep - 1)[:k_keep]
    c = np.mean(x[keep_idx], axis=0)
    return c / np.clip(np.linalg.norm(c), eps, None)


def cal_medoid(embedding, sample_size=None, random_state=0, eps=1e-12):
    x = _normalize_rows(embedding, eps=eps)
    n = x.shape[0]

    if sample_size is not None and sample_size < n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        xc = x[idx]
    else:
        idx = None
        xc = x

    sim = xc @ xc.T
    medoid_local = np.argmax(sim.sum(axis=1))
    if idx is None:
        return xc[medoid_local]
    return x[idx[medoid_local]]


def compute_center(embedding_subset, center_type, cfg):
    center_type = str(center_type).lower()
    if center_type in ("centroid", "mean"):
        return cal_centroid(embedding_subset)
    if center_type == "geometric_median":
        return cal_geometric_median(embedding_subset)
    if center_type == "trimmed_centroid":
        trim_ratio = float(getattr(cfg, "trim_ratio", 0.1))
        return cal_trimmed_centroid(embedding_subset, trim_ratio=trim_ratio)
    if center_type in ("medoid", "mediod"):
        medoid_approx_size = getattr(cfg, "medoid_approx_size", None)
        medoid_approx_size = None if medoid_approx_size is None else int(medoid_approx_size)
        return cal_medoid(
            embedding_subset,
            sample_size=medoid_approx_size,
            random_state=int(getattr(cfg, "seed", 0)),
        )
    raise ValueError(
        f"Unsupported centroid_method '{center_type}'. "
        "Choose from: centroid, geometric_median, trimmed_centroid, medoid."
    )


def generate_or_load_centroids(cfg):
    if getattr(cfg, "centroid_embeds_path", None) is not None:
        return torch.tensor(np.load(cfg.centroid_embeds_path), dtype=torch.float32)

    query_embedding_path = (
        f"outputs/embeddings/{cfg.model_flag}/"
        f"{cfg.dataset_flag}_{cfg.model_flag}_query_embeddings.npy"
    )
    query_embedding = np.load(query_embedding_path)
    num_queries = query_embedding.shape[0]
    num_centroids = int(getattr(cfg, "number_images", num_queries))
    center_type = str(getattr(cfg, "centroid_method", "centroid")).lower()
    sample_size = int(getattr(cfg, "sample_size", min(100, num_queries)))
    sample_size = max(1, min(sample_size, num_queries))

    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))
    centers = []
    for _ in range(num_centroids):
        random_indices = rng.choice(num_queries, size=sample_size, replace=False)
        embedding_subset = query_embedding[random_indices]
        center = compute_center(embedding_subset, center_type, cfg)
        centers.append(center.astype(np.float32))

    centroids = torch.tensor(np.stack(centers, axis=0), dtype=torch.float32)
    print(
        f"Generated {num_centroids} centroids using method='{center_type}', "
        f"sample_size={sample_size} from {query_embedding_path}."
    )
    return centroids


def calculate_recall_and_rank(similarity, k_values, gt_idx, adv_idx):
    """Calculate recall@k and median rank for ground-truth and adversarial targets."""
    num_queries = similarity.shape[0]
    top_k_indices = np.argsort(-similarity, axis=1)
    gt_recalls = {}
    adv_recalls = {}
    gt_ranks = []
    adv_ranks = []

    for i in range(num_queries):
        gt_rank = np.where(top_k_indices[i] == gt_idx[i])[0][0] + 1
        adv_rank = np.where(top_k_indices[i] == adv_idx[i])[0][0] + 1
        gt_ranks.append(gt_rank)
        adv_ranks.append(adv_rank)

    for k in k_values:
        gt_correct = 0
        adv_correct = 0
        for i in range(num_queries):
            if gt_idx[i] in top_k_indices[i, :k]:
                gt_correct += 1
            if adv_idx[i] in top_k_indices[i, :k]:
                adv_correct += 1
        gt_recalls[k] = gt_correct / num_queries
        adv_recalls[k] = adv_correct / num_queries

    gt_median_rank = np.median(gt_ranks)
    adv_median_rank = np.median(adv_ranks)
    return gt_recalls, gt_median_rank, adv_recalls, adv_median_rank


def compute_recall_rank_stats(results):
    """Compute average and standard deviation for recall and rank values."""
    k_values = list(next(iter(results.values()))['gt_recalls'].keys())
    gt_recalls = {k: [] for k in k_values}
    adv_recalls = {k: [] for k in k_values}
    gt_ranks, adv_ranks = [], []

    for adv in results.values():
        for k in k_values:
            gt_recalls[k].append(adv['gt_recalls'][k])
            adv_recalls[k].append(adv['adv_recalls'][k])
        gt_ranks.append(adv['gt_median_rank'])
        adv_ranks.append(adv['adv_median_rank'])

    def format_values(data):
        avg = {k: round(np.mean(data[k]), 3) for k in k_values}
        std = {k: round(np.std(data[k]), 3) for k in k_values}
        return avg, std

    gt_recall_avg, gt_recall_std = format_values(gt_recalls)
    adv_recall_avg, adv_recall_std = format_values(adv_recalls)

    return {
        "gt_recall_avg": gt_recall_avg,
        "gt_recall_std": gt_recall_std,
        "gt_rank_avg": round(np.mean(gt_ranks), 3),
        "gt_rank_std": round(np.std(gt_ranks), 3),
        "adv_recall_avg": adv_recall_avg,
        "adv_recall_std": adv_recall_std,
        "adv_rank_avg": round(np.mean(adv_ranks), 3),
        "adv_rank_std": round(np.std(adv_ranks), 3),
    }


def evaluate_retrieval_and_save(dataset_flag, model_flag, output_dir, adv_test_similarity_matrix, max_adv_to_eval=100):
    """Evaluate adversarial retrieval metrics and save a text report."""
    gallery_embedding_path = f"outputs/embeddings/{model_flag}/{dataset_flag}_{model_flag}_gallery_embeddings.npy"
    query_embedding_path = f"outputs/embeddings/{model_flag}/{dataset_flag}_{model_flag}_query_embeddings.npy"
    query_embedding = np.load(query_embedding_path)
    gallery_embedding = np.load(gallery_embedding_path)

    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    gallery_norm = np.linalg.norm(gallery_embedding, axis=1, keepdims=True)
    query_norm = np.where(query_norm == 0, 1e-12, query_norm)
    gallery_norm = np.where(gallery_norm == 0, 1e-12, gallery_norm)
    similarity = (query_embedding / query_norm) @ (gallery_embedding / gallery_norm).T

    if dataset_flag == 'mscoco':
        if len(query_embedding) % 5 != 0:
            raise ValueError("Expected mscoco query embeddings to be a multiple of 5.")
        gt_idx = np.repeat(np.arange(len(query_embedding) // 5), 5)
        k_values = [1, 5, 10]
    elif dataset_flag == 'audiocaps':
        gt_idx = np.arange(len(query_embedding))
        k_values = [1, 5, 10]
    elif dataset_flag == 'cub_200':
        gt_idx = np.load('outputs/cub_200/gt_idx.npy')
        k_values = [1, 3, 5, 10]
    else:
        raise ValueError(f"Unsupported dataset_flag for evaluation: {dataset_flag}")

    results = {}
    num_adv_to_eval = min(max_adv_to_eval, adv_test_similarity_matrix.shape[0])
    adv_idx = np.repeat(len(gallery_embedding), len(query_embedding))

    for i, adv in enumerate(tqdm(adv_test_similarity_matrix[:num_adv_to_eval], desc="Evaluating adversarial retrieval")):
        test_data_temp = np.column_stack((similarity, adv.flatten()))
        gt_recalls, gt_median_rank, adv_recalls, adv_median_rank = calculate_recall_and_rank(
            test_data_temp, k_values, gt_idx, adv_idx
        )
        results[f'adv_{i}'] = {
            "gt_recalls": gt_recalls,
            "gt_median_rank": gt_median_rank,
            "adv_recalls": adv_recalls,
            "adv_median_rank": adv_median_rank,
        }

    stats = compute_recall_rank_stats(results)
    eval_output_path = Path(output_dir) / "evaluation_results.txt"
    with eval_output_path.open("w", encoding="utf-8") as f:
        for key, value in stats.items():
            line = f"{key}: {value}"
            print(line)
            f.write(line + "\n")

    return stats, str(eval_output_path)