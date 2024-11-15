import os
import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets

import librosa
from tqdm import tqdm
import imagebind.data as data
from pycocotools.coco import COCO
import random

DATA_PATH = {
    'imagenet': '../adversarial_illusions/data/imagenet/',
    'audiocaps': '../adversarial_illusions/data/AudioCaps/',
    'audioset': '../adversarial_illusions/data/AudioCaps/',
    'mscoco': '../adversarial_illusions/data/coco/',
    'cub_200': '../adversarial_illusions/data/CUB_200_2011/',
}

TEMPLATES = {
    # 'imagenet': 'A photo of a {}.'
    'imagenet': '{}.'
}


def get_embeddings(embs_file, labels, device, dataset_flag, model=None, batch_size=1000, device_override=False):
    if embs_file is not None and os.path.isfile(embs_file):
        print(f'Reading label embeddings from {embs_file}...')
        return torch.tensor(np.load(embs_file)).to(device)
    
    print(f'No label embeddings found. Generating...')
    embs = []
    if dataset_flag == 'cub_200':
        for i in tqdm(range(int(np.ceil(len(labels) / batch_size)))):
            batch_paths = labels[i * batch_size:(i + 1) * batch_size]
            batch_images = [imagenet_loader(path, model, device) for path in batch_paths]
            batch_images = torch.cat(batch_images, dim=0)
            with torch.no_grad():
                embs_batch = model.cuda().forward(batch_images, 'vision', normalize=False)
                embs.append(embs_batch.to('cuda'))
    else:
        if dataset_flag in TEMPLATES:
            labs = np.array([TEMPLATES[dataset_flag].format(labels[i].split(',')[0]) for i in range(len(labels))])
        else:
            labs = np.array(labels)
        for i in tqdm(range(int(np.ceil(len(labs) / batch_size)))):
            batch = labs[i*batch_size:(i+1)*batch_size]
            with torch.no_grad():
                embs_batch = model.cuda().forward(batch, 'text', normalize=False)
                embs.append(embs_batch.to('cuda'))

    if not device_override:
        model.to(device)

    if embs_file is not None:
        print(f'Writing label embeddings to {embs_file}...')
        folder_path = os.path.dirname(embs_file)
        os.makedirs(folder_path, exist_ok=True)
        # Move embeddings to CPU before saving
        np.save(embs_file, torch.cat(embs).cpu())
    return torch.cat(embs).to(device)



class WrappedImageNetDataset(Dataset):
    def __init__(
        self, dataset, labels, model,
        mapping=None, device='cpu', seed=0,
        embs_input=None, embedding_batch_size=250,
        embedding_override=False
    ):
        self.dataset = dataset
        self.seed = seed
        self.model = model
        np.random.seed(seed=self.seed)
        self.mapping = mapping if mapping is not None else np.random.permutation(len(dataset))
        self.device = device
        self.embs_file = embs_input
        self.label_texts = labels
        if self.embs_file is not None:
            self.labels = get_embeddings(self.embs_file, self.label_texts, self.device, 'imagenet',
                                         self.model, embedding_batch_size, embedding_override)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y_orig_id = self.dataset[idx]
        gt, y_id = self.dataset[self.mapping[idx]]
        if self.embs_file is not None:
            y = self.labels[y_id].to(self.device)
            return torch.squeeze(x), torch.squeeze(y), torch.squeeze(gt), y_id, y_orig_id
        return torch.squeeze(x), torch.squeeze(gt), y_id, y_orig_id


class WrappedAudioCapsDataset(Dataset):
    def __init__(
        self, dataset, model,
        mapping=None, device='cpu', seed=0,
        embs_input=None, embedding_batch_size=250,
        embedding_override=False
    ):
        self.dataset = dataset
        unique_captions = {}
        for data in self.dataset:
            audio, caption = data
            if caption not in unique_captions:
                unique_captions[caption] = audio

        # Convert the dictionary back to a list of tuples
        self.dataset = [(audio, caption) for caption, audio in unique_captions.items()]

        self.seed = seed
        self.model = model
        np.random.seed(seed=self.seed)
        self.mapping = mapping if mapping is not None else np.random.permutation(len(self.dataset))
        self.device = device
        self.embs_file = embs_input
        self.label_texts = list(dict.fromkeys([y for _, y in dataset]))
        if self.embs_file is not None:
            self.labels = get_embeddings(self.embs_file, self.label_texts, self.device, 'AudioCaps',
                                         self.model, embedding_batch_size, embedding_override)
        self.lab_to_id = {l: i for i, l in enumerate(self.label_texts)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y_orig = self.dataset[idx]
        gt, y_str = self.dataset[self.mapping[idx]]
        y_orig_id, y_str_id = self.lab_to_id[y_orig], self.lab_to_id[y_str]
        if self.model.flag == 'imagebind':
            x = torch.squeeze(x)[:, None, :, :]
            gt = torch.squeeze(gt)[:, None, :, :]
        x = (0.0001 * torch.randn_like(x)) + x.detach()
        if self.embs_file is not None:
            y = self.labels[y_str_id].to(self.device)
            return torch.squeeze(x), torch.squeeze(y), gt, y_str_id, y_orig_id
        return x, gt, y_str_id, y_orig_id


class AudioDataset(Dataset):
    def __init__(self, audio_dir, split_file, extension='wav', device='cpu', model_flag='imagebind'):
        self.audio_files = glob.glob(f'{audio_dir}*.{extension}')
        self.split = pd.read_csv(split_file, index_col='youtube_id')[['caption']]
        self.device = device
        self.model_flag = model_flag
        assert len(self.audio_files) > 0
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        path = self.audio_files[idx]
        if self.model_flag == 'imagebind':
            X = data.load_and_transform_audio_data([path], self.device)
        elif self.model_flag == 'audioclip':
            X = librosa.load(path, sr=44100, dtype=np.float32)[0]
            X = torch.tensor(X).to(self.device)
        y = self.split.loc[self.get_id(path)].iloc[-1].item()
        return X, y

    def get_id(self, path):
        return path.split('/')[-1].split('.')[0]


def imagenet_loader(path, model, device='cpu'):
    if model.flag == 'imagebind' or model.flag == 'audioclip':
        return data.load_and_transform_vision_data([path], device)
    elif model.flag == 'openclip':
        image_outputs = []
        with open(path, 'rb') as fopen:
            image = Image.open(fopen).convert('RGB')

        image = model.preprocess(image).to(device)
        image_outputs.append(image)
        return torch.stack(image_outputs, dim=0)
    else:
        raise NotImplementedError()


# Updated WrappedMSCOCODataset class
class WrappedMSCOCODataset(Dataset):
    def __init__(
        self, dataset, model,
        mapping=None, device='cpu', seed=0,
        embs_input=None, embedding_batch_size=1000,
        embedding_override=False
    ):
        self.dataset = dataset  # List of (image_path, caption_ids) tuples
        self.seed = seed
        self.model = model
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.mapping = mapping if mapping is not None else np.random.permutation(len(self.dataset))
        self.device = device
        self.embs_file = embs_input
        # Collect all unique captions
        self.caption_texts = []
        self.caption_ids = []
        for _, captions in self.dataset:
            for caption_id, caption_text in captions:
                self.caption_texts.append(caption_text)
                self.caption_ids.append(caption_id)
        if self.embs_file is not None:
            self.labels = get_embeddings(
                self.embs_file, self.caption_texts, self.device, 'mscoco',
                self.model, embedding_batch_size, embedding_override
            )
            self.caption_id_to_emb = {cid: emb for cid, emb in zip(self.caption_ids, self.labels)}
        # Map image indices to their corresponding caption IDs
        self.image_idx_to_caption_ids = [
            [caption_id for caption_id, _ in captions] for _, captions in self.dataset
        ]
        self.label_texts= self.caption_texts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, captions = self.dataset[idx]
        gt_img_path, captions_gt = self.dataset[self.mapping[idx]]

        # Load and preprocess images using imagenet_loader
        x = imagenet_loader(img_path, self.model, self.device)
        gt = imagenet_loader(gt_img_path, self.model, self.device)

        # Randomly select one caption ID for this image
        caption_ids = self.image_idx_to_caption_ids[idx]
        selected_caption_id = random.choice(caption_ids)

        # Randomly select one caption ID for gt image
        caption_ids_gt = self.image_idx_to_caption_ids[self.mapping[idx]]
        selected_caption_id_gt = random.choice(caption_ids_gt)

        return torch.squeeze(x), torch.squeeze(gt), selected_caption_id_gt, selected_caption_id

class WrappedCUB200Dataset(Dataset):
    def __init__(
        self, data_dir, model,
        mapping=None, device='cpu', seed=0,
        embs_input=None, embedding_batch_size=1000,
        embedding_override=False,
        split='test'  # Add a parameter to specify the split
    ):
        self.data_dir = data_dir
        self.device = device
        self.model = model
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.mapping = mapping

        # Read images.txt
        images_txt = os.path.join(self.data_dir, 'images.txt')
        with open(images_txt, 'r') as f:
            lines = f.readlines()
        image_id_to_name = {}
        for line in lines:
            image_id, image_name = line.strip().split()
            image_id_to_name[int(image_id)] = image_name

        # Read image_class_labels.txt
        labels_txt = os.path.join(self.data_dir, 'image_class_labels.txt')
        with open(labels_txt, 'r') as f:
            lines = f.readlines()
        image_id_to_class_id = {}
        for line in lines:
            image_id, class_id = line.strip().split()
            image_id_to_class_id[int(image_id)] = int(class_id)

        # Read train_test_split.txt
        split_txt = os.path.join(self.data_dir, 'train_test_split.txt')
        with open(split_txt, 'r') as f:
            lines = f.readlines()
        image_id_to_is_train = {}
        for line in lines:
            image_id, is_train = line.strip().split()
            image_id_to_is_train[int(image_id)] = int(is_train)

        # Filter images based on split
        self.samples = []
        for image_id, image_name in image_id_to_name.items():
            if (split == 'train' and image_id_to_is_train[image_id] == 1) or (split == 'test' and image_id_to_is_train[image_id] == 0):
                class_id = image_id_to_class_id[image_id]
                image_path = os.path.join(self.data_dir, 'images', image_name)
                self.samples.append((image_id, image_path, class_id))

        # Sort the samples by class_id to ensure that images are in the same class in order
        self.samples.sort(key=lambda x: x[2])
        
        # Create a list of class ids
        class_ids = sorted(set([class_id for _, _, class_id in self.samples]))

        # Randomly select one image per class as the representative image
        self.class_representatives = {}
        remaining_samples = []
        representative_samples = []
        self.texts = []  # To store the text of all representative images as strings

        for class_id in class_ids:
            class_samples = [sample for sample in self.samples if sample[2] == class_id]
            representative_sample = random.choice(class_samples)
            self.class_representatives[class_id] = representative_sample
            representative_samples.append(representative_sample)  # Collect representative samples (tuple)
            self.texts.append(representative_sample[1])  # Store representative image path as text in `self.texts`
            # Remove the representative image from the dataset
            remaining_samples.extend([sample for sample in class_samples if sample != representative_sample])

        # Update samples to include representative images and remaining samples separately
        self.samples = representative_samples
        self.label_texts = [sample[1] for sample in remaining_samples]

        # Generate embeddings for representative images
        self.embs_file = embs_input
        if self.embs_file is not None:
            self.labels = get_embeddings(
                self.embs_file, self.label_texts, self.device, 'cub_200',
                self.model, embedding_batch_size, embedding_override
            )

        if mapping is None:
            self.mapping = np.random.permutation(len(self.samples))
        else:
            self.mapping = mapping

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, image_path, class_id = self.samples[idx]
        x = imagenet_loader(image_path, self.model, self.device)

        # Get ground truth image and label
        gt_idx = self.mapping[idx]
        gt_image_id, gt_image_path, gt_class_id = self.samples[gt_idx]
        gt = imagenet_loader(gt_image_path, self.model, self.device)

        # Get representative embedding for the class of gt image
        y_id = int(self.class_representatives[gt_class_id][2])

        return torch.squeeze(x), torch.squeeze(gt), y_id, int(class_id)
    

def create_dataset(dataset_flag, model=None, mapping=None, device='cpu', embs_input=None, seed=0):
    assert model is not None
    if dataset_flag == 'imagenet':
        loader = lambda p: imagenet_loader(p, model, device)
        imagenet = datasets.ImageNet(DATA_PATH[dataset_flag], split='val', loader=loader)
        with open(DATA_PATH[dataset_flag] + 'imagenet1000_clsidx_to_labels.txt') as f:
            labels = eval(f.read().replace('\n', ''))
        return WrappedImageNetDataset(imagenet, labels, model, mapping, device, seed, embs_input)
    elif dataset_flag == 'audiocaps':
        audiocaps = AudioDataset(DATA_PATH[dataset_flag] + 'raw/',
                                 DATA_PATH[dataset_flag] + 'splits/retrieval_test.csv',
                                 'wav',
                                 model_flag=model.flag)
        return WrappedAudioCapsDataset(audiocaps, model, mapping, device, seed, embs_input)
    elif dataset_flag == 'audioset':
        audioset = AudioDataset(DATA_PATH[dataset_flag] + 'raw/',
                                DATA_PATH[dataset_flag] + 'splits/classification_test.csv',
                                'wav',
                                model_flag=model.flag)
        return WrappedAudioCapsDataset(audioset, model, mapping, device, seed, embs_input)
    elif dataset_flag == 'mscoco':
        # Initialize COCO API
        annotations_path = os.path.join(DATA_PATH[dataset_flag], 'annotations/captions_val2017_modified.json')
        images_path = os.path.join(DATA_PATH[dataset_flag], 'images/val2017/')
        coco = COCO(annotations_path)
        img_ids = sorted(coco.getImgIds())
        imgs = coco.loadImgs(img_ids)

        # Create dataset as a list of (image_path, captions) tuples
        dataset = []
        for img_info in tqdm(imgs, desc="Preparing MSCOCO dataset"):
            img_id = img_info['id']
            img_file = img_info['file_name']
            img_path = os.path.join(images_path, img_file)
            # Get captions for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            captions = [(ann['id'], ann['caption']) for ann in anns]
            dataset.append((img_path, captions))

        return WrappedMSCOCODataset(dataset, model, mapping, device, seed, embs_input)
    elif dataset_flag == 'cub_200':
        data_dir = DATA_PATH[dataset_flag]
        cub_dataset = WrappedCUB200Dataset(
            data_dir, model, mapping, device, seed, embs_input, split='test')
        return cub_dataset
    else:
        raise NotImplementedError
