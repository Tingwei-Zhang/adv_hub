import torch
import torch.nn as nn

from imagebind.models import imagebind_model
import imagebind.data as data

from AudioCLIP import AudioCLIP
import open_clip
from open_clip import tokenizer
from transformers import CLIPProcessor, CLIPModel


def load_model(model_flag, device):
    if model_flag == 'imagebind':
        model = ImageBindWrapper(imagebind_model.imagebind_huge(pretrained=True), device=device)
    elif model_flag == 'audioclip':
        model = AudioCLIPWrapper(AudioCLIP(pretrained=f'bpe/AudioCLIP-Full-Training.pt'))
    elif model_flag == 'audioclip_partial':
        model = AudioCLIPWrapper(AudioCLIP(pretrained=f'bpe/AudioCLIP-Partial-Training.pt'))
    elif model_flag == 'openclip':
        m, _, p = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir='bpe/')
        model = OpenCLIPWrapper(m, p)
    elif model_flag == 'openclip_rn50':
        m, _, p = open_clip.create_model_and_transforms('RN50', pretrained='openai', cache_dir='bpe/')
        model = OpenCLIPWrapper(m, p)
    elif model_flag == 'openclip_vit_b32':
        m, _, p = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', cache_dir='bpe/')
        model = OpenCLIPWrapper(m, p)
    elif 'openclip' in model_flag:
        _, backbone, pretrained = model_flag.split(';')
        m, _, p = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, cache_dir='bpe/')
        model = OpenCLIPWrapper(m, p)
    else:
        raise NotImplementedError()

    model.to(device)
    # if model_flag != 'audioclip':
    #     model.eval()
    model.eval()
    return model


class ImageBindWrapper(nn.Module):
    def __init__(self, model, device):
        super(ImageBindWrapper, self).__init__()
        self.model = model
        self.device = device
        self.flag = 'imagebind'
    
    def forward(self, X, modality, normalize=True):
        if modality == 'text':
            if isinstance(X, str):
                X = [X]
            X = data.load_and_transform_text(X, self.device)
            X = X.to(next(self.model.parameters()).device)
        return self.model.forward({modality: X}, normalize=normalize)[modality]


class AudioCLIPWrapper(nn.Module):
    def __init__(self, model):
        super(AudioCLIPWrapper, self).__init__()
        self.model = model
        self.flag = 'audioclip'
    
    def forward(self, X, modality, normalize=True):
        if modality == 'vision':
            modality = 'image'
        if modality == 'text':
            if isinstance(X, str):
                X = [X]
            X = [[i] for i in X]

        features = self.model.forward(**{modality: X}, normalize=normalize)[0][0]
        if modality == 'audio':
            return features[0]
        elif modality == 'image':
            return features[1]
        elif modality == 'text':
            return features[2]
        else:
            raise NotImplementedError()
        

class OpenCLIPWrapper(nn.Module):
    def __init__(self, model, preprocess):
        super(OpenCLIPWrapper, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.flag = 'openclip'
    
    def forward(self, X, modality, normalize=True):
        if modality == 'vision':
            modality = 'image'
        elif modality == 'text':
            X = tokenizer.tokenize(X)
        if next(self.model.parameters()).is_cuda:
            X = X.cuda()
        features = self.model.forward(**{modality: X})
        if modality == 'image':
            return features[0]
        elif modality == 'text':
            return features[1]
        else:
            raise NotImplementedError()

class HuggingFaceCLIPWrapper(nn.Module):
    def __init__(self, model_name, device):
        super(HuggingFaceCLIPWrapper, self).__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.flag = 'huggingface_clip'
        self.device = device

    
    def forward(self, X, modality, normalize=True):
        if modality == 'vision':
            # if normalize==False:
            #     X = unnorm(X)
            inputs = self.processor(images=X, return_tensors="pt", do_rescale=False).to(self.device)
            features = self.model.get_image_features(**inputs)
        elif modality == 'text':
            if isinstance(X, str):
                X = [X]
            inputs = self.processor(text=X, return_tensors="pt", padding=True)
            features = self.model.get_text_features(**inputs)
        else:
            raise NotImplementedError()

        return features
    

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