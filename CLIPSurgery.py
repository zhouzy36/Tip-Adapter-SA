# coding=utf-8
import argparse
import clip_surgery as clip
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from tqdm import tqdm
BICUBIC = InterpolationMode.BICUBIC

from utils import get_test_dataset, get_class_names, evaluate, search_best_threshold

"""
Example:
python CLIPSurgery.py --dataset coco2014
python CLIPSurgery.py --dataset voc2012
python CLIPSurgery.py --dataset LaSO
"""

# parse arguments
parser = argparse.ArgumentParser(description="CLIP Surgery")
parser.add_argument("--dataset", type=str, choices=["voc2012", "coco2014", "LaSO"], default="coco2014")
parser.add_argument("--image-size", type=int, default=224, choices=[224, 512], help="input image size. (default: 224)")
args = parser.parse_args()
print(args)
device = torch.device("cuda:0")

# model
patch_size = 16
model, _ = clip.load("CS-ViT-B/16", device=device)
model.eval()

# get class names
class_names, NUM_CLASSES = get_class_names(args.dataset)
print("Dataset:", args.dataset)
print("The number of classes in dataset:", NUM_CLASSES)
print("The number of classes in vocabulary:", len(class_names))

# classifier weights
with torch.no_grad():
    text_features = clip.encode_text_with_prompt_ensemble(model, class_names, device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True) # [num_classes, d]
print(text_features.shape, text_features.dtype)

# dataloader
preprocess = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size), interpolation=BICUBIC), 
    transforms.ToTensor(), 
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
dataset = get_test_dataset(args.dataset, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# inference
pred_logits = []
label_vectors = []
with torch.no_grad():
    for image, label in tqdm(dataloader):
        image = image.to(device)
        # Extract image features
        image_features = model.encode_image(image) # [1, L+1, D]
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # classify
        logits = clip.clip_feature_surgery(image_features, text_features) # [1, L+1, D]
        logits = logits[:, 1:, :].squeeze()
        logits_max = torch.max(logits, dim=0)[0] # [N, C]
        logits_max = logits_max[:NUM_CLASSES]
        # save logits and labels for evaluation
        pred_logits.append(logits_max.cpu())
        label_vectors.append(label.cpu())
pred_logits = torch.stack(pred_logits, dim=0)
label_vectors = torch.cat(label_vectors, dim=0)

evaluate(pred_logits, label_vectors)
search_best_threshold(pred_logits, label_vectors)