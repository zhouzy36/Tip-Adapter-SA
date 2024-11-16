# coding=utf-8
# author: Ziyang Zhou
import argparse
import clip_surgery as clip
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from tqdm import tqdm
BICUBIC = InterpolationMode.BICUBIC

from utils import evaluation, compute_F1
from dataloader import NumpyDataset
from clip_text import class_names_voc, BACKGROUND_CATEGORY_VOC, class_names_coco, BACKGROUND_CATEGORY_COCO

"""
Example:
python CLIPSurgery.py --dataset coco2014
python CLIPSurgery.py --dataset voc2012
"""

# parse arguments
parser = argparse.ArgumentParser(description="CLIP Surgery")
parser.add_argument("--dataset", type=str, choices=["voc2012", "coco2014"], default="coco2014")
parser.add_argument("--image-size", type=int, default=224, choices=[224, 512], help="input image size. (default: 224)")
args = parser.parse_args()
print(args)
device = torch.device("cuda:0")

# data path
if args.dataset == "voc2012":
    img_root = "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
    image_file = "imageset/voc2012/formatted_val_images.npy"
    full_label_file = "imageset/voc2012/formatted_val_labels.npy"
    class_names = class_names_voc
    NUM_CLASSES = len(class_names_voc)
elif args.dataset == "coco2014":
    img_root = "datasets/coco2014"
    image_file = "imageset/coco2014/formatted_val_images.npy"
    full_label_file = "imageset/coco2014/formatted_val_labels.npy"
    class_names = class_names_coco
    NUM_CLASSES = len(class_names_coco)
else:
    raise NotImplementedError

image_list = np.load(image_file)
full_label_list = np.load(full_label_file)
print("Dataset:", args.dataset)
print("The number of classes in dataset:", NUM_CLASSES)
print("The number of classes in vocabulary:", len(class_names))

# model
patch_size = 16
model, _ = clip.load("CS-ViT-B/16", device=device)
model.eval()

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
dataset = NumpyDataset(img_root, image_list, full_label_list, transform=preprocess)
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

_ = evaluation(pred_logits, label_vectors)

# search best threshold
step = 100
best_F1 = 0
for thres in np.linspace(0, 1, step+1)[1:-1].tolist():
    F1, P, R = compute_F1(pred_logits.clone(), label_vectors.clone(),  mode_F1='overall', k_val=thres, use_relative=True)
    if F1 > best_F1:
        best_F1 = F1
        best_thres = thres
F1, P, R = compute_F1(pred_logits.clone(), label_vectors.clone(),  mode_F1='overall', k_val=best_thres, use_relative=True)
print(f"best threshold: {best_thres:.2f}, F1: {F1:.6f}, Precision: {P:.6f}, Recall: {R:.6f}")