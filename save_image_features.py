# coding=utf-8
import argparse
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

from module import FeatureExtractor
from utils import get_test_dataset, get_split_dataset

"""
Examples:
python save_image_features.py --dataset ${dataset} --backbone ${backbone} --image-size ${size} features/${dataset}/${backbone}@${size}/val_all.pt

python save_image_features.py --split-file splits/${dataset}/exp${split}/${k}shots_filtered.txt --dataset ${dataset} --backbone ${backbone} --image-size ${size} features/${dataset}/${backbone}@${size}/exp${split}_${k}shots_filtered.pt
"""

# parse arguments
parser = argparse.ArgumentParser(description="Save image features with vision backbone.")
parser.add_argument("output_path", type=str, help="The output path.")
parser.add_argument("--split-file", type=str, help="The path to split file.")
parser.add_argument("--dataset", type=str, choices=["voc2012", "coco2014", "LaSO"], default="coco2014")
parser.add_argument("--backbone", type=str, choices=["resnet50", "resnet101", "inception_v3", "vit_b_16"], default="resnet50")
parser.add_argument("--image-size", type=int, choices=[224, 448], default=224)
args = parser.parse_args()
print(args)

# init device
device = torch.device("cuda:0")

# check the existence of the output path
output_dir = os.path.dirname(args.output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

assert not args.split_file or os.path.exists(args.split_file)

# dataloader
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
if args.split_file:
    dataset = get_split_dataset(args.dataset, args.split_file, transform=transform)
else:
    dataset = get_test_dataset(args.dataset, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# feature extractor
feature_extractor = FeatureExtractor(args.backbone, image_size=args.image_size, weights="IMAGENET1K_V1")
feature_extractor.to(device)
feature_extractor.eval()

# data to save
all_features = []
all_label_vectors = []

# extract features
print("Start extracting features.")
with torch.no_grad():
    for images, labels in tqdm(dataloader):
        # move data to device
        images = images.to(device)
        # forward
        feats = feature_extractor(images) # [B, D]
        # save logits and labels for evaluation
        all_features.append(feats.cpu())
        all_label_vectors.append(labels)
print("Finish extracting features.")

all_features = torch.cat(all_features, dim=0) # [num_samples, D]
all_label_vectors = torch.cat(all_label_vectors, dim=0) # [num_samples, C]
print(f"all features size: {all_features.shape}, all label vector size: {all_label_vectors.shape}")

# save features
data = OrderedDict()
data["feats"] = all_features
data["labels"] = all_label_vectors
torch.save(data, args.output_path)
print(f"Save features to {args.output_path}")