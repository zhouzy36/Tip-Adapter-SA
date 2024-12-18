# coding=utf-8
import argparse
import clip
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

from dataset import ResizeToPatchSizeDivisible
from utils import get_split_dataset, get_class_names

"""
Example:
python build_CLIP_cache.py --split-file SPLIT_FILE_PATH --dataset coco2014 OUTPUT_CACHE_PATH
python build_CLIP_cache.py --split-file SPLIT_FILE_PATH --dataset voc2012 OUTPUT_CACHE_PATH
python build_CLIP_cache.py --split-file SPLIT_FILE_PATH --dataset LaSO --prototype OUTPUT_CACHE_PATH
"""

# arguments
parser = argparse.ArgumentParser(description="Build and save cache model with CLIP features.")
parser.add_argument("cache_path", type=str, help="Cache output path.")
parser.add_argument("--split-file", type=str, required=True, help="The path of split file.")
parser.add_argument("--dataset", type=str, default="coco2014", choices=["coco2014", "voc2012", "LaSO"])
parser.add_argument("--keep-resolution", action="store_true", help="Keep image original resolution if set.")
parser.add_argument("--prototype", action="store_true", help="Save the class prototype as a cache key if set.")
args = parser.parse_args()
print(args)
assert args.dataset != "LaSO" or args.prototype, "If the experimental dataset is LaSO, prototype must be set to true."
device = torch.device("cuda:0")

# check the existence of the output path
output_dir = os.path.dirname(args.cache_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load model
model_path = "pretrained_models/ViT-B-16.pt"
patch_size = 16
model, preprocess = clip.load(model_path, device)
model.eval()
logit_scale = model.logit_scale.exp().detach()
print(f"Load model from {model_path}")

# get class names
_, NUM_CLASSES = get_class_names(args.dataset)
print("The number of classes in dataset:", NUM_CLASSES)

# dataloader
if args.keep_resolution:
    # original resolution
    transform = transforms.Compose([
        ResizeToPatchSizeDivisible(patch_size),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = get_split_dataset(args.dataset, args.split_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
else:
    # 224*224
    dataset = get_split_dataset(args.dataset, args.split_file, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# pre-allocate keys and values
cache_keys = []
cache_values = []

# build cache model
print("Start building cache model.")
with torch.no_grad():
    for image, label in tqdm(dataloader):
        # move data to device
        image = image.to(device)
        # forward
        h, w = image.shape[-2:]
        image_features = model.encode_image(image, h, w)
        # update keys and values
        cache_keys.append(image_features.cpu())
        cache_values.append(label)
print("Finish building cache model.")
cache_values = torch.cat(cache_values, dim=0) # [num_samples, num_classes]

# calculate prototype
if args.prototype:
    protos = []
    for cls_id in range(NUM_CLASSES):
        sample_ids = torch.nonzero(cache_values[:, cls_id]).flatten()
        protos.append(torch.mean(cache_keys[sample_ids, :], dim=0)) # [D]
    cache_keys = torch.stack(protos, dim=0) # [num_classes, D]
    cache_values = torch.eye(NUM_CLASSES) # [num_classes, num_classes]
else:
    cache_keys = torch.cat(cache_keys, dim=0) # [num_samples, D]

# normalize cache keys
cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
print(f"Cache keys shape: {cache_keys.shape}\nCache values shape: {cache_values.shape}")

# save cache model
cache = OrderedDict()
cache["keys"] = cache_keys
cache["values"] = cache_values
torch.save(cache, args.cache_path)
print(f"Save cache model to {args.cache_path}")