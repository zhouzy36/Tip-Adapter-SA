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
from utils import get_test_dataset, get_class_names, evaluation

"""
Example:
python save_CLIP_features.py features/voc2012/CLIP/val_all.pt --dataset voc2012
python save_CLIP_features.py features/coco2014/CLIP/val_all.pt --dataset coco2014
"""

# parse arguments
parser = argparse.ArgumentParser(description="Save CLIP features of test images.")
parser.add_argument("output_path", type=str, help="The output path.")
parser.add_argument("--dataset", type=str, choices=["voc2012", "coco2014"], default="coco2014")
parser.add_argument("--keep-resolution", action="store_true", help="Keep image original resolution if set.")
args = parser.parse_args()
print(args)
device = torch.device("cuda:0")

# check the existence of the output path
output_dir = os.path.dirname(args.output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# model
model_path = "pretrained_models/ViT-B-16.pt"
patch_size = 16
model, preprocess = clip.load(model_path, device)
model.eval()
logit_scale = model.logit_scale.exp().detach()

class_names, NUM_CLASSES = get_class_names(args.dataset)
print("The number of classes in dataset:", NUM_CLASSES)
print("The number of classes in vocabulary:", len(class_names))

# classifier weights
with torch.no_grad():
    text_features = clip.encode_text_with_prompt_ensemble(model, class_names, device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True) # [num_classes, d]
print(text_features.shape, text_features.dtype)

# dataloader
if args.keep_resolution:
    # original resolution
    transform = transforms.Compose([
        ResizeToPatchSizeDivisible(patch_size),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = get_test_dataset(args.dataset, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
else:
    # 224*224
    dataset = get_test_dataset(args.dataset, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# data to save
all_features = []
all_label_vectors = []
all_zeroshot_logits = []

# extract features
print("Start extracting features.")
with torch.no_grad():
    for image, label in tqdm(dataloader):
        image = image.to(device)
        h, w = image.shape[-2:]
        # forward
        image_features = model.encode_image(image, h, w)
        all_features.append(image_features.cpu())
        # classify
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        logits = logits[:, :NUM_CLASSES].softmax(dim=1)
        # save logits and labels for evaluation
        all_label_vectors.append(label)
        all_zeroshot_logits.append(logits.cpu())
print("Finish extracting features.")

all_features = torch.cat(all_features, dim=0) # [num_samples, D]
all_label_vectors = torch.cat(all_label_vectors, dim=0) # [num_samples, C]
all_zeroshot_logits = torch.cat(all_zeroshot_logits, dim=0) # [num_samples, C]
print(f"all features size: {all_features.shape}, all label vector size: {all_label_vectors.shape}")

# save features
data = OrderedDict()
data["feats"] = all_features
data["logits"] = all_zeroshot_logits
data["labels"] = all_label_vectors
torch.save(data, args.output_path)
print(f"Save features to {args.output_path}")

# verify zero shot logits
_ = evaluation(all_zeroshot_logits, all_label_vectors)