# coding=utf-8
import argparse
import clip
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import NumpyDataset, ResizeToPatchSizeDivisible
from clip_text import class_names_voc, BACKGROUND_CATEGORY_VOC, class_names_coco, BACKGROUND_CATEGORY_COCO
from utils import post_process, patch_classify, evaluation, compute_F1

"""
Example:
python ZSCLIP.py --dataset coco2014
python ZSCLIP.py --dataset voc2012
"""

# parse arguments
parser = argparse.ArgumentParser(description="Vanilla zero-shot CLIP")
parser.add_argument("--dataset", type=str, choices=["voc2012", "coco2014"], default="coco2014")
parser.add_argument("--keep-resolution", action="store_true", help="Keep image original resolution.")
parser.add_argument("--use-feature-map", action="store_true", help="Use patch features to classify.")
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
model_path = "pretrained_models/ViT-B-16.pt"
patch_size = 16
model, preprocess = clip.load(model_path, device)
model.eval()
logit_scale = model.logit_scale.exp().detach()

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
    dataset = NumpyDataset(img_root, image_list, full_label_list, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
else:
    # 224*224
    dataset = NumpyDataset(img_root, image_list, full_label_list, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# inference
pred_logits = []
label_vectors = []
if not args.use_feature_map:
    with torch.no_grad():
        for image, label in tqdm(dataloader):
            image = image.to(device)
            h, w = image.shape[-2:]
            # forward
            image_features = model.encode_image(image, h, w)
            # classify
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            logits = logit_scale * image_features @ text_features.t()
            logits = logits[:,:NUM_CLASSES].softmax(dim=1)
            # save logits and labels for evaluation
            pred_logits.append(logits.cpu())
            label_vectors.append(label)
else:
    hook_handles = []
    last_features = None

    def output_hook_fn(module, args, output):
        assert isinstance(output, torch.Tensor)
        global last_features
        last_features = output

    for name, module in model.visual.named_modules(remove_duplicate=False):
        if name == "transformer":
            hook = module.register_forward_hook(output_hook_fn)
            hook_handles.append(hook)

    with torch.no_grad():
        for image, label in tqdm(dataloader):
            image = image.to(device)
            h, w = image.shape[-2:]
            # forward
            last_features = None
            image_features = model.encode_image(image, h, w)
            # classify
            aligned_features = post_process(model, last_features, batch_first=False, only_class=False) # [L+1, N, D]
            logits = patch_classify(aligned_features, text_features, logit_scale=logit_scale) # [N, L, C]
            logits = torch.max(logits, dim=1)[0]
            # save logits and labels for evaluation
            pred_logits.append(logits.cpu())
            label_vectors.append(label)

    for hook in hook_handles:
        hook.remove()

pred_logits = torch.cat(pred_logits, dim=0)
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