# coding=utf-8
import argparse
import clip
import numpy as np
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import patch_classify, post_process, compute_F1, evaluation
from module import key_smoothing
from dataloader import NumpyDataset, ResizeToPatchSizeDivisible
from clip_text import class_names_voc, BACKGROUND_CATEGORY_VOC, class_names_coco, BACKGROUND_CATEGORY_COCO

"""
Example:
python MaskCLIP.py --dataset coco2014
python MaskCLIP.py --dataset voc2012
"""

# parse arguments
parser = argparse.ArgumentParser(description="MaskCLIP")
parser.add_argument("--dataset", type=str, choices=["voc2012", "coco2014"], default="coco2014")
parser.add_argument("--keep-resolution", action="store_true", help="Keep original resolution if set.")
args = parser.parse_args()
print(args)
device = torch.device("cuda:0")

# data path
if args.dataset == "voc2012":
    img_root = "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
    image_file = "imageset/voc2012/formatted_val_images.npy"
    full_label_file = "imageset/voc2012/formatted_val_labels.npy"
    class_names = class_names_voc + BACKGROUND_CATEGORY_VOC
    NUM_CLASSES = len(class_names_voc)
elif args.dataset == "coco2014":
    img_root = "datasets/coco2014"
    image_file = "imageset/coco2014/formatted_val_images.npy"
    full_label_file = "imageset/coco2014/formatted_val_labels.npy"
    class_names = class_names_coco + BACKGROUND_CATEGORY_COCO
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

# define hook
hook_handles = []
attention_weights = []
last_features = None
penultimate_features = None

def penultimate_hook_fn(module, args, output):
    assert isinstance(output, torch.Tensor)
    global penultimate_features
    penultimate_features = output

def attention_hook_fn(module, args, output):
    assert isinstance(module, nn.MultiheadAttention)
    attention_weights.append(output[1]) # [N, L, L]

def input_hook_fn(module, args, kwargs):
    assert isinstance(module, nn.MultiheadAttention)
    x = args[0]
    # attn_mask = (1 - torch.eye(x.shape[0])) * -torch.inf # float(-1000)
    attn_mask = torch.empty([x.shape[0], x.shape[0]]).fill_(-torch.inf)
    attn_mask.fill_diagonal_(0)
    attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
    # print(attn_mask)
    kwargs['attn_mask'] = attn_mask

def output_hook_fn(module, args, output):
    assert isinstance(output, torch.Tensor)
    global last_features
    last_features = output

pattern_attn = re.compile(r'^transformer\.resblocks\.\d+\.attn$')
pattern_mlp = re.compile(r'^transformer\.resblocks\.\d+\.mlp$')

for name, module in model.visual.named_modules(remove_duplicate=False):
    if pattern_attn.match(name):
        hook = module.register_forward_hook(attention_hook_fn)
        hook_handles.append(hook)
    if name == "transformer.resblocks.11.attn":
        hook = module.register_forward_pre_hook(input_hook_fn, with_kwargs=True)
        hook_handles.append(hook)
    if name == "transformer":
        hook = module.register_forward_hook(output_hook_fn)
        hook_handles.append(hook)
    if name == "transformer.resblocks.10":
        hook = module.register_forward_hook(penultimate_hook_fn)
        hook_handles.append(hook)


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
else:
    # 224*224
    dataset = NumpyDataset(img_root, image_list, full_label_list, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# inference
pred_logits = []
label_vectors = []
with torch.no_grad():
    for image, label in tqdm(dataloader):
        # preprocess
        attention_weights = []
        last_features = None
        penultimate_features = None
        image = image.to(device)
        h, w = image.shape[-2:]
        # infer
        image_features = model.encode_image(image, h, w)
        aligned_features = post_process(model, last_features, batch_first=False, only_class=False)
        logits = patch_classify(aligned_features, text_features, logit_scale=logit_scale)
        # refine
        logits = key_smoothing(logits, model, penultimate_features)
        # predict
        logits = logits.squeeze()
        logits_max = torch.max(logits, dim=0)[0]
        logits_max = logits_max[:NUM_CLASSES]
        pred_logits.append(logits_max.cpu())
        label_vectors.append(label)
        
pred_logits = torch.stack(pred_logits, dim=0)
label_vectors = torch.cat(label_vectors, dim=0)

# evaluation
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

# clear hooks
for hook in hook_handles:
    hook.remove()