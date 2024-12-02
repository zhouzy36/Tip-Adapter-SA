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

from dataset import ResizeToPatchSizeDivisible
from module import double_mask_attention_refine
from utils import get_test_dataset, get_class_names, patch_classify, post_process, compute_F1, evaluation

"""
Example:
python MyZSCLIP.py --dataset coco2014
python MyZSCLIP.py --dataset voc2012
"""

# parse arguments
parser = argparse.ArgumentParser(description="My Zero-Shot CLIP without feature fusion.")
parser.add_argument("--dataset", type=str, default="coco2014", choices=['voc2012', 'coco2014'])
args = parser.parse_args()
print(args)
device = torch.device("cuda:0")

# model
model_path = "pretrained_models/ViT-B-16.pt"
patch_size = 16
model, _ = clip.load(model_path, device)
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

# add hook
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

# get class names
class_names, NUM_CLASSES = get_class_names(args.dataset, include_background=True)
print("Dataset:", args.dataset)
print("The number of classes in dataset:", NUM_CLASSES)
print("The number of classes in vocabulary:", len(class_names))

# classifier weights
with torch.no_grad():
    text_features = clip.encode_text_with_prompt_ensemble(model, class_names, device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True) # [num_classes, d]
print(text_features.shape, text_features.dtype)

# dataloader
transform = transforms.Compose([
    ResizeToPatchSizeDivisible(patch_size),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
dataset = get_test_dataset(args.dataset, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# inference
pred_logits = []
label_vectors = []
with torch.no_grad():
    for image, label in tqdm(dataloader):
        image = image.to(device)
        # reset global variables
        attention_weights = []
        last_features = None
        penultimate_features = None
        # forward
        h, w = image.shape[-2:]
        class_feature = model.encode_image(image, h, w)
        aligned_features = post_process(model, last_features, batch_first=False, only_class=False) # [1, L+1, D]
        # patch level classify
        logits = patch_classify(aligned_features, text_features, logit_scale=logit_scale, drop_first=True)
        # refine
        logits = double_mask_attention_refine(logits.squeeze(), h, w, patch_size, attention_weights)
        # predict
        logits_max = torch.max(logits, dim=0)[0]
        logits_max = logits_max[:NUM_CLASSES]
        pred_logits.append(logits_max.cpu())
        label_vectors.append(label)
        
pred_logits = torch.stack(pred_logits, dim=0)
label_vectors = torch.cat(label_vectors, dim=0)

# evaluation
evaluation(pred_logits, label_vectors)

# search best threshold
step = 100
best_F1 = 0
for thres in np.linspace(0, 1, step+1)[1:-1].tolist():
    F1, P, R = compute_F1(pred_logits.clone(), label_vectors.clone(),  mode_F1='overall', k_val=thres, use_relative=True)
    if F1 > best_F1:
        best_F1 = F1
        best_thres = thres
F1, P, R = compute_F1(pred_logits.clone(), label_vectors.clone(),  mode_F1='overall', k_val=best_thres, use_relative=True)
print(f"best threshold: {best_thres:.2f}, F1: {F1:.6f}, Precision: {P:.6f}, Recall: {R:.6f}\n")

# clear hooks
for hook in hook_handles:
    hook.remove()