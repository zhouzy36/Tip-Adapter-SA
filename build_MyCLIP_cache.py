# coding=utf-8
import argparse
import clip
import os
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

from dataset import ResizeToPatchSizeDivisible
from module import double_mask_attention_refine, extract_class_specific_features
from utils import get_split_dataset, get_class_names, patch_classify, post_process

"""
Example:
python build_MyCLIP_cache.py --split-file SPLIT_FILE_PATH --dataset coco2014 OUTPUT_CACHE_PATH
python build_MyCLIP_cache.py --split-file SPLIT_FILE_PATH --dataset voc2012 OUTPUT_CACHE_PATH
python build_MyCLIP_cache.py --split-file SPLIT_FILE_PATH --dataset LaSO --prototype OUTPUT_CACHE_PATH
"""

# arguments
parser = argparse.ArgumentParser(description="Build and save cache model with MyCLIP features.")
parser.add_argument("cache_path", type=str, help="Cache output path.")
parser.add_argument("--split-file", type=str, required=True, help="The path of split file.")
parser.add_argument("--dataset", type=str, default="coco2014", choices=["coco2014", "voc2012", "LaSO"])
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
model, _ = clip.load(model_path, device)
model.eval()
logit_scale = model.logit_scale.exp().detach()
print(f"Load model from {model_path}")

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
dataset = get_split_dataset(args.dataset, args.split_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# pre-allocate keys and values
cache_keys = []
cache_values = []

# build cache model
print("Start building cache model.")
with torch.no_grad():
    for image, label in tqdm(dataloader):
        # move data to device
        image = image.to(device)

        # reset global variables
        attention_weights = []
        last_features = None
        penultimate_features = None

        # forward
        h, w = image.shape[-2:]
        class_feature = model.encode_image(image, h, w)
        aligned_features = post_process(model, last_features, batch_first=False, only_class=False) # [1, L+1, D]
        patch_features = aligned_features[:, 1:, :] # [1, L, D]

        # patch level classify
        logits = patch_classify(aligned_features, text_features, logit_scale=logit_scale, drop_first=True)

        # refine
        logits = double_mask_attention_refine(logits.squeeze(), h, w, patch_size, attention_weights)
        
        # extract class-specific features with shape [num_labels, D]
        class_specific_features = extract_class_specific_features(patch_features.squeeze(), logits, label)

        # update keys and values
        cache_keys.append(class_specific_features.cpu())
        cache_values.append(label)
print("Finish building cache model.")
cache_values = torch.cat(cache_values, dim=0)

# calculate prototype
if args.prototype:
    protos = []
    for cls_id in range(NUM_CLASSES):
        sample_ids = torch.nonzero(cache_values[:, cls_id]).flatten().tolist()
        cls_spec_feats = []
        for idx in sample_ids:
            feat_idx = cache_values[idx, :cls_id].sum().int().item()
            cls_spec_feats.append(cache_keys[idx][feat_idx])
        protos.append(torch.stack(cls_spec_feats, dim=0).mean(dim=0))
    cache_keys = torch.stack(protos, dim=0) # [num_classes, D]
    cache_values = torch.eye(NUM_CLASSES) # [num_classes, num_classes]
else:
    cache_keys = torch.cat(cache_keys, dim=0)

# normalize cache keys
cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
print(f"Cache keys shape: {cache_keys.shape}\nCache values shape: {cache_values.shape}")

# save cache model
cache = OrderedDict()
cache["keys"] = cache_keys
cache["values"] = cache_values
torch.save(cache, args.cache_path)
print(f"Save cache model to {args.cache_path}")

# clear hooks
for hook in hook_handles:
    hook.remove()