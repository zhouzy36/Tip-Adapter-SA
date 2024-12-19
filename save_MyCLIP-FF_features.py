# coding=utf-8
import argparse
import clip
import math
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

from dataset import ResizeToPatchSizeDivisible
from module import double_mask_attention_refine, extract_class_specific_features
from utils import get_test_dataset, get_class_names, get_split_dataset, patch_classify, post_process, evaluate

"""
Examples:
python save_MyCLIP-FF_features.py --dataset coco2014 features/coco2014/MyCLIP_FF/val_all.pt
python save_MyCLIP-FF_features.py --dataset voc2012 features/voc2012/MyCLIP_FF/val_all.pt

python save_MyCLIP-FF_features.py --dataset coco2014 --split-file splits/coco2014/exp1/1shots_filtered.txt features/coco2014/MyCLIP_FF/exp1_1shots_filtered.pt
python save_MyCLIP-FF_features.py --dataset voc2012 --split-file splits/voc2012/exp1/1shots_filtered.txt features/voc2012/MyCLIP_FF/exp1_1shots_filtered.pt
"""

# parse arguments
parser = argparse.ArgumentParser(description="Save image features extracted by my Zero-Shot CLIP with multi-scale feature fusion.")
parser.add_argument("output_path", type=str, help="The output path.")
parser.add_argument("--split-file", type=str, help="The path to split file.")
parser.add_argument("--dataset", type=str, default="coco2014", choices=["voc2012", "coco2014", "LaSO"])
parser.add_argument("--num-scales", type=int, default=2, help="The number of input scales (default: 2).")
parser.add_argument("--max-reduction", type=float, default=2, help="The maximum downsampling rate of the image (default: 2).")
args = parser.parse_args()
print(args)
device = torch.device("cuda:0")

# check the existence of the file path
output_dir = os.path.dirname(args.output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

assert not args.split_file or os.path.exists(args.split_file)

# compute scale factor
scale_factor = 1 / math.pow(args.max_reduction, 1 / (args.num_scales - 1))

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
print("The number of classes in dataset:", NUM_CLASSES)
print("The number of classes in vocabulary:", len(class_names))

# classifier weights
with torch.no_grad():
    text_features = clip.encode_text_with_prompt_ensemble(model, class_names, device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True) # [num_classes, d]
print(text_features.shape, text_features.dtype)

# dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
adaptive_resize = ResizeToPatchSizeDivisible(patch_size)

if args.split_file:
    dataset = get_split_dataset(args.dataset, args.split_file, transform=transform)
else:
    dataset = get_test_dataset(args.dataset, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# data to save
all_features = []
all_label_vectors = []
all_zeroshot_logits = []

# extract features
FULL_LABEL_VECTOR = torch.ones(NUM_CLASSES).unsqueeze(0).to(device)
print("Start extracting features.")
with torch.no_grad():
    for image, label in tqdm(dataloader):
        # move image to device
        image = image.to(device)
        
        # perform multi-scale feature fusion
        fused_patch_features = None
        hr_attention_weights = None # record the attention weights of the image at the original resolution
        hr_h, hr_w = None, None # record the height and width of the image at the original resolution
        # process multi-scale inputs
        for ii in range(args.num_scales):
            # reset global variable
            attention_weights = []
            last_features = None
            penultimate_features = None
            # forward
            image = adaptive_resize(image)
            h, w = image.shape[-2:]
            model.encode_image(image, h=h, w=w)
            aligned_features = post_process(model, last_features, batch_first=False, only_class=False) # [1, L+1, D]
            patch_features = aligned_features[:, 1:, :] # [1, L, D]
            # perform feature fusion
            if ii == 0:
                hr_h, hr_w = h, w
                hr_attention_weights = attention_weights.copy()
                fused_patch_features = patch_features # [1, L, D]
            else:
                N, _, D = patch_features.shape
                feature_map = patch_features.reshape([N, h // patch_size, w // patch_size, D]).permute([0, 3, 1, 2])
                # upsample the low-resolution feature map back to the original resolution
                feature_map = F.interpolate(feature_map, 
                                            size=(hr_h // patch_size, hr_w // patch_size), 
                                            mode='bilinear', 
                                            align_corners=False)
                patch_features = feature_map.reshape([N, D, -1]).permute([0, 2, 1])
                fused_patch_features += patch_features
            # downsample image
            image = F.interpolate(image, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        fused_patch_features /= args.num_scales # average multi-scale feature maps

        # patch level classify
        logits = patch_classify(fused_patch_features, text_features, drop_first=False)
        
        # refine
        logits = double_mask_attention_refine(logits.squeeze(), hr_h, hr_w, patch_size, hr_attention_weights)
        
        # extract class-specific features
        if args.split_file:
            # extract features of labeled classes when address few-shot data
            class_specific_features = extract_class_specific_features(fused_patch_features.squeeze(), logits, label) # [num_labels, D]
        else:
            # extract features of all classes when address test data
            class_specific_features = extract_class_specific_features(fused_patch_features.squeeze(), logits, FULL_LABEL_VECTOR) # [num_classes, D]
        all_features.append(class_specific_features)
        
        # predict
        logits_max = torch.max(logits, dim=0)[0]
        logits_max = logits_max[:NUM_CLASSES] # [num_classes]

        # save
        all_label_vectors.append(label)
        all_zeroshot_logits.append(logits_max.cpu())
print("Finish extracting features.")

if args.split_file:
    all_features = torch.cat(all_features, dim=0) # [num_samples, D]
else:
    all_features = torch.stack(all_features, dim=0) # [num_samples, num_classes, D]
all_label_vectors = torch.cat(all_label_vectors, dim=0) # [num_samples, C]
all_zeroshot_logits = torch.stack(all_zeroshot_logits, dim=0) # [num_samples, C]
print(f"all features size: {all_features.shape}, all label vector size: {all_label_vectors.shape}, all logits size: {all_zeroshot_logits.shape}")

# save features
data = OrderedDict()
data["feats"] = all_features
data["logits"] = all_zeroshot_logits
data["labels"] = all_label_vectors
torch.save(data, args.output_path)
print(f"Save features to {args.output_path}")

# verify zero shot logits when extract features of test image
if not args.split_file:
    evaluate(all_zeroshot_logits, all_label_vectors)

# clear hooks
for hook in hook_handles:
    hook.remove()