# coding=utf-8
# Copy from https://github.com/linyq2117/TagCLIP/ and modify
import argparse
import numpy as np
from PIL import Image
import os
import tag_clip as clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

from utils import scoremap2bbox, evaluate, search_best_threshold
from clip_text import class_names_voc, BACKGROUND_CATEGORY_VOC, class_names_coco, BACKGROUND_CATEGORY_COCO

"""
Example:
python TagCLIP.py --dataset coco2014
python TagCLIP.py --dataset voc2012
"""

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_resize(h, w):
    return Compose([
        Resize((h,w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def mask_attn(logits_coarse, logits, h, w, attn_weight):
    patch_size = 16
    candidate_cls_list = []
    logits_refined = logits.clone()
    
    logits_max = torch.max(logits, dim=0)[0]
        
    for tempid,tempv in enumerate(logits_max):
        if tempv > 0:
            candidate_cls_list.append(tempid)
    for ccls in candidate_cls_list:
        temp_logits = logits[:,ccls]
        temp_logits = temp_logits - temp_logits.min()
        temp_logits = temp_logits / temp_logits.max()
        mask = temp_logits
        mask = mask.reshape(h // patch_size, w // patch_size)
        
        box, cnt = scoremap2bbox(mask.detach().cpu().numpy(), threshold=temp_logits.mean(), multi_contour_eval=True)
        aff_mask = torch.zeros((mask.shape[0],mask.shape[1])).to(device)
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1,mask.shape[0] * mask.shape[1])
        trans_mat = attn_weight * aff_mask
        logits_refined_ccls = torch.matmul(trans_mat, logits_coarse[:,ccls:ccls+1])
        logits_refined[:, ccls] = logits_refined_ccls.squeeze()
    return logits_refined

def cwr(logits, logits_max, h, w, image, text_features):
    patch_size = 16
    input_size = 224
    stride = input_size // patch_size
    candidate_cls_list = []
    
    ma = logits.max()
    mi = logits.min()
    step = ma - mi
    thres_abs = 0.5
    thres = mi + thres_abs*step
        
    for tempid,tempv in enumerate(logits_max):
        if tempv > thres:
            candidate_cls_list.append(tempid)
    for ccls in candidate_cls_list:
        temp_logits = logits[:,ccls]
        temp_logits = temp_logits - temp_logits.min()
        temp_logits = temp_logits / temp_logits.max()
        mask = temp_logits > 0.5
        mask = mask.reshape(h // patch_size, w // patch_size)
        
        horizontal_indicies = np.where(np.any(mask.cpu().numpy(), axis=0))[0]
        vertical_indicies = np.where(np.any(mask.cpu().numpy(), axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        
        y1 = max(y1, 0)
        x1 = max(x1, 0)
        y2 = min(y2, mask.shape[-2] - 1)
        x2 = min(x2, mask.shape[-1] - 1)
        if x1 == x2 or y1 == y2:
            return logits_max
        
        mask = mask[y1:y2, x1:x2]
        mask = mask.float()
        mask = mask[None, None, :, :]
        mask = F.interpolate(mask, size=(stride, stride), mode="nearest")
        mask = mask.squeeze()
        mask = mask.reshape(-1).bool()
        
        image_cut = image[:, :, int(y1*patch_size):int(y2*patch_size), int(x1*patch_size):int(x2*patch_size)]
        image_cut = F.interpolate(image_cut, size=(input_size, input_size), mode="bilinear", align_corners=False)
        cls_attn = 1 - torch.ones((stride*stride+1, stride*stride+1))
        for j in range(1, cls_attn.shape[1]):
            if not mask[j - 1]:
                cls_attn[0, j] = -1000

        image_features = model.encode_image_tagclip(image_cut, input_size, input_size, attn_mask=cls_attn)[0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = model.logit_scale.exp()
        cur_logits = logit_scale * image_features @ text_features.t()
        cur_logits = cur_logits[:, 0, :]
        cur_logits = cur_logits.softmax(dim=-1).squeeze()
        cur_logits_norm = cur_logits[ccls]
        logits_max[ccls] = 0.5 * logits_max[ccls] + (1 - 0.5) * cur_logits_norm
            
    return logits_max


def classify():
    global img_root, image_list, all_label_list
    pred_label_id = []
    gt_one_hot = []

    with torch.no_grad():
        text_features = clip.encode_text_with_prompt_ensemble(model, class_names, device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for im_idx, im in enumerate(tqdm(image_list)):
        image_path = os.path.join(img_root, im)
        gt_one_hot.append(all_label_list[im_idx])
        
        pil_img = Image.open(image_path)
        array_img = np.array(pil_img)
        ori_height, ori_width = array_img.shape[:2]
        if len(array_img.shape) == 2:
            array_img = np.stack([array_img, array_img, array_img], axis=2)
            pil_img = Image.fromarray(np.uint8(array_img))
        
        patch_size = 16
        preprocess = _transform_resize(int(np.ceil(int(ori_height) / patch_size) * patch_size), int(np.ceil(int(ori_width) / patch_size) * patch_size))
        image = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract image features
            h, w = image.shape[-2], image.shape[-1]
            
            image_features, attn_weight_list = model.encode_image_tagclip(image, h, w, attn_mask=1)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            

            attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]
            
            attn_vote = torch.stack(attn_weight, dim=0).squeeze()
            
            thres0 = attn_vote.reshape(attn_vote.shape[0], -1)
            thres0 = torch.mean(thres0, dim=-1).reshape(attn_vote.shape[0], 1, 1)
            thres0 = thres0.repeat(1, attn_vote.shape[1], attn_vote.shape[2])
            
            attn_weight = torch.stack(attn_weight, dim=0)[8:-1]
            
            attn_cnt = attn_vote > thres0
            attn_cnt = attn_cnt.float()
            attn_cnt = torch.sum(attn_cnt, dim=0)
            attn_cnt = attn_cnt >= 4
            
            attn_weight = torch.mean(attn_weight, dim=0)[0]
            attn_weight = attn_weight * attn_cnt.float()
    
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()#torch.Size([1, 197, 81])
            logits = logits[:, 1:, :]
            logits = logits.softmax(dim=-1)
            logits_coarse = logits.squeeze()
            logits = torch.matmul(attn_weight, logits)
            logits = logits.squeeze()
            logits = mask_attn(logits_coarse, logits, h, w, attn_weight)

            logits_max = torch.max(logits, dim=0)[0]
            logits_max = logits_max[:NUM_CLASSES]
            logits_max = cwr(logits, logits_max, h, w, image, text_features)
            logits_max = logits_max.cpu()
        pred_label_id.append(logits_max)
        
    predictions = torch.stack(pred_label_id, dim=0)
    labels = torch.from_numpy(np.array(gt_one_hot))
    
    evaluate(predictions, labels, verbose=True)
    search_best_threshold(predictions, labels, verbose=True)

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='coco2014', choices=['voc2012', 'coco2014'])
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
    all_label_list = np.load(full_label_file)
    print("Dataset:", args.dataset)
    print("The number of classes in dataset:", NUM_CLASSES)
    print("The number of classes in vocabulary:", len(class_names))

    model, _ = clip.load("pretrained_models/ViT-B-16.pt", device=device)
    model.eval()

    classify()