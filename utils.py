# coding=utf-8
import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from typing import Optional, List
BICUBIC = InterpolationMode.BICUBIC
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

from dataset import NumpyDataset, TxtDataset
from clip_text import class_names_voc, BACKGROUND_CATEGORY_VOC, class_names_coco, BACKGROUND_CATEGORY_COCO


def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)


def compute_AP(predictions, labels):
    num_class = predictions.size(1)
    ap = torch.zeros(num_class).to(predictions.device)
    empty_class = 0
    for idx_cls in range(num_class):
        prediction = predictions[:, idx_cls]
        label = labels[:, idx_cls]
        #mask = label.abs() == 1
        if (label > 0).sum() == 0:
            empty_class += 1
            continue
        binary_label = torch.clamp(label, min=0, max=1)
        sorted_pred, sort_idx = prediction.sort(descending=True)
        sorted_label = binary_label[sort_idx]
        tmp = (sorted_label == 1).float()
        tp = tmp.cumsum(0)
        fp = (sorted_label != 1).float().cumsum(0)
        num_pos = binary_label.sum()
        rec = tp/num_pos
        prec = tp/(tp+fp)
        ap_cls = (tmp*prec).sum()/num_pos
        ap[idx_cls].copy_(ap_cls)
    return ap


def compute_F1(predictions, labels, mode_F1, k_val, use_relative=False):
    if k_val >= 1:
        idx = predictions.topk(dim=1, k=k_val)[1]
        predictions.fill_(0)
        predictions.scatter_(dim=1, index=idx, src=torch.ones(predictions.size(0), k_val, dtype=predictions.dtype).to(predictions.device))
    else:
        if use_relative:
            ma = predictions.max(dim=1)[0]
            mi = predictions.min(dim=1)[0]
            step = ma - mi
            thres = mi + k_val * step
        
            for i in range(predictions.shape[0]):
                predictions[i][predictions[i] <= thres[i]] = 0 # the order is very important!
                predictions[i][predictions[i] > thres[i]] = 1
        else:
            predictions[predictions > k_val] = 1
            predictions[predictions <= k_val] = 0
        
    if mode_F1 == 'overall':
        predictions = predictions.bool()
        labels = labels.bool()
        TPs = ( predictions &  labels).sum()
        FPs = ( predictions & ~labels).sum()
        FNs = (~predictions &  labels).sum()
        eps = 1.e-9
        Ps = TPs / (TPs + FPs + eps)
        Rs = TPs / (TPs + FNs + eps)
        p = Ps.mean()
        r = Rs.mean()
        f1 = 2*p*r/(p+r)
        
    elif mode_F1 == 'category':
        # calculate P and R
        predictions = predictions.bool()
        labels = labels.bool()
        TPs = ( predictions &  labels).sum(axis=0)
        FPs = ( predictions & ~labels).sum(axis=0)
        FNs = (~predictions &  labels).sum(axis=0)
        eps = 1.e-9
        Ps = TPs / (TPs + FPs + eps)
        Rs = TPs / (TPs + FNs + eps)
        p = Ps.mean()
        r = Rs.mean()
        f1 = 2*p*r/(p+r)
        
    elif mode_F1 == 'sample':
        # calculate P and R
        predictions = predictions.bool()
        labels = labels.bool()
        TPs = ( predictions &  labels).sum(axis=1)
        FPs = ( predictions & ~labels).sum(axis=1)
        FNs = (~predictions &  labels).sum(axis=1)
        eps = 1.e-9
        Ps = TPs / (TPs + FPs + eps)
        Rs = TPs / (TPs + FNs + eps)
        p = Ps.mean()
        r = Rs.mean()
        f1 = 2*p*r/(p+r)

    return f1, p, r

# new added

def evaluation(predictions, labels, thres_abs=0.5, verbose=True):
    """
    Args:
        predictions (tensor): classification logit with size [num_samples, num_classes],
        labels (tensor): label vector {0, 1}^{num_classes} with size [num_samples, num_classes].
        thres_abs (float): threshold. (default: 0.5)
        verbose (bool): verbose flag. (default: True)
    
    Returns:
        ap (tensor): average precision (AP) with shape [num_classes]
        F1 (tensor): F1 scores
        P (tensor): precision
        R (tensor): recall
    """
    # compute AP
    ap = compute_AP(predictions, labels)

    # compute F1, P, R with specific relative threshold
    F1, P, R = compute_F1(predictions.clone(), labels.clone(),  mode_F1='overall', k_val=thres_abs, use_relative=True)

    if verbose:
        print('================================================')
        print('mAP: %.6f' % torch.mean(ap))
        print('F1: %.6f, Precision: %.6f, Recall: %.6f' % (F1, P, R))
        print('================================================')

    return ap, F1, P, R


def topk_acc(output: torch.Tensor, target: torch.Tensor, k: int = 1):
    """Compute batch mean top-k accuracy
    Args:
        output (Tensor): model prediction with size [N, C]
        target (Tensor): ground truth with size [N]
        k (int): k (default: 1)
    Returns:
        acc (Tensor): top-k accuracy
    """
    pred = output.topk(k, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def patch_classify(image_features: torch.Tensor, 
                   text_features: torch.Tensor, 
                   logit_scale: torch.Tensor = 100., 
                   drop_first: bool = True, 
                   use_softmax: bool = True):
    """Perform patch classify (this function will normalize image features)
    Args:
        image_features (Tensor): CLIP image features with size. [N, L, D]
        text_features (Tensor): CLIP text features with size. [C, D]
        logit_scale (Tensor): CLIP logits scale. (default: 100.)
        drop_first (bool): drop the first token if set. (default: True)
        use_softmax (bool): use softmax to normalize logits. (default: True)
    Returns:
        logits (Tensor): classification logits with size [N, L, C].
    """
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t() # [bs, num_patches+1, num_classes]
    if drop_first:
        logits = logits[:, 1:, :]
    if use_softmax:
        logits = logits.softmax(dim=-1)
    return logits


def upsample_logits(logits:torch.Tensor,
                    input_size: Optional[int],
                    output_size: Optional[int],
                    patch_size: int,
                    mode: str = "bilinear"):
    """Upsample patch-level classification logits
    Args:
        logits (Tensor): logits to sampled with size [N, L, C] or [L, C].
        input_size: input spatial size.
        output_size: output spatial size.
        patch_size (int): ViT input patch size.
        mode (str): algorithm used for upsampling. (default: bilinear)
    Returns:
        upsampled_logits (Tensor): upsampled logits with size [N, L', C] where L'=(h*w)//(patch_size**2).
    """
    assert logits.dim() == 2 or logits.dim() == 3
    assert len(input_size) == 2 and len(output_size) == 2
    logits = logits.clone()
    batch_first = True
    if logits.dim() == 2:
        batch_first = False
        logits = logits.unsqueeze(0)
    N, L, C = logits.shape
    # upsample
    h1, w1 = input_size
    h2, w2 = output_size
    logits = logits.reshape([N, h1 // patch_size, w1 // patch_size, C]).permute([0, 3, 1, 2])
    logits = F.interpolate(logits, size=(h2 // patch_size, w2 // patch_size), mode=mode)
    logits = logits.reshape([N, C, -1]).permute([0, 2, 1])
    if not batch_first:
        logits = logits.squeeze(0)
    return logits


def post_process(model: clip.model.CLIP, x: torch.Tensor, batch_first: bool = False, only_class: bool = True):
    """Project intermediate output to joint text-image latent space
    Args:
        model (CLIP): CLIP model using ViT as image encoder
        x (Tensor): output of module with size [L, N, D] or [N, L, D]
        batch_first (bool): set true if the size of x is [N, L, D] (default: False)
        only_class (bool): return the output corresponding to class token if true (default: True)
    Returns:
        out (Tensor): return image-text aligned features with size [N, L, D]
    """
    assert isinstance(model.visual, clip.model.VisionTransformer)
    x = x.clone()
    if not batch_first:
        x = x.permute(1, 0, 2) # LND -> NLD
    # post layer norm
    if only_class:
        out = model.visual.ln_post(x[:, 0, :])
    else:  
        out = model.visual.ln_post(x)
    # project
    out = out @ model.visual.proj
    return out


def setup_seed(seed: int):
    """Set up random seed
    Args:
        seed (int): Random seed value.
    """
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def get_class_names(dataset: str, include_background: bool = False):
    """Get class names according to the dataset name.
    Args:
        dataset (str): Dataset name.
        include_background (bool): Include background classes if set True (default: False).
    Returns:
        class_names (list): The list of class names.
        num_classes (int): The number of classes in the dataset, excluding background classes.
    """
    if dataset == "voc2012":
        class_names = class_names_voc + BACKGROUND_CATEGORY_VOC if include_background else class_names_voc
        num_classes = len(class_names_voc)
    elif dataset == "coco2014":
        class_names = class_names_coco + BACKGROUND_CATEGORY_COCO if include_background else class_names_coco
        num_classes = len(class_names_coco)
    else:
        raise NotImplementedError
    return class_names, num_classes


def get_test_dataset(dataset: str, transform=None, one_hot_label: bool = True):
    """Get pytorch-style test dataset according to the dataset name.
    Args:
        dataset (str): Dataset name.
        transform: Data transformation (default: None).
        one_hot_label (bool): Use one-hot label if set (default: True).
    Returns:
        test_dataset (Dataset): Pytorch-style dataset.
    """
    if dataset == "voc2012":
        img_root = "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
        image_file = "imageset/voc2012/formatted_val_images.npy"
        full_label_file = "imageset/voc2012/formatted_val_labels.npy"
    elif dataset == "coco2014":
        img_root = "datasets/coco2014"
        image_file = "imageset/coco2014/formatted_val_images.npy"
        full_label_file = "imageset/coco2014/formatted_val_labels.npy"
    else:
        raise NotImplementedError
    image_list = np.load(image_file)
    full_label_list = np.load(full_label_file)
    test_dataset = NumpyDataset(img_root, image_list, full_label_list, transform=transform, one_hot_label=one_hot_label)
    return test_dataset


def get_split_dataset(dataset: str, split_file: str, transform=None, one_hot_label: bool = True):
    """Get pytorch-style dataset according to the txt split file.
    Args:
        dataset (str): Dataset name.
        split_file (str): The path to txt split file
        transform: Data transformation (default: None).
        one_hot_label (bool): Use one-hot label if set (default: True).
    Returns:
        dataset (Dataset): Pytorch-style dataset.
    """
    assert os.path.exists(split_file)
    if dataset == "voc2012":
        img_root = "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
    elif dataset == "coco2014":
        img_root = "datasets/coco2014"
    else:
        raise NotImplementedError
    file_list = tuple(open(split_file, "r"))
    file_list = [id_.rstrip().split(" ") for id_ in file_list]
    image_list = [x[0] for x in file_list]
    label_list = [x[1:] for x in file_list]
    dataset = TxtDataset(dataset, img_root, image_list, label_list, transform=transform, one_hot_label=one_hot_label)
    return dataset