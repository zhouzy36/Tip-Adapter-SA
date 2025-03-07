# coding=utf-8
import clip
import cv2
import numpy as np
import random
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional, List, Any, Union
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

from clip_text import class_names_voc, BACKGROUND_CATEGORY_VOC, class_names_coco, BACKGROUND_CATEGORY_COCO, class_names_LaSO
from dataset import NumpyDataset, TxtDataset, LaSOSplitDataset
from loss import IULoss, ANLoss, WANLoss, EMLoss


def scoremap2bbox(scoremap, threshold, multi_contour_eval: bool=False):
    """Copy from https://github.com/linyq2117/TagCLIP/blob/main/utils.py
    """
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


def compute_F1(predictions: Tensor, labels: Tensor, average: str="micro", threshold: float=0.5, use_relative: bool=False):
    """Compute F1 score, precision and recall metrics.
    Args:
        predictions (tensor): Classification logits with size [num_samples, num_classes].
        labels (tensor): Label vector {0, 1}^{num_classes} with size [num_samples, num_classes].
        average (str): The type of averaging performed on the data, see sklearn's doc for details (default: "micro").
        threshold (float): Threshold value. (default: 0.5).
        use_relative (bool): Use relative threshold if set (default: False).
    Returns:
        f1: Averaged F1 scores.
        precision: Averaged precision scores.
        recall: Averaged recall scores.
    """
    assert threshold >= 0 and threshold <= 1
    # binarize predictions
    if use_relative:
        ma = predictions.max(dim=1)[0]
        mi = predictions.min(dim=1)[0]
        step = ma - mi
        threshold = mi + threshold * step
        for i in range(predictions.shape[0]):
            predictions[i][predictions[i] <= threshold[i]] = 0
            predictions[i][predictions[i] > threshold[i]] = 1
    else:
        predictions[predictions > threshold] = 1
        predictions[predictions <= threshold] = 0
    # compute metrics using metric functions from sklearn package
    f1 = f1_score(labels, predictions, average=average)
    precision = precision_score(labels, predictions, average=average)
    recall = recall_score(labels, predictions, average=average)
    return f1, precision, recall


def evaluate(predictions: Tensor, labels: Tensor, threshold: float=0.5, verbose: bool=True):
    """Compute mAP, F1 scores, precision and recall.
    Args:
        predictions (tensor): classification logit with size [num_samples, num_classes],
        labels (tensor): label vector {0, 1}^{num_classes} with size [num_samples, num_classes].
        threshold (float): Threshold value. (default: 0.5)
        verbose (bool): verbose flag. (default: True)
    
    Returns:
        mAP (tensor): Mean average precision (mAP).
        F1 (tensor): F1 scores.
        P (tensor): Precision scores.
        R (tensor): Recall scores.
    """
    # compute mAP
    mAP = average_precision_score(labels, predictions, average="macro")
    # compute F1, P, R with specific relative threshold
    F1, P, R = compute_F1(predictions.clone(), labels.clone(), threshold=threshold, use_relative=True)
    # report
    if verbose:
        print("================================================")
        print(f"mAP: {mAP:.6f}")
        print(f"F1: {F1:.6f}, Precision: {P:.6f}, Recall: {R:.6f}")
        print("================================================")
    return mAP, F1, P, R


def topk_acc(output: Tensor, target: Tensor, k: int=1):
    """Compute batch mean top-k accuracy.
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


def patch_classify(image_features: Tensor, 
                   text_features: Tensor, 
                   logit_scale: Tensor=100., 
                   drop_first: bool=True, 
                   use_softmax: bool=True):
    """Perform patch classify (this function will normalize image features).
    Args:
        image_features (Tensor): CLIP image features with size. [N, L, D]
        text_features (Tensor): CLIP text features with size. [C, D]
        logit_scale (Tensor): CLIP logits scale.
        drop_first (bool): drop the first token if set (default: True).
        use_softmax (bool): use softmax to normalize logits (default: True).
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


def upsample_logits(logits: Tensor,
                    input_size: Optional[int],
                    output_size: Optional[int],
                    patch_size: int,
                    mode: str="bilinear"):
    """Upsample patch-level classification logits.
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


def post_process(model: clip.model.CLIP, x: Tensor, batch_first: bool=False, only_class: bool=True):
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


def setup_seed(seed: int=2024):
    """Set up random seed.
    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_names(dataset: str, include_background: bool=False):
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
    elif dataset == "LaSO":
        class_names = class_names_LaSO + BACKGROUND_CATEGORY_COCO if include_background else class_names_LaSO
        num_classes = len(class_names_LaSO)
    else:
        raise NotImplementedError
    return class_names, num_classes


def get_test_dataset(dataset: str, transform=None, one_hot_label: bool=True) -> Dataset:
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
    elif dataset == "LaSO":
        img_root = "datasets/coco2014"
        image_file = "imageset/LaSO/formatted_val_images.npy"
        full_label_file = "imageset/LaSO/formatted_val_labels.npy"
    else:
        raise NotImplementedError
    image_list = np.load(image_file)
    full_label_list = np.load(full_label_file)
    test_dataset = NumpyDataset(img_root, image_list, full_label_list, transform=transform, one_hot_label=one_hot_label)
    return test_dataset


def get_split_dataset(dataset: str, split_file: str, transform=None, one_hot_label: bool=True) -> Dataset:
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

    if dataset in ["voc2012", "coco2014"]:
        if dataset == "voc2012":
            img_root = "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
        else:
            img_root = "datasets/coco2014"

        file_list = tuple(open(split_file, "r"))
        file_list = [id_.rstrip().split(" ") for id_ in file_list]
        image_list = [x[0] for x in file_list]
        label_list = [x[1:] for x in file_list]

        dataset = TxtDataset(dataset, img_root, image_list, label_list, transform=transform, one_hot_label=one_hot_label)

    elif dataset == "LaSO":
        coco_root = "datasets/coco2014"
        dataset = LaSOSplitDataset(coco_root, split_file, transform=transform, one_hot_label=one_hot_label)

    else:
        raise NotImplementedError

    return dataset


def load_results(path: str) -> list:
    """Load results from specified path.
    Args:
        path (str): The csv file path to save the results.
    Returns:
        df (dict): Result data.
    """
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df.to_dict(orient='records')
        except Exception as e:
            print(f"Error loading data from {path}: {e}")
            return []
    else:
        print(f"File {path} does not exist. Return empty list.")
        return []


def write_results(data: list, path: str):
    """Write results to specified path.
    Args:
        data (dict): Result data.
        path (str): The csv file path to save the results.
    """
    root_path = os.path.dirname(path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    try:
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        print(f"Successfully saved data to {path}.")
    except Exception as e:
        print(f"Error saving data to {path}: {e}")


def append_results(data: Union[dict, list], path: str):
    """Append results to specified path.
    Args:
        data (dict): Result data.
        path (str): The csv file path to save the results.
    """
    result_data = load_results(path)
    result_data.append(data) if isinstance(data, dict) else result_data.extend(data)
    write_results(result_data, path)


def search_best_threshold(predictions: Tensor, labels: Tensor, step: int=20, verbose: bool=True):
    """Search best threshold that maximize F1 score.
    Args:
        predictions (tensor): The classification logits with size [num_samples, num_classes],
        labels (tensor): Label vector {0, 1}^{num_classes} with size [num_samples, num_classes].
        step (int): The number of search step (default: 20).
        verbose (bool): Verbose flag (default: True).
    Returns:
        best_threshold: The best threshold ever found.
        F1: The best F1 scores.
        P: Precision.
        R: Recall.
    """
    best_F1 = 0.
    best_threshold = 0.
    # search loop
    for threshold in np.linspace(0, 1, num=step+1)[1:-1]:
        F1, P, R = compute_F1(predictions.clone(), labels.clone(), threshold=threshold, use_relative=True)
        if F1 > best_F1:
            best_F1 = F1
            best_threshold = threshold
    # reproduce best F1
    F1, P, R = compute_F1(predictions.clone(), labels.clone(), threshold=best_threshold, use_relative=True)
    # report
    if verbose:
        print(f"Best threshold: {best_threshold:.2f}, F1: {F1:.6f}, Precision: {P:.6f}, Recall: {R:.6f}")
    return best_threshold, F1, P, R


def get_loss_fn(loss: str, dataset: str, **kwargs):
    """Get loss function according to the loss name.
    Args:
        loss (str): Loss function name.
        dataset (str): Dataset name.
        kwargs: others
    Returns:
        loss_fn (nn.Module): Loss function.
    """
    _, NUM_CLASSES = get_class_names(dataset)
    
    if loss == "CE":
        loss_fn = nn.CrossEntropyLoss()

    elif loss == "IU":
        loss_fn = IULoss()

    elif loss == "AN":
        assert "epsilon" not in kwargs
        loss_fn = ANLoss(**kwargs)

    elif loss == "AN-LS":
        if "epsilon" not in kwargs:
            kwargs["epsilon"] = 0.1
        loss_fn = ANLoss(**kwargs)

    elif loss == "WAN":
        if "gamma" not in kwargs:
            kwargs["gamma"] = 1 / (NUM_CLASSES - 1)
        loss_fn = WANLoss(**kwargs)

    elif loss == "EM":
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.2 if dataset == "voc2012" else 0.1
        loss_fn = EMLoss(**kwargs)

    else:
        raise NotImplementedError
    
    return loss_fn