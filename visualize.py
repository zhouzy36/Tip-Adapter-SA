# coding=utf-8
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from typing import List

def show_image(image: torch.Tensor, labels: torch.Tensor, class_names: List, is_label_vector=True):
    """visualize image tensor and label
    Args:
        image (Tensor): image tensor with shape [channel, height, width].
        labels (Tensor): label vector with shape [NUM_CLASSES] or [1, NUM_CLASSES] label tensor with shape [NUM_LABELS] or [1, NUM_LABELS].
        class_names (List): The list of class names.
    Returns:
        None
    """
    assert len(image.shape) == 3 or (len(image.shape) == 4 and image.shape[0] == 1)
    assert len(labels.shape) == 1 or (len(labels.shape) == 2 and labels.shape[0] == 1)
    fig, ax = plt.subplots()
    tmp = torchvision.utils.make_grid(image, normalize=True, scale_each=True)
    ax.axis("off")
    ax.imshow(transforms.functional.to_pil_image(tmp))
    plt.show()
    if is_label_vector:
        print(f"Ground Truth: {", ".join([class_names[int(i)] for i in torch.where(labels.squeeze() == 1)[0].tolist()])}")
    else:
        print(f"Ground Truth: {", ".join([class_names[int(i)] for i in labels.flatten().tolist()])}")

    # return fig
    

def show_local_classification(logits, h, w, patch_size, scale_threshold=False, threshold=0.5):
    """ Visualize patch-level classification results
    Args:
        logits (Tensor): classification logits with size [L, C]
        h (int): image height
        w (int): image width
        patch_size (int): patch size
        scale_threshold (bool): set True to scale threshold according the minimum and maximum of logits. (default: False)
        threshold (float): threshold value. (default: 0.5)
    Returns:
        None
    """
    assert len(logits.shape) == 2, f"The expected input size of logits is [L, C], but get logits with size {logits.shape}"
    
    global class_names
    
    if scale_threshold:
        ma = logits.max() # global maximum of logits
        mi = logits.min() # global minimum of logits
        step = ma - mi
        threshold = mi + threshold * step

    values, idx = torch.max(logits, dim=1)
    pred = torch.where(values > threshold, idx, -1) # [L]; -1 means others
    num_prediction_all = pred.unique().numel()
    num_prediction_dataset = num_prediction_all - 1

    # build map between prediction and pixel value
    id2pixel = {cls_id: i for i, cls_id in enumerate(pred.unique().tolist())}
    pixel2id = {v: k for k, v in id2pixel.items()}

    # create mask
    mask = torch.tensor([id2pixel.get(v.item()) for v in pred.flatten()]).reshape(h // patch_size, w // patch_size)
    
    # create colormap
    if num_prediction_dataset <= 10:
        colors = plt.get_cmap('tab10').colors
    elif num_prediction_dataset > 10 and num_prediction_dataset <= 20:
        colors = plt.get_cmap('tab20').colors
    elif num_prediction_dataset > 20 and num_prediction_dataset <= 40:
        colors = plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors
    elif num_prediction_dataset > 40 and num_prediction_dataset <= 60:
        colors = plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').color
    else:
        print("Too much prediction!")
        return
    color_list = [(0., 0., 0.)] + [colors[i] for i in range(num_prediction_dataset)]
    colormap = ListedColormap(color_list)

    # visualize
    fig, ax = plt.subplots()
    mask_img = ax.imshow(mask.cpu().numpy(), cmap=colormap)
    ax.axis("off")
    cbar = fig.colorbar(mask_img, fraction=0.05)

    # set ticks
    interval = 1 / num_prediction_all
    cbar.set_ticks([(interval / 2 + i * interval)*num_prediction_dataset for i in range(num_prediction_all)])    
    cbar.set_ticklabels([class_names[pixel2id[i]] if pixel2id[i] != -1 else "background" for i in range(num_prediction_all)])

    plt.show()
    print(f"The number of predicted classes: {num_prediction_dataset}")
    print("Prediction:", [class_names[int(i)] for i in pred.unique().tolist() if i >= 0])

    # return fig