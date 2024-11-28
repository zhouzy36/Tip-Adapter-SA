# coding=utf-8
import clip
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from utils import scoremap2bbox


@torch.no_grad()
def key_smoothing(logits: torch.Tensor, model: clip.model.CLIP, penultimate_features: torch.Tensor):
    """Refine logits using key similarities of the last ViT layer proposed in MaskCLIP.
    Args:
        logits (Tensor): The patch-level classification logits with size [N, L, C].
        model (CLIP): CLIP model.
        penultimate_features (Tensor): The output of penultimate ViT layer with size [L+1, N, C].
    Returns:
        refined_logits: The refined logits with the same size as input logits.
    """
    # check batch size and patch length
    assert logits.shape[0] == penultimate_features.shape[1]
    assert (logits.shape[1]+1) == penultimate_features.shape[0]

    # get weights of q, k, v projector in last vit layer
    last_vit_layer = model.visual.transformer.resblocks[11]
    last_vit_ln1 = last_vit_layer.ln_1
    last_vit_msa = last_vit_layer.attn
    qkv_weight = last_vit_msa.in_proj_weight # [3*D', D']
    qkv_bias = last_vit_msa.in_proj_bias # [D']

    # compute q, k, v manuallyf
    qkv = F.linear(last_vit_ln1(penultimate_features), qkv_weight, qkv_bias)
    q, k, v = torch.split(qkv, penultimate_features.shape[-1], dim=-1) # [L+1, 1, D']

    # compute key similarity
    k = k.permute(1, 0, 2) # [L+1, N, D'] -> [N, L+1, D']
    k = k / k.norm(dim=-1, keepdim=True) # normalized
    key_sim = torch.bmm(k, k.transpose(1, 2)) # [N, L+1, D'] @ [N, D', L+1] -> [N, L+1, L+1]
    patch_key_sim = key_sim[:, 1:, 1:]
    # # visualize 
    # fig, ax = plt.subplots()
    # ax.imshow(patch_key_sim.squeeze().detach().cpu().numpy())
    # plt.show()

    # refine
    refined_logits = torch.bmm(patch_key_sim, logits)
    # refined_logits = F.softmax(refined_logits, dim=-1)
    return refined_logits


@torch.no_grad()
def raw_attention_refine(logits: torch.Tensor,
                     attn_weight_list: List[torch.Tensor],
                     last_k: int = 3,
                     mask: bool = True,
                     vote_k: int = 4):
    """Refine logits using attention weights.
    Args:
        logits (Tensor): The patch-level classification logits with size [N, L, C].
        attn_weight_list (List): The list of attention weight with size [N, L+1, L+1].
        last_k (int): Use attention weights of the last k layers except the last (default: 3).
        mask (bool): Generate attention mask according to eq(12) of TagCLIP paper if set (default: True).
        vote_k (int): The K in eq(12) of TagCLIP paper (default: 4).
    Returns:
        refined_logits (Tensor): The refined logits with the same size as input logits.
    """
    attn_weights = torch.stack(attn_weight_list[:-1], dim=1) # discard the last attention weights
    attn_weights = attn_weights[..., 1:, 1:] # discard class token related attention weights
    refine_weights = torch.mean(attn_weights[-last_k:], dim=1)

    if mask:
        # compute attention mean of each layer (layer-wise mean)
        attn_vote = attn_weights.clone()
        N, K, L = attn_vote.shape[:3]
        threshold = attn_vote.view(N, K, -1).mean(dim=-1)
        threshold = threshold.view(N, K, 1, 1).repeat(1, 1, L, L)
        # generate mask
        attn_mask = attn_vote > threshold
        attn_mask = torch.sum(attn_mask.float(), dim=1)
        attn_mask = attn_mask >= vote_k
        # compute masked attention weights as refine weights
        refine_weights = refine_weights * attn_mask.to(refine_weights.dtype)
        
    refined_logits = torch.bmm(refine_weights, logits)
    return refined_logits


@torch.no_grad()
def class_attention_refine(logits: torch.Tensor,
                           h: int,
                           w: int,
                           patch_size: int,
                           attn_weight_list: List[torch.Tensor],
                           last_k: int = 3):
    """Refine logtis using masked attention weights where masks are generated according to logits.
    Args:
        logits (Tensor): The patch-level classification logits with size [L, C].
        h (int): Image height.
        w (int): Image width.
        patch_size (int): ViT input patch size.
        attn_weights (Tensor): The list of attention weight with size [N, L+1, L+1].
        last_k (int): Use attention weights of the last k layers except the last (default: 3).
    Returns:
        refined_logits (Tensor): The refined logits with the same size as input logits.
    """
    refined_logits = logits.clone()
    attn_weights = torch.stack(attn_weight_list[:-1], dim=1) # discard the last attention weights
    attn_weights = attn_weights[..., 1:, 1:] # discard class token related attention weights
    refine_weights = torch.mean(attn_weights[-last_k:], dim=1).squeeze(0)
    
    for cls_idx in range(logits.shape[-1]):
        # get logits map
        cls_logits = logits[..., cls_idx] # [L]
        logits_map = cls_logits.clone()
        logits_map = (logits_map - logits_map.min()) / (logits_map.max() - logits_map.min()) # min-max normalize
        logits_map = logits_map.reshape(h // patch_size, w // patch_size)
        # generate mask according to bbx (performance bottleneck)
        box, cnt = scoremap2bbox(logits_map.detach().cpu().numpy(), 
                                 threshold=logits_map.mean(), 
                                 multi_contour_eval=True)
        cls_mask = torch.zeros_like(logits_map).to(refine_weights.device)
        for i in range(cnt):
            x0, y0, x1, y1 = box[i]
            cls_mask[y0:y1, x0:x1] = 1.
        # refine
        cls_mask = cls_mask.reshape([1, -1])
        cls_refine_weights = refine_weights * cls_mask.to(refine_weights.dtype) # [L, L]
        refined_logits[..., cls_idx] = cls_refine_weights @ refined_logits[..., cls_idx] # [L, L] @ [L]
    return refined_logits


@torch.no_grad()
def double_mask_attention_refine(logits: torch.Tensor,
                                 h: int,
                                 w: int,
                                 patch_size: int,
                                 attn_weight_list: List[torch.Tensor],
                                 last_k: int = 3,
                                 vote_k: int = 4
                                 ):
    """Modified DMAR module proposed in TagCLIP: generate class-wise mask based on coarse logits for all classes.
    Args:
        logits (Tensor): The patch-level classification logits with size [L, C].
        h (int): Image height.
        w (int): Image width.
        patch_size (int): ViT input patch size.
        attn_weight_list (List): The list of attention weight with size [1, L+1, L+1].
        last_k (int): Use attention weights of the last k layers except the last (default: 3).
        vote_k (int): The K in eq(12) of TagCLIP paper (default: 4).
    Returns:
        refined_logits (Tensor): The refined logits with the same size as input logits.
    """
    assert logits.dim() == 2, "The expected input size of logits is [L, C]"

    coarse_logits = logits.clone()
    attn_weights = torch.stack(attn_weight_list[:-1], dim=1).squeeze(0) # [11, L+1, L+1]
    attn_weights = attn_weights[:, 1:, 1:] # discard class token related attention weights
    refine_weights = torch.mean(attn_weights[-last_k:], dim=0) # [L, L]

    # compute attention mean of each layer (layer-wise mean)
    attn_vote = attn_weights.clone() # [11, L, L]
    K, L = attn_vote.shape[:2]
    threshold = attn_vote.view(K, -1).mean(dim=-1)
    threshold = threshold.view(K, 1, 1).repeat(1, L, L) # [11, L, L]

    # generate attention mask by voting-style approach
    attn_mask = attn_vote > threshold
    attn_mask = torch.sum(attn_mask.float(), dim=0) # [L, L]
    attn_mask = attn_mask >= vote_k

    refine_weights = refine_weights * attn_mask.to(refine_weights.dtype)
    attn_refined_logits = refine_weights @ coarse_logits
    refined_logits = attn_refined_logits.clone()

    for cls_idx in range(logits.shape[-1]):
        cls_logits = coarse_logits[:, cls_idx]
        logits_map = cls_logits.clone()
        logits_map = (logits_map - logits_map.min()) / (logits_map.max() - logits_map.min()) # min-max normalize
        logits_map = logits_map.reshape(h // patch_size, w // patch_size)
        # generate mask according to bbx (performance bottleneck)
        box, cnt = scoremap2bbox(logits_map.detach().cpu().numpy(), 
                                    threshold=logits_map.mean(), 
                                    multi_contour_eval=True)
        cls_mask = torch.zeros_like(logits_map).to(refine_weights.device)
        for i in range(cnt):
            x0, y0, x1, y1 = box[i]
            cls_mask[y0:y1, x0:x1] = 1.
            
        # refine
        cls_mask = cls_mask.reshape([1, -1])
        cls_refine_weights = refine_weights * cls_mask.to(refine_weights.dtype) # [L, L]
        refined_logits[:, cls_idx] = cls_refine_weights @ coarse_logits[:, cls_idx]

    return refined_logits

@torch.no_grad()
def my_double_mask_attention_refine(logits: torch.Tensor,
                                 h: int,
                                 w: int,
                                 patch_size: int,
                                 attn_weight_list: List[torch.Tensor],
                                 last_k: int = 3,
                                 vote_k: int = 4
                                 ):
    """Modifed DMAR module proposed in TagCLIP: do OR operation between two masks rather than multiplication.
    Args:
        logits (Tensor): The patch-level classification logits with size [L, C].
        h (int): Image height.
        w (int): Image width.
        patch_size (int): ViT input patch size.
        attn_weight_list (List): The list of attention weight with size [1, L+1, L+1].
        last_k (int): Use attention weights of the last k layers except the last (default: 3).
        vote_k (int): The K in eq(12) of TagCLIP paper (default: 4).
    Returns:
        refined_logits (Tensor): The refined logits with the same size as input logits.
    """
    refined_logits = logits.clone()
    attn_weights = torch.stack(attn_weight_list[:-1], dim=1).squeeze(0) # [11, L+1, L+1]
    attn_weights = attn_weights[:, 1:, 1:] # discard class token related attention weights
    refine_weights = torch.mean(attn_weights[-last_k:], dim=0) # [L, L]

    # compute attention mean of each layer (layer-wise mean)
    attn_vote = attn_weights.clone() # [11, L, L]
    K, L = attn_vote.shape[:2]
    threshold = attn_vote.view(K, -1).mean(dim=-1)
    threshold = threshold.view(K, 1, 1).repeat(1, L, L) # [11, L, L]

    # generate attention mask by voting-style approach
    attn_mask = attn_vote > threshold
    attn_mask = torch.sum(attn_mask.float(), dim=0) # [L, L]
    attn_mask = attn_mask >= vote_k

    for cls_idx in range(logits.shape[-1]):
        # get logits map
        cls_logits = logits[:, cls_idx] # [L] DMAR*: generate logits map according to coarse logits
        logits_map = cls_logits.clone()
        logits_map = (logits_map - logits_map.min()) / (logits_map.max() - logits_map.min()) # min-max normalize
        logits_map = logits_map.reshape(h // patch_size, w // patch_size)
        # generate class-wise mask according to bbx (performance bottleneck)
        box, cnt = scoremap2bbox(logits_map.detach().cpu().numpy(), 
                                    threshold=logits_map.mean(), 
                                    multi_contour_eval=True)
        cls_mask = torch.zeros_like(logits_map).to(refine_weights.device)
        for i in range(cnt):
            x0, y0, x1, y1 = box[i]
            cls_mask[y0:y1, x0:x1] = 1.
            
        # refine
        cls_mask = cls_mask.reshape([1, -1])
        mask = attn_mask | cls_mask.to(attn_mask.dtype)
        cls_refine_weights = refine_weights * mask.to(refine_weights.dtype) # [L, L]
        refined_logits[:, cls_idx] = cls_refine_weights @ logits[:, cls_idx]

    return refined_logits


def extract_class_specific_features(patch_feats: torch.Tensor, 
                                    logits: torch.Tensor, 
                                    label: torch.Tensor, 
                                    one_hot_label: bool = True):
    """Extract class specific features by averaging class specific patch features.
    Args:
        patch_feats (Tensor): The patch features with size [L, D].
        logits (Tensor): The refined patch classification logtis with size [L, C].
        labels (Tensor): One hot label vector with size [1, num_classes] or label tensor with size [1, num_labels].
        one_hot_label (bool): Indicate that the label is a one-hot label (default: True).
    Retuens:
        class_specific_features (Tensor): Features with size [num_labels, D].
    """
    assert patch_feats.dim() == 2 and logits.dim() == 2
    assert patch_feats.shape[0] == logits.shape[0]

    if one_hot_label:
        label = torch.nonzero(label, as_tuple=True)[1].unsqueeze(0)

    class_specific_features = torch.empty([label.shape[-1], patch_feats.shape[-1]]).to(patch_feats.device)

    for i, ccls in enumerate(label.flatten().tolist()):
        temp_logits = logits[:,ccls]
        temp_logits = temp_logits - temp_logits.min()
        temp_logits = temp_logits / temp_logits.max()
        idx = torch.where(temp_logits > 0.5) # idx = torch.where(temp_logits > 0.5)[0]
        class_specific_features[i] = torch.mean(patch_feats[idx], dim=0, keepdim=True)
    
    return class_specific_features