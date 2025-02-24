# coding=utf-8
import argparse
import clip
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FeatDataset
from utils import setup_seed, get_class_names, evaluate, append_results

"""
Example:
python Tip-Adapter-SA.py --test-data-path features/${dataset}/MyCLIP/val_all.pt --dataset ${dataset} --cache-path caches/${dataset}/MyCLIP/exp${split}_${k}shots_filtered.pt --k-shot ${k} --normalize ${normalize} --search-hp --save-results
"""

def normalize(logits: torch.Tensor, method: str="softmax"):
    """Normalize classification logits.
    Args:
        logits (Tensor): Classification logits with size [num_samples, num_classes].
        method (str): Normalization method (default: "gaussian")
    Returns:
        normalized_logits (Tensor): Normalized logits.
    """
    if method == "softmax":
        normalized_logits = F.softmax(logits, dim=-1)
    elif method == "min-max":
        logits_min = logits.min(dim=-1, keepdim=True).values
        logits_max = logits.max(dim=-1, keepdim=True).values
        normalized_logits = (logits - logits_min) / (logits_max - logits_min)
    elif method == "gaussian":
        logits_std = torch.std(logits, dim=-1, keepdim=True)
        logits_mean = torch.mean(logits, dim=-1, keepdim=True)
        normalized_logits = (logits - logits_mean) / logits_std
    else:
        raise NotImplementedError
    return normalized_logits


class TipAdatperSA(nn.Module):
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, k:int, beta: float):
        super(TipAdatperSA, self).__init__()
        self.keys = nn.Parameter(keys)
        self.register_buffer("values", values)
        self.k = k
        self.beta = beta

    def forward(self, x):
        assert x.dim() == 3
        N, C, D = x.shape
        assert self.k*C == self.keys.shape[0]
        x = x.repeat(1, 1, self.k).reshape([-1, D]) # [N, C, D] -> [N*C*k, D]
        affinity = torch.sum(self.keys.repeat(N, 1)*x, dim=-1).reshape([N, -1]) # [N, C*k]
        affinity[torch.isnan(affinity)] = 0.
        out = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.values
        return out, affinity
    
    def set_beta(self, beta: float):
        self.beta = beta


def parse_args():
    # define parser
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--test-data-path", type=str, required=True, help="The path to test features.")
    parser.add_argument("--dataset", type=str, default="coco2014", choices=["coco2014", "voc2012", "LaSO"])
    parser.add_argument("--cache-path", required=True, type=str, metavar="PATH", help="cache path")
    parser.add_argument("--k-shot", required=True, type=int, help="Shot number.")
    
    # logits fusion setting
    parser.add_argument("--normalize", type=str, default="softmax", choices=["softmax", "min-max", "gaussian"], help="Normalize method (default: softmax).")
        
    # save results
    parser.add_argument("--save-results", action="store_true", help="Save experiment results if set.")
    parser.add_argument("--result-root", type=str, default="results", help="The root path to save results (default: results).")
    
    # hyper parameter search
    parser.add_argument("--search-hp", action='store_true', help="Search hyper-parameters if set")
    parser.add_argument("--init-alpha", type=float, default=0.5, help="Residual Ratio alpha. Larger alpha denotes using more knowledge from the few-shot training set and less otherwise (default: 0.5).")
    parser.add_argument("--init-beta", type=float, default=1.0, help="Sharpness Ratio beta. When beta is large, only the nearby training samples to the test image have large influences to its class prediction and vice versa.(default: 1.0).")
    parser.add_argument("--search-scale", nargs="+", type=int, default=[1, 10], help="Search scale. (default: [1, 10])")
    parser.add_argument("--search-step", nargs="+", type=int, default=[10, 20], help="Search step. (default: [10, 20])")
    parser.add_argument("--search-metric", type=str, default="mAP", choices=["mAP", "F1", "precision", "recall"], help="Search metric. (default: mAP")

    # parse args
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    # set up random seed
    setup_seed()
    
    # parse arguments
    args = parse_args()
    
    # initialize device
    device = torch.device("cuda:0")

    # initialize method name and split name
    method_name = "Tip-Adapter-SA"
    method_name += f"-{args.cache_path.split('/')[2]}"
    split_name = os.path.basename(args.cache_path).split(".")[0]

    # load CLIP model
    model_path = "pretrained_models/ViT-B-16.pt"
    patch_size = 16
    model, _ = clip.load(model_path, device)
    model.eval()
    logit_scale = model.logit_scale.exp().detach()

    # load class name
    class_names, NUM_CLASSES = get_class_names(args.dataset)
    print("The number of classes in dataset:", NUM_CLASSES)
    print("The number of classes in vocabulary:", len(class_names))

    # get text classifier weights
    with torch.no_grad():
        text_features = clip.encode_text_with_prompt_ensemble(model, class_names, device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # [num_classes, d]
    print(text_features.shape, text_features.dtype)

    # test loader
    test_dataset = FeatDataset(args.test_data_path)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=128, 
                                 shuffle=False)
    
    # load cache model
    assert os.path.exists(args.cache_path)
    cache = torch.load(args.cache_path, map_location=torch.device("cpu"))
    cache_keys = cache["keys"] # [NK, D]
    cache_values = cache["values"] # [NK, N]
    print(f"Load cache from file {args.cache_path}")

    # define adapter
    adapter = TipAdatperSA(cache_keys, cache_values, k=args.k_shot, beta=args.init_beta)
    adapter = adapter.to(device)

    # prepare for hyper-parameter search
    all_affinity = []
    all_zeroshot_logits = []

    # test
    adapter.eval()
    pred_logits = []
    label_vectors = []
    with torch.no_grad():
        for feats, labels, zeroshot_logits in tqdm(test_dataloader):
            # move data to device
            feats = feats.to(device)
            zeroshot_logits = zeroshot_logits.to(device)
            # forward
            feats = feats / feats.norm(dim=-1, keepdim=True)
            cache_logits, affinity = adapter(feats)
            all_affinity.append(affinity.cpu())
            # normalize logits
            zeroshot_logits = normalize(zeroshot_logits, method=args.normalize)
            cache_logits = normalize(cache_logits, method=args.normalize)
            # fuse logits
            final_logits = zeroshot_logits * (1 - args.init_alpha) + cache_logits * args.init_alpha
            pred_logits.append(final_logits.cpu())
            label_vectors.append(labels)
    all_zeroshot_logits = test_dataset.get_all_logits()
    all_affinity = torch.cat(all_affinity, dim=0)
    pred_logits = torch.cat(pred_logits, dim=0)
    label_vectors = torch.cat(label_vectors, dim=0)

    # evaluate
    mAP, F1, P, R = evaluate(pred_logits, label_vectors, verbose=True)

    if args.search_hp:
        # move data to device
        all_zeroshot_logits = all_zeroshot_logits.to(device)
        all_affinity = all_affinity.to(device)
        cache_values = cache_values.to(device)
        # search range
        search_scale, search_step = args.search_scale, args.search_step
        alpha_list = torch.linspace(0, search_scale[0], (search_step[0]+1))[1:].tolist()
        beta_list = torch.linspace(0, search_scale[1], (search_step[1]+1))[1:].tolist()
        print(f"Search scale: {search_scale}\nSearch step: {search_step}.")
        # search metric
        all_metrics = {"mAP": mAP, "F1": F1, "precision": P, "recall": R}
        metric = args.search_metric
        best_metric = all_metrics[metric]
        best_alpha, best_beta = args.init_alpha, args.init_beta

        print("Start searching hyperparameters.")
        start_time = time.time()
        for alpha in tqdm(alpha_list):
            for beta in beta_list:
                all_cache_logits = ((-1) * (beta - beta * all_affinity)).exp() @ cache_values
                all_cache_logits = normalize(all_cache_logits, method=args.normalize)
                # fuse logits
                all_final_logits = (1 - alpha) * all_zeroshot_logits + alpha * all_cache_logits # [num_samples, num_classes]
                # evaluate
                mAP, F1, P, R = evaluate(all_final_logits.cpu(), label_vectors, verbose=False)
                all_metrics = {"mAP": mAP, "F1": F1, "precision": P, "recall": R}
                # update the best metric
                if all_metrics[metric] > best_metric:
                    best_metric = all_metrics[metric]
                    best_alpha = alpha
                    best_beta = beta
        end_time = time.time()
        print(f"After searching, the best {metric}: {best_metric:.6f}, cost: {(end_time-start_time):.1f}s")
        # reproduce the best metric
        all_cache_logits = ((-1) * (best_beta - best_beta * all_affinity)).exp() @ cache_values # [num_samples, num_classes]
        all_cache_logits = normalize(all_cache_logits, method=args.normalize)
        all_final_logits = (1 - best_alpha) * all_zeroshot_logits + best_alpha * all_cache_logits # [num_samples, num_classes]
        mAP, F1, P, R = evaluate(all_final_logits.cpu(), label_vectors, verbose=False)
        print(f"The best setting: alpha: {best_alpha:.2f}, beta: {best_beta:.2f}")
        print(f"mAP: {mAP:.6f}, F1: {F1:.6f}, Precision: {P:.6f}, Recall: {R:.6f}")
        # set the alpha and adapter's beta to the best value ever found.
        adapter.set_beta(best_beta)
        args.init_alpha = best_alpha

    # save results
    if args.save_results:
        result_data = {"normalize": args.normalize,
                       "mAP": mAP.item(),
                       "F1": F1.item()}
        if args.search_hp:
            result_data["best_alpha"] = best_alpha
            result_data["best_beta"] = best_beta
        # write results
        result_path = os.path.join(args.result_root, 
                                   args.dataset, 
                                   method_name, 
                                   f"{split_name}.csv")
        append_results(result_data, result_path)