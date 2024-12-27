# coding=utf-8
import argparse
import clip
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import FeatDataset
from utils import setup_seed, get_class_names, evaluate, append_results

"""
Example:
# Tip-Adapter
python Tip-Adapter.py --test-data-path features/${dataset}/CLIP/val_all.pt --dataset ${dataset} --cache-path caches/${dataset}/CLIP/exp{split}_{k}shots_filtered.pt --search-hp --tensorboard --save-results

# Tip-Adapter-F
python Tip-Adapter.py --test-data-path features/${dataset}/CLIP/val_all.pt --dataset ${dataset} --cache-path caches/${dataset}/CLIP/exp{split}_{k}shots_filtered.pt --search-hp --train --train-data-path features/${dataset}/CLIP/exp{split}_{k}shots_filtered.pt --tensorboard --save-results
"""

class TipAdapter(nn.Module):
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, beta: float):
        super(TipAdapter, self).__init__()
        self.fc = nn.Linear(keys.shape[1], values.shape[0], bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(keys)
        self.register_buffer("values", values)
        self.beta = beta

    def forward(self, x):
        out = self.fc(x)
        out = ((-1) * (self.beta - self.beta * out)).exp() @ self.values
        return out

    def set_beta(self, beta: float):
        self.beta = beta


def train_loop(dataloader, adapter, loss_fn, optimizer):
    global logit_scale, text_features
    adapter.train()
    num_batches = len(dataloader)
    train_loss = 0.

    for X, y, _ in dataloader:
        # move data to device
        X = X.to(device)
        y = y.to(device)

        # compute prediction and loss
        X = X / X.norm(dim=-1, keepdim=True)
        cache_logits = adapter(X)
        clip_logits = logit_scale * X @ text_features.t()
        tip_logits = clip_logits + cache_logits * args.init_alpha
        loss = loss_fn(tip_logits, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # record loss
        train_loss += loss.item()
        
    train_loss /= num_batches

    return train_loss


def test_loop(dataloader, adapter, loss_fn):
    global logit_scale, text_features
    adapter.eval()
    num_batches = len(dataloader)
    test_loss = 0.
    pred_logits = []
    label_vectors = []
    with torch.no_grad():
        for X, y, _ in dataloader:
            # move data to device
            X = X.to(device)
            y = y.to(device)

            # compute prediction and loss
            X = X / X.norm(dim=-1, keepdim=True)
            cache_logits = adapter(X)
            clip_logits = logit_scale * X @ text_features.t()
            tip_logits = clip_logits + cache_logits * args.init_alpha
            loss = loss_fn(tip_logits, y)
            
            # record loss and prediction results
            test_loss += loss.item()
            pred_logits.append(tip_logits.softmax(dim=-1).cpu())
            label_vectors.append(y.cpu())
    test_loss /= num_batches
    # evaluate
    pred_logits = torch.cat(pred_logits, dim=0)
    label_vectors = torch.cat(label_vectors, dim=0)
    mAP, F1, P, R = evaluate(pred_logits, label_vectors, verbose=False)
    return test_loss, mAP, F1, P, R


def parse_args():
    # # define parser
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--test-data-path", type=str, required=True, help="The path to test features.")
    parser.add_argument("--dataset", type=str, default="coco2014", choices=["coco2014", "voc2012"])
    parser.add_argument("--cache-path", required=True, type=str, metavar="PATH", help="cache path")
    
    # training setting
    parser.add_argument("--train", action="store_true", help="Fine-tuning adapter if set.")
    parser.add_argument("--train-data-path", type=str, help="The path to training features.")
    parser.add_argument("--loss", type=str, choices=["CE"], default="CE", help="Loss type (default: CE).")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--num-epochs", type=int, default=20)
    
    # early stop
    parser.add_argument("--test-interval", type=int, default=5, help="Test model every 'test-interval' epochs (default: 5).")
    parser.add_argument("--patience", type=int, default=2, help="Stop training if the model does not improve for more than 'patience' test intervals (default: 2).")
    
    # tensorboard
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard if set.")
    parser.add_argument("--log-root", type=str, default="runs", help="The root path to save tensorboard logs (default: runs).")
    
    # save results
    parser.add_argument("--save-results", action="store_true", help="Save experiment results if set.")
    parser.add_argument("--result-root", type=str, default="results", help="The root path to save results (default: results).")
    
    # hyper parameter search
    parser.add_argument("--search-hp", action='store_true', help="Search hyper-parameters if set")
    parser.add_argument("--init-alpha", type=float, default=1.0, help="Residual Ratio alpha. Larger alpha denotes using more knowledge from the few-shot training set and less otherwise (default: 1.0).")
    parser.add_argument("--init-beta", type=float, default=1.0, help="Sharpness Ratio beta. When beta is large, only the nearby training samples to the test image have large influences to its class prediction and vice versa.(default: 1.0).")
    parser.add_argument("--search-scale", nargs="+", type=int, default=[5, 10], help="Search scale. (default: [5, 10])")
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
    
    # initialize
    method_name = "Tip-Adapter-F" if args.train else "Tip-Adapter"
    split_name = os.path.basename(args.train_data_path).split(".")[0]
    
    # initialize tensorboard writer
    writer = None
    if args.tensorboard and args.train:
        log_dir = os.path.join(args.log_root, # log root path
                               args.dataset, # dataset 
                               method_name, # method
                               split_name, # traing data
                               f"{args.loss}_bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_ep{args.num_epochs}") # hyperparameters
        writer = SummaryWriter(log_dir)

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
                                 batch_size=256, 
                                 shuffle=False, 
                                 num_workers=args.num_workers, 
                                 pin_memory=args.pin_memory)
    
    # load cache model
    assert os.path.exists(args.cache_path)
    cache = torch.load(args.cache_path, map_location=torch.device("cpu"))
    cache_keys = cache["keys"] # [NK, D]
    cache_values = cache["values"] # [NK, N]
    print(f"Load cache from file {args.cache_path}")

    # define Tip-Adapter
    adapter = TipAdapter(cache_keys, cache_values, beta=args.init_beta)
    adapter = adapter.to(device)

    # prepare for hyperparameter search
    all_clip_logits = []
    all_affinity = []
    def hook_fn(module, input, output):
        global all_affinity
        all_affinity.append(output.detach().cpu())
    hook = adapter.fc.register_forward_hook(hook_fn)

    # test
    adapter.eval()
    pred_logits = []
    label_vectors = []
    with torch.no_grad():
        for X, y, _ in tqdm(test_dataloader):
            X = X.to(device)
            # forward
            X = X / X.norm(dim=-1, keepdim=True)
            cache_logits = adapter(X)
            clip_logits = logit_scale * X @ text_features.t()
            all_clip_logits.append(clip_logits.cpu())
            tip_logits = clip_logits + cache_logits * args.init_alpha
            pred_logits.append(tip_logits.softmax(dim=-1).cpu())
            label_vectors.append(y)
    all_clip_logits = torch.cat(all_clip_logits, dim=0)
    all_affinity = torch.cat(all_affinity, dim=0)
    pred_logits = torch.cat(pred_logits, dim=0)
    label_vectors = torch.cat(label_vectors, dim=0)

    # evaluate
    mAP, F1, P, R = evaluate(pred_logits, label_vectors, verbose=True)

    # %% search hyperparameters
    if args.search_hp:
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
                cache_logits = ((-1) * (beta - beta * all_affinity)).exp() @ cache_values # [num_samples, num_classes]
                tip_logits = all_clip_logits + alpha * cache_logits # [num_samples, num_classes]
                # evaluate
                mAP, F1, P, R = evaluate(tip_logits.softmax(dim=-1), label_vectors, verbose=False)
                all_metrics = {"mAP": mAP, "F1": F1, "precision": P, "recall": R}
                # update the best metric
                if all_metrics[metric] > best_metric:
                    best_metric = all_metrics[metric]
                    best_alpha = alpha
                    best_beta = beta
                    # print(f"New best setting, alpha: {best_alpha:.2f}, beta: {best_beta:.2f}; {metric}: {best_metric:.6f}")
                    # print(f"mAP: {mAP:.6f}, F1: {F1:.6f}, Precision: {P:.6f}, Recall: {R:.6f}")
        end_time = time.time()
        print(f"After searching, the best {metric}: {best_metric:.6f}, cost: {(end_time-start_time):.1f}s")
        # reproduce the best metric
        cache_logits = ((-1) * (best_beta - best_beta * all_affinity)).exp() @ cache_values
        tip_logits = all_clip_logits + cache_logits * best_alpha
        mAP, F1, P, R = evaluate(tip_logits.softmax(dim=-1), label_vectors, verbose=False)
        print(f"The best setting: alpha: {best_alpha:.2f}, beta: {best_beta:.2f}")
        print(f"mAP: {mAP:.6f}, F1: {F1:.6f}, Precision: {P:.6f}, Recall: {R:.6f}")
        # set the alpha and adapter's beta to the best value ever found.
        adapter.set_beta(best_beta)
        args.init_alpha = best_alpha
    hook.remove()

    # %% Fine-tune adapter
    if args.train:
        assert args.train_data_path, "Expect training dataset path."
        # train loader
        train_dataset = FeatDataset(args.train_data_path)
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=args.batch_size, 
                                      shuffle=True, 
                                      num_workers=args.num_workers, 
                                      pin_memory=args.pin_memory)

        # get loss function
        if args.loss == "CE":
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        
        # optimizer
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

        # early stop mechanism
        best_mAP = 0.
        best_F1 = 0.
        best_mAP_epoch = 0
        best_F1_epoch = 0
        patience_counter = 0

        # train
        print("Start training.")
        for epoch in tqdm(range(args.num_epochs)):
            train_loss = train_loop(train_dataloader, adapter, loss_fn, optimizer)
            if writer:
                writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch+1)
                writer.add_scalar("Loss/train", train_loss, epoch+1)
            lr_scheduler.step()

            # test
            if (epoch + 1) % args.test_interval == 0 or (epoch + 1) == args.num_epochs:
                test_loss, mAP, F1, P, R = test_loop(test_dataloader, adapter, loss_fn)
                if writer:
                    writer.add_scalar("Loss/test", test_loss, epoch+1)
                    writer.add_scalar("mAP", mAP, epoch+1)
                    writer.add_scalar("F1", F1, epoch+1)
                    writer.add_scalar("Precision", P, epoch+1)
                    writer.add_scalar("Recall", R, epoch+1)
                else:
                    print("================================================")
                    print(f"[{epoch+1}/{args.num_epochs}] test loss: {test_loss:.6f}")
                    print(f"mAP: {mAP:.6f}, F1: {F1:.6f}, Precision: {P:.6f}, Recall: {R:.6f}")
                    print("================================================")


                # increment patience_counter If neither mAP nor F1 score improves
                if mAP > best_mAP or F1 > best_F1:
                    patience_counter = 0
                else:
                    patience_counter += 1

                # update best results
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_mAP_epoch = epoch + 1
                if F1 > best_F1:
                    best_F1 = F1
                    best_F1_epoch = epoch + 1

                # early stop if the patience threshold is exceeded
                if patience_counter > args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # summary
        print(f"The best mAP is {best_mAP:.6f}, obtained after {best_mAP_epoch} epochs training.")
        print(f"The best F1 score is {best_F1:.6f}, obtained after {best_F1_epoch} epochs training.")


    # %% save results
    if args.save_results:
        if args.train:
            result_data = {"loss": args.loss, 
                           "batch_size": args.batch_size, 
                           "lr": args.lr, 
                           "weight_decay": args.weight_decay, 
                           "num_epochs": args.num_epochs, 
                           "best_mAP": best_mAP.item(), 
                           "best_F1": best_F1.item(), 
                           "best_mAP_epoch": best_mAP_epoch,
                           "best_F1_epoch": best_F1_epoch}
        else:
            result_data = {"mAP": mAP.item(),
                           "F1": F1.item()}
        if args.search_hp:
            result_data["best_alpha"] = best_alpha
            result_data["best_beta"] = best_beta
        # write results
        result_path = os.path.join(args.result_root, 
                                   args.dataset, 
                                   method_name, 
                                   os.path.basename(args.cache_path).split(".")[0]+".csv")
        append_results(result_data, result_path)