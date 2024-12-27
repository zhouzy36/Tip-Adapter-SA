# coding=utf-8
import argparse
import clip
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import FeatDataset
from utils import get_class_names, evaluate, append_results, setup_seed, search_best_threshold

"""
Examples:
python CLIP-Adapter.py --train-data-path features/${dataset}/CLIP/exp${split}_${k}shots_filtered.pt --test-data-path features/${dataset}/CLIP/val_all.pt --dataset ${dataset} --batch-size ${bs} --lr ${lr} --num-epochs ${epoch} --tensorboard --save-results
"""


class CLIPAdapter(nn.Module):
    def __init__(self, feat_dim, reduction=2, ratio=0.2) -> None:
        super(CLIPAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(feat_dim // reduction, feat_dim, bias=False),
            nn.ReLU()
        )
        self.ratio = ratio

    def forward(self, x):
        out = self.fc(x)
        out = self.ratio * out + (1 - self.ratio) * x
        return out


def train_loop(dataloader, adapter, loss_fn, optimizer):
    global logit_scale, text_features # variables from main
    adapter.train()
    num_batches = len(dataloader)
    train_loss = 0.

    for X, y, _ in dataloader:
        # move data to device
        X = X.to(device)
        y = y.to(device)

        # compute prediction and loss
        X = adapter(X)
        X = X / X.norm(dim=-1, keepdim=True)
        pred = logit_scale * X @ text_features.t()
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # record loss
        train_loss += loss.item()
        
    train_loss /= num_batches

    return train_loss


def test_loop(dataloader, adapter, loss_fn):
    global logit_scale, text_features # variables from main
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
            X = adapter(X)
            X = X / X.norm(dim=-1, keepdim=True)
            pred = logit_scale * X @ text_features.t()
            loss = loss_fn(pred, y)
            
            # record loss and prediction
            pred_logits.append(pred.softmax(dim=-1).cpu())
            test_loss += loss.item()
            label_vectors.append(y.cpu())
    test_loss /= num_batches
    
    pred_logits = torch.cat(pred_logits, dim=0)
    label_vectors = torch.cat(label_vectors, dim=0)

    return test_loss, pred_logits, label_vectors


def parse_args():
    # define parser
    parser = argparse.ArgumentParser(description="CLIP-Adapter: Fine-tuning lightweight adapters with residual connections.")

    # data
    parser.add_argument("--train-data-path", type=str, required=True, help="The path to training features.")
    parser.add_argument("--test-data-path", type=str, required=True, help="The path to test features.")
    parser.add_argument("--dataset", type=str, choices=["voc2012", "coco2014"], default="coco2014", help="Experimental dataset (default: voc2012).")

    # CLIP-Adapter parameters
    parser.add_argument("--reduction", type=int, default=4, help="The dimensionality reduction ratio of adapter hidden layer (default: 4). ")
    parser.add_argument("--alpha", type=float, default=0.2, help="The residual ratio of adapter  (default: 0.2).")
    
    # loss
    parser.add_argument("--loss", type=str, default="CE", choices=["CE"], help="Loss type (default: CE).")
    
    # training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size (default: 16).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=50)
    
    # early stop
    parser.add_argument("--test-interval", type=int, default=5, help="Test model every 'test-interval' epochs (default: 5).")
    parser.add_argument("--patience", type=int, default=2, help="Stop training if the model does not improve for more than 'patience' test intervals (default: 2).")
    
    # tensorboard
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard if set.")
    parser.add_argument("--log-root", type=str, default="runs", help="The root path to save tensorboard logs (default: runs).")
    
    # save results
    parser.add_argument("--save-results", action="store_true", help="Save experiment results if set.")
    parser.add_argument("--result-root", type=str, default="results", help="The root path to save results (default: results).")

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
    method_name = "CLIP-Adapter"
    split_name = os.path.basename(args.train_data_path).split(".")[0]
    
    # initialize tensorboard writer
    writer = None
    if args.tensorboard:
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

    # classifier weights
    with torch.no_grad():
        text_features = clip.encode_text_with_prompt_ensemble(model, class_names, device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # [num_classes, d]
    feat_dim = text_features.shape[-1]
    print(text_features.shape, text_features.dtype)

    # dataloader
    # train loader
    train_dataset = FeatDataset(args.train_data_path)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers, 
                                  pin_memory=args.pin_memory)
    # test loader
    test_dataset = FeatDataset(args.test_data_path)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=256, 
                                 shuffle=False, 
                                 num_workers=args.num_workers, 
                                 pin_memory=args.pin_memory)

    # define visual adapter
    visual_adapter = CLIPAdapter(feat_dim, reduction=args.reduction, ratio=args.alpha)
    visual_adapter = visual_adapter.to(device)

    # get loss function
    if args.loss == "CE":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # optimizer
    optimizer = torch.optim.AdamW(visual_adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
        train_loss = train_loop(train_dataloader, visual_adapter, loss_fn, optimizer)
        if writer:
            writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch+1)
            writer.add_scalar("Loss/train", train_loss, epoch+1)
        lr_scheduler.step()

        # test
        if (epoch + 1) % args.test_interval == 0 or (epoch + 1) == args.num_epochs:
            test_loss, pred_logits, label_vectors = test_loop(test_dataloader, visual_adapter, loss_fn)
            mAP, F1, P, R = evaluate(pred_logits, label_vectors, verbose=False)
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
                search_best_threshold(pred_logits, label_vectors, verbose=True)
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

    # save results
    if args.save_results:
        result_data = {"reduction": args.reduction,
                       "alpha": args.alpha,
                       "loss": args.loss, 
                       "batch_size": args.batch_size, 
                       "lr": args.lr, 
                       "weight_decay": args.weight_decay, 
                       "num_epochs": args.num_epochs, 
                       "best_mAP": best_mAP.item(), 
                       "best_F1": best_F1.item(), 
                       "best_mAP_epoch": best_mAP_epoch,
                       "best_F1_epoch": best_F1_epoch}
        print(result_data)
        result_path = os.path.join(args.result_root, 
                                   args.dataset, 
                                   method_name, 
                                   f"{split_name}.csv")
        append_results(result_data, result_path)