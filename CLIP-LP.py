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
from utils import get_class_names, evaluation, append_results
from loss import IULoss, ANLoss, WANLoss


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    num_batches = len(dataloader)
    train_loss = 0.

    for X, y, _ in dataloader:
        # move data to device
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # record loss
        train_loss += loss.item()
        
    train_loss /= num_batches

    return train_loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0.
    pred_logits = []
    label_vectors = []
    with torch.no_grad():
        for X, y, _ in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            pred_logits.append(F.sigmoid(pred).detach().cpu())
            label_vectors.append(y.detach().cpu())
    test_loss /= num_batches
    # evaluate
    pred_logits = torch.cat(pred_logits, dim=0)
    label_vectors = torch.cat(label_vectors, dim=0)
    ap, F1, P, R = evaluation(pred_logits, label_vectors, verbose=False)
    return test_loss, ap, F1, P, R


def parse_args():
    # define arguments
    parser = argparse.ArgumentParser(description="Linear-probe CLIP: Train linear classifier using training data features.")
    parser.add_argument("--train-data-path", type=str, required=True, help="The path to training features.")
    parser.add_argument("--test-data-path", type=str, required=True, help="The path to test features.")
    parser.add_argument("--dataset", type=str, choices=["voc2012", "coco2014"], default="coco2014", help="Experimental dataset (default: voc2012).")
    
    # loss
    parser.add_argument("--loss", type=str, choices=["CE", "IU", "AN", "WAN", "AN-LS"], default="CE", help="Loss type (default: CE).")
    
    # training parameters
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
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
    # parse arguments
    args = parse_args()
    device = torch.device("cuda:0")
    
    # initialize tensorboard writer
    writer = None
    if args.tensorboard:
        log_dir = os.path.join(args.log_root, # log root path
                               args.dataset, # dataset 
                               "CLIP-LP", # method
                               os.path.basename(args.train_data_path).split(".")[0], # traing data
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

    # define linear classifier
    classifier = nn.Sequential(nn.Linear(feat_dim, NUM_CLASSES))
    classifier = classifier.to(device)

    # get loss function
    if args.loss == "CE":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "IU":
        loss_fn = IULoss()
    elif args.loss == "AN":
        loss_fn = ANLoss()
    elif args.loss == "AN-LS":
        loss_fn = ANLoss(epsilon=0.1)
    elif args.loss == "WAN":
        loss_fn = WANLoss(gamma=1/(NUM_CLASSES-1))
    else:
        raise NotImplementedError

    # optimizer
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, eps=1e-4, weight_decay=args.weight_decay)

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
        train_loss = train_loop(train_dataloader, classifier, loss_fn, optimizer)
        if writer:
            writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch+1)
            writer.add_scalar("Loss/train", train_loss, epoch+1)
        lr_scheduler.step()

        # test
        if (epoch + 1) % args.test_interval == 0:
            test_loss, ap, F1, P, R = test_loop(test_dataloader, classifier, loss_fn)
            mAP = torch.mean(ap)
            # print("================================================")
            # print(f"[{epoch+1}/{args.num_epochs}] test loss: {test_loss:.6f}")
            # print(f"mAP: {mAP:.6f}, F1: {F1:.6f}, Precision: {P:.6f}, Recall: {R:.6f}")
            # print("================================================")
            if writer:
                writer.add_scalar("Loss/test", test_loss, epoch+1)
                writer.add_scalar("mAP", mAP, epoch+1)
                writer.add_scalar("F1", F1, epoch+1)
                writer.add_scalar("Precision", P, epoch+1)
                writer.add_scalar("Recall", R, epoch+1)

            # increment patience_counter If neither mAP nor F1 score improves
            if mAP > best_mAP:
                best_mAP = mAP
                best_mAP_epoch = epoch + 1
                patience_counter = 0
            elif F1 > best_F1:
                best_F1 = F1
                best_F1_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            # early stop if the patience threshold is exceeded
            if patience_counter > args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # final test
    test_loss, ap, F1, P, R = test_loop(test_dataloader, classifier, loss_fn)
    print("================================================")
    print(f"[{epoch+1}/{args.num_epochs}] test loss: {test_loss:.6f}")
    print(f"mAP: {torch.mean(ap):.6f}, F1: {F1:.6f}, Precision: {P:.6f}, Recall: {R:.6f}")
    print("================================================")
    if writer:
        writer.add_scalar("Loss/test", test_loss, args.num_epochs)
        writer.add_scalar("mAP", torch.mean(ap), args.num_epochs)
        writer.add_scalar("F1", F1, args.num_epochs)
        writer.add_scalar("Precision", P, args.num_epochs)
        writer.add_scalar("Recall", R, args.num_epochs)
        writer.close()

    if mAP > best_mAP:
        best_mAP = mAP
        best_mAP_epoch = epoch + 1
    if F1 > best_F1:
        best_F1 = F1
        best_F1_epoch = epoch + 1

    # summary
    print(f"The best mAP is {best_mAP:.6f}, obtained after {best_mAP_epoch} epochs training.")
    print(f"The best F1 score is {best_F1:.6f}, obtained after {best_F1_epoch} epochs training.")

    # save results
    if args.save_results:
        result_data = {"loss": args.loss, 
                       "batch_size": args.batch_size, 
                       "lr": args.lr, 
                       "weight_decay": args.weight_decay, 
                       "num_epochs": args.num_epochs, 
                       "best_mAP": best_mAP.item(), 
                       "best_F1": best_F1.item(), 
                       "best_mAP_epoch": best_mAP_epoch,
                       "best_F1_epoch": best_F1_epoch}
        print(result_data)
        result_path = os.path.join(args.result_root, args.dataset, "CLIP-LP", os.path.basename(args.train_data_path).split(".")[0]+".csv")
        append_results(result_data, result_path)