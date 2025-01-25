# coding=utf-8
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import FeatDataset
from module import FeatureExtractor
from utils import setup_seed, get_class_names, get_split_dataset, get_test_dataset, get_loss_fn, evaluate, append_results


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    num_batches = len(dataloader)
    train_loss = 0.

    for batch in dataloader:
        if args.train_mode == "linear":
            X, y, _ = batch
        else:
            X, y = batch

        # move data to device
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
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
        for batch in dataloader:
            if args.train_mode == "linear":
                X, y, _ = batch
            else:
                X, y = batch
            
            # move data to device
            X = X.to(device)
            y = y.to(device)

            # compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # record loss
            test_loss += loss.item()
            pred_logits.append(F.sigmoid(pred).cpu())
            label_vectors.append(y.cpu())
    test_loss /= num_batches

    # evaluate
    pred_logits = torch.cat(pred_logits, dim=0)
    label_vectors = torch.cat(label_vectors, dim=0)
    mAP, F1, P, R = evaluate(pred_logits, label_vectors, verbose=False)
    
    return test_loss, mAP, F1, P, R


def parse_args():
    # define parser
    parser = argparse.ArgumentParser(description="")

    # data
    parser.add_argument("--train-data-path", type=str, required=True, help="The path to training dataset/features.")
    parser.add_argument("--test-data-path", type=str, help="The path to test dataset/features.")
    parser.add_argument("--dataset", type=str, choices=["voc2012", "coco2014", "LaSO"], default="coco2014", help="Experimental dataset (default: coco2014).")
    parser.add_argument("--image-size", type=int, choices=[224, 448], default=448, help="Image size (default: 448).")
    
    # model
    parser.add_argument("--train-mode", type=str, choices=["end-to-end", "linear"], default="end-to-end", help="Train mode (default: end-to-end).")
    parser.add_argument("--backbone", type=str, choices=["resnet50", "resnet101", "inception_v3", "vit_b_16"], default="resnet50")
    
    # loss
    parser.add_argument("--loss", type=str, choices=["IU", "AN", "WAN", "AN-LS", "EM"], default="AN", help="Loss type (default: AN).")

    # TODO: pseudo-labeling algorithm
    
    # training parameters
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16).")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--num-epochs", type=int, default=50)
    
    # early stop
    parser.add_argument("--test-interval", type=int, default=5, help="Test model every 'test-interval' epochs (default: 5).")
    parser.add_argument("--patience", type=int, default=1, help="Stop training if the model does not improve for more than 'patience' test intervals (default: 1).")
    
    # tensorboard
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard if set.")
    parser.add_argument("--log-root", type=str, default="runs", help="The root path to save tensorboard logs (default: runs).")
    
    # save results
    parser.add_argument("--save-results", action="store_true", help="Save experiment results if set.")
    parser.add_argument("--result-root", type=str, default="results", help="The root path to save results (default: results).")

    # parse args
    args = parser.parse_args()

    # check arguments
    assert args.train_mode != "linear" or args.test_data_path

    print(args)
    
    return args

if __name__ == "__main__":
    # set up random seed
    setup_seed()

    # parse arguments
    args = parse_args()

    # initialize device
    device = torch.device("cuda:0")

    # define method name and split name
    method_name = f"{args.backbone}@{args.image_size}_{args.loss}_{args.train_mode}"
    if args.train_data_path.endswith(".txt"):
        split_name = args.train_data_path.split("/")[-2] + "_" + os.path.basename(args.train_data_path).split(".")[0]
    else:
        split_name = os.path.basename(args.train_data_path).split(".")[0]

    # initialize tensorboard writer
    writer = None
    if args.tensorboard:
        log_dir = os.path.join(args.log_root, # log root path
                               args.dataset, # dataset 
                               method_name, # method name
                               split_name,
                               f"{args.loss}_bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_ep{args.num_epochs}") # hyperparameters
        writer = SummaryWriter(log_dir)

    # data
    class_names, NUM_CLASSES = get_class_names(args.dataset)
    print("The number of classes in dataset:", NUM_CLASSES)
    print("The number of classes in vocabulary:", len(class_names))

    if args.train_mode == "linear":
        train_dataset = FeatDataset(args.train_data_path)
        test_dataset = FeatDataset(args.test_data_path)
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        train_dataset = get_split_dataset(args.dataset, args.train_data_path, transform=train_transform)

        test_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        test_dataset = get_test_dataset(args.dataset, transform=test_transform)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers, 
                                  pin_memory=args.pin_memory)

    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=128, 
                                 shuffle=False)
    
    # model
    if args.train_mode == "linear":
        feat_dim = train_dataset.get_feat_dim()
        model = torch.nn.Linear(feat_dim, NUM_CLASSES, bias=True)
    else:
        feature_extractor = FeatureExtractor(args.backbone, image_size=args.image_size, weights='IMAGENET1K_V1')
        feat_dim = feature_extractor.get_feat_dim()
        linear_classifier = torch.nn.Linear(feat_dim, NUM_CLASSES, bias=True)
        model = nn.Sequential(feature_extractor, linear_classifier)
    model.to(device)
        
    # loss
    loss_fn = get_loss_fn(args.loss, args.dataset)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TODO: learning rate scheduler

    # early stop mechanism
    best_mAP = 0.
    best_F1 = 0.
    best_mAP_epoch = 0
    best_F1_epoch = 0
    patience_counter = 0

    # train
    print("Start training.")
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        if writer:
            # writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch+1)
            writer.add_scalar("Loss/train", train_loss, epoch+1)
        # lr_scheduler.step()

        # test
        if (epoch + 1) % args.test_interval == 0 or (epoch + 1) == args.num_epochs:
            test_loss, mAP, F1, P, R = test_loop(test_dataloader, model, loss_fn)
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

    # save results
    if args.save_results:
        result_data = {"train_mode": args.train_mode,
                       "backbone": args.backbone,
                       "image_size": args.image_size,
                       "loss": args.loss, 
                       "batch_size": args.batch_size, 
                       "lr": args.lr, 
                       "num_epochs": args.num_epochs, 
                       "best_mAP": best_mAP.item(), 
                       "best_F1": best_F1.item(), 
                       "best_mAP_epoch": best_mAP_epoch,
                       "best_F1_epoch": best_F1_epoch}
        print(result_data)
        result_path = os.path.join(args.result_root, args.dataset, method_name, f"{split_name}.csv")
        append_results(result_data, result_path)