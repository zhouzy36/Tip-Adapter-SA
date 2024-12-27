# coding=utf-8
# Copy from CoOp (https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py) and modify cfg in PromptLearner
import argparse
import clip
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip
from tqdm import tqdm

from utils import get_class_names, get_test_dataset, get_split_dataset, evaluate, setup_seed, append_results
from warmup_scheduler import ConstantWarmupScheduler, LinearWarmupScheduler
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

"""
Example:
python CoOp.py --dataset ${dataset} --train-data-path splits/${dataset}/exp${split}/${k}shots_filtered.txt --batch-size ${bs} --lr ${lr} --num-epochs ${epoch} --warmup constant --warmup-epoch 1 --warmup-lr 1.e-5 --tensorboard --save-results
"""

# %% model: copy from CoOp (https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py)
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x



class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX # number of context tokens
        ctx_init = cfg.CTX_INIT # 
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT_SIZE[0] # input size (default: 224)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.CSC: # class-specific context (False or True)
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION # class token position (end or middle)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        h, w = image.shape[-2:] # new add
        image_features = self.image_encoder(image.type(self.dtype), h, w)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

# %% My Code

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    num_batches = len(dataloader)
    train_loss = 0.

    for X, y in dataloader:
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
        for X, y in dataloader:
            # move data to device
            X = X.to(device)
            y = y.to(device)
            
            # inference and compute loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # record loss and prediction
            test_loss += loss.item()
            pred_logits.append(pred.softmax(dim=-1).cpu())
            label_vectors.append(y.cpu())

    test_loss /= num_batches
    pred_logits = torch.cat(pred_logits, dim=0)
    label_vectors = torch.cat(label_vectors, dim=0)

    return test_loss, pred_logits, label_vectors


def parse_args():
    # define parser
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--dataset", type=str, required=True, choices=["coco2014", "voc2012"])
    parser.add_argument("--train-data-path", type=str, required=True, help="The path to few-shot dataset split file.")
    
    # CoOp parameters
    parser.add_argument("--N_CTX", type=int, default=16, help="The number of context vectors (default: 16).")
    parser.add_argument("--CSC", action="store_true", help="Use class-specific context if set.")
    parser.add_argument("--CTX_INIT", type=str, default="", help="Initialization words (default: '').")
    parser.add_argument("--PREC", type=str, choices=["fp16", "fp32", "amp"], default="fp32", help="Specify the model precision (default: 'fp32').")
    parser.add_argument("--CLASS_TOKEN_POSITION", type=str, choices=["front", "middle", "end"], default="end", help="Specify the position of class token (default: 'end).")

    # loss
    parser.add_argument("--loss", type=str, default="CE", choices=["CE"], help="Loss type (default: CE).")
    
    # training parameters
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 16).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate (default: 0.001).")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-epochs", type=int, default=200)

    # warmup settings
    parser.add_argument("--warmup", type=str, choices=["constant", "linear"], help="Specify the warmup type (e.g., 'constant' or 'linear').")
    parser.add_argument("--warmup-epochs", type=int, help="The number of warmup epochs.")
    # constant warmup
    constant_group = parser.add_argument_group("Parameters of constant warmup.")
    constant_group.add_argument("--warmup-lr", type=float, help="The number of warmup epochs.")
    # linear warmup
    linear_group = parser.add_argument_group("Parameters of linear warmup.")
    linear_group.add_argument("--min-lr", type=float, help="The minimum learning rate of linear warmup.")

    # early stop mechanism
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
    
    # check args
    if args.warmup is not None:
        assert args.warmup_epochs is not None, f"--warmup {args.warmup} requires --warmup-epochs."
        if args.warmup == "constant":
            assert args.warmup_lr is not None, f"--warmup {args.warmup} requires --warmup-lr."
        else:
            assert args.min_lr is not None, f"--warmup {args.warmup} requires --min-lr."
    
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
    method = "CoOp"
    split_name = args.train_data_path.split("/")[-2] + "_" + os.path.basename(args.train_data_path).split(".")[0]

    # initialize tensorboard writer
    writer = None
    if args.tensorboard:
        log_dir = os.path.join(args.log_root, # log root path
                               args.dataset, # dataset 
                               method, # method
                               split_name,
                               f"{args.loss}_bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_ep{args.num_epochs}") # hyperparameters
        writer = SummaryWriter(log_dir)

    # load CLIP model
    model_path = "pretrained_models/ViT-B-16.pt"
    patch_size = 16
    clip_model, preprocess = clip.load(model_path, torch.device("cpu"))

    # load class name
    class_names, NUM_CLASSES = get_class_names(args.dataset)
    print("The number of classes in dataset:", NUM_CLASSES)
    print("The number of classes in vocabulary:", len(class_names))

    # dataloader
    # train loader
    train_transforms = transforms.Compose([
        RandomResizedCrop(size=(224, 224)),
        RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    train_dataset = get_split_dataset(args.dataset, args.train_data_path, transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers, 
                                  pin_memory=args.pin_memory)

    # test loader
    test_dataset = get_test_dataset(args.dataset, transform=preprocess)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=128, 
                                 shuffle=False, 
                                 num_workers=args.num_workers, 
                                 pin_memory=args.pin_memory)

    # define configs for CoOp
    class Config(object):
        N_CTX = args.N_CTX
        CSC = args.CSC
        CTX_INIT = args.CTX_INIT
        PREC = args.PREC
        CLASS_TOKEN_POSITION = args.CLASS_TOKEN_POSITION
        INPUT_SIZE = (224, 224)

    cfg = Config()
    
    # build CoOp model
    model = CustomCLIP(cfg, class_names, clip_model)
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    model.to(device)

    # get loss
    if args.loss == "CE":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # params = [p for p in model.prompt_learner.parameters()] # equivalent code
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # learning rate scheduler
    if args.warmup:
        post_warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs-args.warmup_epochs)
        if args.warmup == "constant":
            lr_scheduler = ConstantWarmupScheduler(optimizer, post_warmup_scheduler, warmup_epoch=args.warmup_epochs, warmup_lr=args.warmup_lr)
        elif args.warmup == "linear":
            lr_scheduler = LinearWarmupScheduler(optimizer, post_warmup_scheduler, warmup_epoch=args.warmup_epochs, min_lr=args.min_lr)
        else:
            raise NotImplementedError
    else:
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
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        if writer:
            writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch+1)
            writer.add_scalar("Loss/train", train_loss, epoch+1)
        lr_scheduler.step()

        # test
        if (epoch + 1) % args.test_interval == 0 or (epoch + 1) == args.num_epochs:
            test_loss, pred_logits, label_vectors = test_loop(test_dataloader, model, loss_fn)
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
                print("================================================")

            # increment patience counter If neither mAP nor F1 score improves
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
        result_data = {"N_CTX": args.N_CTX,
                       "CSC": args.CSC,
                       "CLASS_TOKEN_POSITION": args.CLASS_TOKEN_POSITION,
                       "loss": args.loss, 
                       "batch_size": args.batch_size, 
                       "lr": args.lr, 
                       "weight_decay": args.weight_decay, 
                       "num_epochs": args.num_epochs, 
                       "warmup": args.warmup,
                       "best_mAP": best_mAP.item(), 
                       "best_F1": best_F1.item(), 
                       "best_mAP_epoch": best_mAP_epoch,
                       "best_F1_epoch": best_F1_epoch}
        print(result_data)
        result_path = os.path.join(args.result_root, 
                                   args.dataset, 
                                   method, 
                                   f"{split_name}.csv")
        append_results(result_data, result_path)