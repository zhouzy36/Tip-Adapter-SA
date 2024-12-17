# coding=utf-8
import numpy as np
import os
import pickle
from PIL import Image
from collections.abc import Iterable
from typing import List, Union
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pycocotools.coco import COCO


class ResizeToPatchSizeDivisible(torch.nn.Module):
    def __init__(self, patch_size, interpolation=transforms.InterpolationMode.BICUBIC):
        super().__init__()
        self.patch_size = patch_size
        self.interpolation = interpolation

    def forward(self, img):
        if not isinstance(img, torch.Tensor): # PIL image
            width, height = img.size
        else: # [C, H, W]
            height, width = img.shape[-2:]
        return transforms.functional.resize(img, size=(int(np.ceil(height/self.patch_size))*self.patch_size, int(np.ceil(width/self.patch_size))*self.patch_size), interpolation=self.interpolation)
    


class NumpyDataset(Dataset):
    def __init__(self, 
                 img_root: str, 
                 file_paths: np.array, 
                 label_list: np.array, 
                 transform=None, 
                 one_hot_label: bool=True):
        assert os.path.exists(img_root)
        self.img_root = img_root
        self.file_paths = file_paths
        self.label_list = label_list
        self.transform = transform
        self.num_classes = self.label_list.shape[-1]
        self.one_hot_label = one_hot_label
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img_file = self.file_paths[idx]
        img_path = os.path.join(self.img_root, img_file)
        img = Image.open(img_path).convert('RGB')
        # transform
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.functional.to_tensor(img)

        label_vector = self.label_list[idx]
        if self.one_hot_label:
            label = label_vector
        else:
            label = np.nonzero(label_vector)[0]
        return img, torch.from_numpy(label)

    def getImgIds(self, imgIds: List[int]=[], catIds: Union[int, List[int]]=[]):
        # type check
        if isinstance(catIds, int):
            catIds = [catIds]
        if not isinstance(imgIds, List):
            assert isinstance(imgIds, Iterable)
            imgIds = list(imgIds)

        assert len(catIds) < self.num_classes

        if not catIds:
            catIds = list(range(self.num_classes))
        if not imgIds:
            imgIds = list(range(self.label_list.shape[0]))
            
        candidateIds = imgIds
        for cat in catIds:
            if not candidateIds:
                return []
            labels = self.label_list[candidateIds, cat]
            idx = labels.nonzero()[0]
            candidateIds = [candidateIds[id] for id in idx]
        return candidateIds



class TxtDataset(Dataset):
    def __init__(self, 
                 dataset: str, 
                 img_root: str, 
                 file_paths: List, 
                 label_list: List, 
                 transform=None, 
                 one_hot_label: bool=True):
        assert os.path.exists(img_root)
        self.img_root = img_root
        self.file_paths = file_paths
        self.label_list = label_list
        self.transform = transform
        self.one_hot_label = one_hot_label

        if dataset == "voc2012":
            self.num_classes = 20
        elif dataset == "coco2014":
            self.num_classes = 80
        else:
            raise NotImplementedError
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img_file = self.file_paths[idx]
        img_path = os.path.join(self.img_root, img_file)
        img = Image.open(img_path).convert('RGB')
        # transform
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.functional.to_tensor(img)

        label = self.label_list[idx]
        label = [int(lid) for lid in label]
        if self.one_hot_label:
            label_vector = torch.zeros(self.num_classes)
            label_vector[label] = 1
            label = label_vector
        else:
            label = torch.tensor(label)

        return img, label



class FeatDataset(Dataset):
    def __init__(self, data_path: str):
        assert os.path.exists(data_path)
        data = torch.load(data_path, map_location=torch.device("cpu"))
        self.feats = data["feats"]
        self.labels = data["labels"]
        self.logits = data["logits"]
        self.feat_dim = self.feats.shape[-1]

    def __len__(self):
        return self.feats.shape[0]
    
    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx], self.logits[idx]
    
    def get_all_labels(self):
        return self.labels
    
    def get_all_logits(self):
        return self.logits
    
    def get_all_features(self):
        return self.feats



class LaSOSplitDataset(Dataset):
    def __init__(self, 
                 coco_root: str,
                 split_file_path: str,
                 transform=None, 
                 one_hot_label: bool=True):
        assert os.path.exists(split_file_path), f"Split file {split_file_path} does not exist."
        self.transform = transform
        self.one_hot_label = one_hot_label
        
        # invoke COCO api
        self.coco = COCO(os.path.join(coco_root, "annotations/instances_train2014.json"))
        self.img_root = os.path.join(coco_root, "train2014")
        self.coco_img_ids = self.coco.getImgIds()
        
        # load dict-style dataset from split file
        with open(split_file_path, 'rb') as f:
            data = pickle.load(f) # key: COCO label id, value: image id
        self.num_classes = len(data.keys())
        
        dataset = {} # LaSO dataset dict, key: image id, value: label id range from 0
        COCO2LaSO = {} # Map COCO label id to label id
        LaSO2COCO = {} # Map label id to COCO label id
        for label_id, coco_label_id in enumerate(data.keys()):
            for img_id in data[coco_label_id]:
                if img_id in dataset:
                    dataset[img_id].append(label_id)
                else:
                    dataset[img_id] = [label_id]
            # build label map between LaSO dataset and COCO
            COCO2LaSO[coco_label_id] = label_id
            LaSO2COCO[label_id] = coco_label_id

        self.dataset = dataset
        self.img_list = list(dataset.keys())
        self.COCO2LaSO = COCO2LaSO
        self.LaSO2COCO = LaSO2COCO

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx: int):
        img_id = self.img_list[idx]
        img_path = self.load_image_path(img_id)
        img = Image.open(img_path).convert('RGB')
        # transform
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.functional.to_tensor(img)
        
        label = self.dataset[img_id]
        if self.one_hot_label:
            label_vector = torch.zeros(self.num_classes)
            label_vector[label] = 1
            label = label_vector
        else:
            label = torch.tensor(label)

        return img, label
    
    def load_image_path(self, img_id: int):
        real_img_id = self.coco_img_ids[img_id]
        img_info = self.coco.loadImgs(real_img_id)[0]
        img_path = os.path.join(self.img_root, img_info['file_name'])
        return img_path



if __name__ == "__main__":
    # NumpyDataset example
    img_root = "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
    image_file = "imageset/voc2012/formatted_train_images.npy"
    full_label_file = "imageset/voc2012/formatted_train_labels.npy"
    image_list = np.load(image_file)
    label_list = np.load(full_label_file)
    dataset = NumpyDataset(img_root, image_list, label_list)

    # TxtDataset example
    img_root = "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
    file_list = tuple(open("splits/voc2012/exp1/1shots_filtered.txt", "r"))
    file_list = [id_.rstrip().split(" ") for id_ in file_list]
    image_list = [x[0] for x in file_list]
    label_list = [x[1:] for x in file_list]
    dataset = TxtDataset("voc2012", img_root, image_list, label_list)

    # FeatDataset example
    data_path = "features/voc2012/CLIP/val_all.pt"
    dataset = FeatDataset(data_path)

    # LaSOSplitDataset example
    coco_root = "datasets/coco2014"
    split_file_path = "splits/LaSO/5shotRun1ClassIdxDict.pkl"
    dataset = LaSOSplitDataset(coco_root, split_file_path)