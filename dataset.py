# coding=utf-8
import numpy as np
import os
from PIL import Image
from collections.abc import Iterable
from typing import List, Union
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


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



if __name__ == "__main__":
    # NumpyDataset example
    img_root = "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
    image_file = "imageset/voc2012/formatted_train_images.npy"
    full_label_file = "imageset/voc2012/formatted_train_labels.npy"
    image_list = np.load(image_file)
    full_label_list = np.load(full_label_file)
    dataset = NumpyDataset(img_root, image_list, full_label_list)

    # TxtDataset example
    img_root = "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
    file_list = tuple(open("splits/voc2012/exp1/1shots_filtered.txt", "r"))
    file_list = [id_.rstrip().split(" ") for id_ in file_list]
    image_list = [x[0] for x in file_list]
    label_list = [x[1:] for x in file_list]
    dataset = TxtDataset("voc2012", img_root, image_list, label_list)