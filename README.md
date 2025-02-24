# Tip-Adapter-SA

This is the official implementation for Tip-Adapter-SA.

## Requirements

We recommend using conda to configure the experimental environment as follows:
```bash
# create conda environment with Python3.9
conda create -n $ENV_NAME python=3.9

# intall PyTorch
pip install torch torchvision

# install other dependencies
pip install ftfy regex tqdm pycocotools opencv-python pandas scikit-learn
```

## Prepare Datasets

We conduct experiments on COCO2014 and Pascal VOC 2012 datasets. Download each dataset from the official website ([COCO2014](https://cocodataset.org/), [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/)) and put them under `./datasets` directory. The structure of `./datasets` is organized as follows:
```
datasets
    ├──coco2014
    │      ├──annotations
    │      ├──train2014
    │      ├──val2014
    │      └──...
    ├── voc2012
    │      ├──VOCdevkit
    │      │      ├─VOC2012
    │      │      │     ├──ImageSets
    │      │      │     ├──JPEGImages
    │      │      │     └──...
    │      │      └──...
    │      └──...
    └──...
```

We provide datasets in `npy` format in the `./imageset` directory, where `format_{train,val}_images.npy` and `format_{train,val}_labels.npy` store the image paths and corresponding one-hot labels of the training set and test set respectively. These files are generated using scripts in `./preproc` as follows (You do not need to run again):
```bash
cd preproc
python format_coco2014.py
python format_voc2012.py
```

In order to conduct experiments on our proposed few-shot single positive multi-label learning (FS-SPML) setting, we construct few-shot dataset (support set) from the training split of two datasets. In the $C$-way $K$-shot FS-SPML setting, $C$ corresponds to the number of classes in the dataset (20 for VOC2012 and 40 for COCO2014) and $K$ is set to 1, 2, 4, 8 and 16 following CLIP. Specifically, we sample $K$ images without replacement from training split for each class, retaining only the label of currently sampled class to form the support set. Considering that the quality of the support data significantly impacts the performance in few-shot scenarios, we generate 5 splits for each setting and report the mean mAP over 5 split in our paper. All split files are provided under `./splits` directory.

## Quick Start

You can conduct an experiment in 16-shot FS-SPML setting on VOC2012 by following the steps:

1. Save features for evaluation:

```bash
python save_MyCLIP_features.py --dataset voc2012 features/voc2012/MyCLIP/val.pt
```

2. Build cache model using few-shot support set:

```bash
# voc2012
python build_MyCLIP_cache.py --split-file splits/voc2012/exp1/16shots_filtered.txt --dataset voc2012 caches/voc2012/MyCLIP/exp1_16shots_filtered.pt
```

3. Perform adaptive inference using Tip-Adapter-SA:
```bash
python Tip-Adapter-SA.py --test-data-path features/voc2012/MyCLIP/val.pt --dataset voc2012 --cache-path caches/voc2012/MyCLIP/exp1_16shots_filtered.pt --k-shot 16 --search-hp
```

## Citation

If you think our work is helpful to your research, please consider citing our paper: