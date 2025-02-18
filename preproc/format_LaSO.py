import argparse
import numpy as np
import os
from pycocotools.coco import COCO

pp = argparse.ArgumentParser(description="Format LaSO test data.")
pp.add_argument("--load-path", type=str, default="../datasets/coco2014", help="Path to a directory containing a copy of the COCO dataset.")
pp.add_argument("--save-path", type=str, default="../imageset/LaSO", help="Path to output directory.")
args = pp.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

img_root = os.path.join(args.load_path, "val2014")
coco = COCO(os.path.join(args.load_path, "annotations/instances_val2014.json"))

label_set = {'bicycle', 'boat', 'stop sign', 'bird', 'backpack', 'frisbee', 'snowboard', 'surfboard', 'cup', 'fork', 'spoon', 'broccoli', 'chair', 'keyboard', 'microwave', 'vase'}
NUM_CLASSES = len(label_set)
label_set_ids = coco.getCatIds(catNms=list(label_set))
cat2label = {cat_id: i for i, cat_id in enumerate(label_set_ids)}
label2cat = {v: k for k, v in cat2label.items()}

img_ids = coco.getImgIds()

image_list = []
label_list = []
for i, img_id in enumerate(sorted(img_ids)):
    # load image information
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    image_path = os.path.join(img_root, file_name)
    
    # load annotations related to the image being processed
    annotation_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(annotation_ids)
    category_ids = {ann['category_id'] for ann in annotations}

    # filter category ids
    filtered_category_ids = [cat_id for cat_id in category_ids if cat_id in label_set_ids]
    label = [cat2label[cat_id] for cat_id in filtered_category_ids]

    if len(label) > 0:
        image_list.append(os.path.join("val2014", file_name))
        label_vector = np.zeros(NUM_CLASSES)
        label_vector[label] = 1
        label_list.append(label_vector)

# save formatted image paths and labels
assert len(image_list) == len(label_list)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
np.save(os.path.join(args.save_path, "formatted_val_images.npy"), image_list)
np.save(os.path.join(args.save_path, "formatted_val_labels.npy"), np.stack(label_list, axis=0))