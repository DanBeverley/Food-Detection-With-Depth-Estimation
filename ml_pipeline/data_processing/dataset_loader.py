import os
from typing import Callable

import cv2
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

class UECFoodDataset(Dataset):
    def __init__(self, root_dir:str, transform:Callable, image_ext:str):
        """
        Args:
            root_dir (str): Root directory containing subfolders for each food category.
            transform (callable, optional): Albumentations transformation pipeline.
            image_ext (str): Extension of the image files (default '.jpg').
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_ext = image_ext
        self.data = []            # Each dict corresponds to one image
        self.id_to_category = {}  # Mapping numerical id to category name
        self._read_category_file()
        self._load_dataset()

    def _read_category_file(self):
        categories_file = os.path.join(self.root_dir, "category.txt")
        if os.path.exists(categories_file):
            with open(categories_file, "r") as f:
                lines = f.readlines()
            # Skip the header line
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts)<2:
                    continue
                try:
                    cat_id = int(parts[0])
                except ValueError:
                    continue
                cat_name = ' '.join(parts[1:])
                self.id_to_category[cat_id] = cat_name
        else:
            print(f"⚠️ Warning: category.txt not found in the root directory.")
            self.id_to_category = None

    def _load_dataset(self):
        for category in sorted(os.listdir(self.root_dir)):
            category_path = os.path.join(self.root_dir, category)
            if not os.path.isdir(category_path):
                continue
            try:
                label = int(category)
            except ValueError:
                print(f"⚠️ Warning: Folder name '{category} is not numeric . Skipping...'")

            # The annotation file in the folder
            bb_info_file = os.path.join(category_path, "bb_info.txt")
            if not os.path.exists(bb_info_file):
                print(f"⚠️ Warning: {bb_info_file} not found. Skipping folder {category}...")
                continue

            # Read bounding box and group by image id
            image_to_bboxes = defaultdict(list)
            with open(bb_info_file, "r") as f:
                lines = f.readlines()
                # Skip header (e.g, "img, x1, y1, x2, y2")
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts)<5:
                        continue
                    img_id = parts[0]
                    # Parse bounding box coordinates (in Pascal VOC format)
                    x1, y1, x2, y2 = map(int, parts[1:5])
                    image_to_bboxes[img_id].append([x1, y1, x2, y2])
            # Create an entry for each image
            for img_id, bboxes in image_to_bboxes.items():
                # Construct image path
                image_path = os.path.join(category_path, img_id + self.image_ext)
                if not os.path.exists(image_path):
                    image_path = os.path.join(category_path, img_id + self.image_ext.lower())
                if not os.path.exists(image_path):
                    print(f"⚠️ Warning: Image {image_path} not found. Skipping...")
                    continue
                self.data.append({"image_path":image_path,
                                  "bboxes":bboxes,
                                  "label":label})
    def len(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        # Load the image using OpenCV and convert BGR to RGB
        image = cv2.imread(item["image_path"])
        if image is None:
            raise ValueError(f"⚠️ Image not found: {item['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = item["bboxes"]
        labels = [item["label"]]*len(bboxes)

        if self.transform:
            transformed = self.transform(image = image, bboxes=bboxes, labels = labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]
        target = {"boxes":torch.tensor(bboxes, dtype = torch.float32), # Shape: [num_boxes,4]
                  "labels":torch.tensor(labels, dtype = torch.int64)}  # Shape: [num_boxes]
        return image, target

# -------------------------
# Define Augmentations
# -------------------------
# Training augmentations include resizing, cropping, flips, brightness/contrast adjustments,
# and geometric transformations. The bounding boxes are transformed accordingly.

train_transform = A.Compose([A.Resize(height=256, width=256),
                             A.RandomCrop(height=224, width=224),
                             A.HorizontalFlip(p=0.5),
                             A.RandomBrightnessContrast(p=0.2),
                             A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                                rotate_limit=15, p=0.5),
                             A.Normalize(mean=[0.485, 0.456, 0.406],
                                         std =[0.229, 0.224, 0.225]),
                             ToTensorV2()], bbox_params=A.BboxParams(format="pascal_voc",
                                                                     label_fields=["labels"]))

val_transform = A.Compose([A.Resize(height=224, width=224),
                           A.Normalize(mean=[0.485, 0.456, 0.406],
                                       std =[0.229, 0.224, 0.225]),
                           ToTensorV2()], bbox_params=A.BboxParams(format="pascal_voc",
                                                                   label_fields=["labels"]))




