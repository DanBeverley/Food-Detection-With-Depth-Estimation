import os
from typing import Callable, Any, Optional, Iterable, List, Tuple
from shape_mapping import UEC256ShapeMapper

import cv2
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
from ml_pipeline.utils.transforms import get_train_transforms, get_val_transforms
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
import logging

class UECFoodDataset(Dataset):
    def __init__(self, root_dir:Optional[str]=None,
                 transform:Optional[Callable]=None, image_ext:str=".jpg",
                 nutrition_mapper:Optional[NutritionMapper]=None) -> None:
        """
        Args:
            root_dir (str): Root directory containing subfolders for each food category.
            transform (callable, optional): Albumentations transformation pipeline.
            image_ext (str): Extension of the image files (default '.jpg').
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or get_train_transforms()
        self.image_ext = image_ext
        self.nutrition_mapper = nutrition_mapper
        self.shape_mapper = UEC256ShapeMapper()
        self.data = []            # Each dict corresponds to one image
        self.id_to_category = {}  # Mapping numerical id to category name
        self._read_category_file()
        self._load_dataset()
        self._validate_nutrition_data()
        self.nutrition_cache: Dict[str, Dict[str, float]] = {}  # {category_name: nutrition_data}

    async def initialize(self) -> None:
        """Async initialization hook"""
        if self.nutrition_mapper:
            await self._load_nutrition_data()
            self._validate_nutrition_data()

    async def _load_nutrition_data(self) -> None:
        """Async nutrition data loading with parallel requests"""
        if not self.nutrition_mapper:
            return
        # Create tasks for all categories
        tasks = []
        category_names = list(self.id_to_category.values())
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.nutrition_mapper.map_food_label_to_nutrition, cat_name)
                       for cat_name in category_names]
        # Process results
        for cat_name, future in zip(category_names, futures):
            try:
                result = future.result()
                self.nutrition_cache[cat_name] = result
            except Exception as e:
                logging.warning(f"Failed to load nutrition for {cat_name}: {str(e)}")
                self.nutrition_cache[cat_name] = self.nutrition_mapper.get_default_nutrition()

    def _validate_nutrition_data(self) -> None:
        """Validate cached nutrition data"""
        if not self.nutrition_mapper:
            return
        for cat_id, food_name in self.id_to_category.items():
            try:
                nutrition = self.nutrition_mapper.get_nutrition_data(food_name)
                if not nutrition:
                    logging.warning(f"Missing nutrition data for {food_name}")
                    continue

                required_keys = ["calories", "protein", "fat", "carbohydrates"]
                missing_keys = [k for k in required_keys if k not in nutrition]
                if missing_keys:
                    logging.warning(f"Missing keys {missing_keys} for {food_name}")
            except Exception as e:
                logging.error(f"Error validating nutrition for {food_name}: {e}")

    def process_batch(self, indices:Iterable[int]) -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        return [self.__getitem__(i) for i in indices]

    def _read_category_file(self) -> None:
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
            logging.warning(f"⚠️ Warning: category.txt not found in the root directory.")
            self.id_to_category = None

    def _load_dataset(self) -> None:
        for category in sorted(os.listdir(self.root_dir)):
            category_path = os.path.join(self.root_dir, category)
            if not os.path.isdir(category_path):
                continue
            try:
                label = int(category)
            except ValueError:
                logging.warning(f"⚠️ Warning: Folder name '{category} is not numeric . Skipping...'")
                continue
            # The annotation file in the folder
            bb_info_file = os.path.join(category_path, "bb_info.txt")
            if not os.path.exists(bb_info_file):
                logging.warning(f"⚠️ Warning: {bb_info_file} not found. Skipping folder {category}...")
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

    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        item:dict = self.data[idx]

        # Mask loading verification
        mask_path = Path(item["image_path"]).with_suffix(".png")
        mask = None
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask>127).astype(np.float32) # Convert to binary float mask
            item["mask"] = mask

        if mask is not None and not isinstance(mask, np.ndarray):
            raise ValueError(f"Invalid mask type: {type(mask)}")

        # Load the image using OpenCV and convert BGR to RGB
        image = cv2.imread(item["image_path"])
        if image is None:
            raise ValueError(f"⚠️ Image not found: {item['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = item["bboxes"]
        labels = [item["label"]]*len(bboxes)

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"] # Transformed PascalVoc
            labels = transformed["labels"]

        # Convert to YOLO format (normalized x,y,w,h)
        height, width = image.shape[1], image.shape[2] # Transformed image shape (C,H,W)
        yolo_boxes = []
        for (x1, y1, x2, y2) in bboxes:
            x_center = ((x1+x2)/2)/width
            y_center = ((y1+y2)/2)/height
            w = (x2-x1)/width
            h = (y2-y1)/height
            yolo_boxes.append([x_center, y_center, w, h])

        target = {"boxes":torch.tensor(yolo_boxes, dtype = torch.float32), # Shape: [num_boxes,4]
                  "labels":torch.tensor(labels, dtype = torch.int64),      # Shape: [num_boxes]
                  }

        # Portion estimation calculations
        nutrition_data = torch.zeros(4, dtype=torch.float32) # [cal, prot, fat, carbs]

        if self.nutrition_mapper:
            food_name = self.id_to_category.get(item["label"], "unknown")
            try:
                # nutrition = self.nutrition_mapper.get_nutrition_data(food_name)
                nutrition = self.nutrition_cache.get(food_name,
                                                     self.nutrition_mapper.get_default_nutrition())
            except(KeyError, ConnectionError) as e:
                logging.warning(f"Nutrition data unavailable for {food_name}: {e}")
                nutrition = self.nutrition_mapper.get_default_nutrition()
            nutrition_data = torch.tensor([nutrition.get("calories",0),
                                           nutrition.get("protein",0),
                                           nutrition.get("fat",0),
                                           nutrition.get("carbohydrates",0)],
                                          dtype=torch.float32)
            if 'mask' in item:
                mask_area = np.sum(item["mask"])
                portion = self._estimate_portion(food_name, mask_area)
            else:   # Fallback to bbox area
                # Calculate area based portion estimation
                x1, y1, x2, y2 = bboxes[0]
                bbox_area = (x2-x1)*(y2-y1)
                portion =  self._estimate_portion(food_name, bbox_area)

                target.update({# "boxes":torch.tensor(yolo_boxes, dtype=torch.float32),
                              # "labels":torch.tensor(labels, dtype=torch.int64),
                              "portions":torch.tensor([portion], dtype=torch.float32),
                              "nutrition":nutrition_data})
        if "nutrition" not in target:
            target["nutrition"] = torch.zeros(4, dtype=torch.float32)
        if "portions" not in target:
            target["portions"] = torch.zeros(1, dtype=torch.float32)
        if "mask" in item:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels,
                                         mask=item["mask"])
            mask = transformed["mask"]
        else:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        target["mask"] = mask if mask is not None else torch.zeros(1)

        return image, target


    def _estimate_portion(self, food_name:str, bbox_area:float) -> float:
        """Estimate portion (volume in ml) using food-specific density per pixel area"""
        # density = self.nutrition_mapper.get_density(food_name) # g/pixel_area
        shape_prior = self.shape_mapper.get_shape_prior(food_name)
        # Example : Dome shape food volume = area^(3/2) * height_ratio
        volume = (bbox_area**1.5)*shape_prior.height_ratio
        return volume*shape_prior.volume_modifier

# Custom Collate Function
def collate_fn(batch:Iterable[Tuple[torch.Tensor,
               Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    image = []
    detection_targets = []
    nutrition_targets = []
    for img, target in batch:
        image.append(img)
        # Combine labels and boxes into [class_id, x, y, w, h]
        yolotarget = torch.cat([target["labels"].unsqueeze(1),
                                target["boxes"],
                                target.get("portion", torch.zeros(1).unsqueeze(1))], dim = 1)
        detection_targets.append(yolotarget)
        nutrition_targets.append(target["nutrition"])
    image = torch.stack(image, dim = 0)
    return (image, {"detection":detection_targets,
                     "nutrition":torch.stack(nutrition_targets,0)})


# Creating DataLoaders
dataset_root = ""
train_dataset = UECFoodDataset(root_dir = dataset_root, transform = get_train_transforms())
val_dataset = UECFoodDataset(root_dir = dataset_root, transform = get_val_transforms())
train_loader = DataLoader(train_dataset, batch_size = 32,
                           shuffle = True, num_workers = 4,
                           pin_memory=True, collate_fn=collate_fn, persistent_workers=True)

val_loader = DataLoader(val_dataset, batch_size=32,
                        shuffle = False, num_workers = 4,
                        pin_memory = True, collate_fn = collate_fn, persistent_workers=True)

# Quick test: Retrieve one batch from the training loader.
images, targets = next(iter(train_loader))
print("Batch Images Shape:", images.shape)
print("Example Target:", targets[0])

# Optionally, print the category mapping (id-to-name) if available.
if train_dataset.id_to_category is not None:
    print("Category Mapping:", train_dataset.id_to_category)
