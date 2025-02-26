import asyncio
import os
from typing import Callable, Any, Optional, Iterable, List, Tuple

import aiohttp

from shape_mapping import UEC256ShapeMapper

import cv2
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
from ml_pipeline.utils.transforms import get_train_transforms, get_val_transforms

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
        if self.root_dir is None:
            raise TypeError("The root directory is empty...")
        self.transform = transform or get_train_transforms()
        self.image_ext = image_ext
        self.nutrition_cache: Dict[str, Dict[str, float]] = {}  # {category_name: nutrition_data}
        self.nutrition_mapper = nutrition_mapper
        if self.nutrition_mapper:
            self._load_nutrition_data()
        self._validate_nutrition_data()
        self.shape_mapper = UEC256ShapeMapper()
        self.data = []            # Each dict corresponds to one image
        self.id_to_category = {}  # Mapping numerical id to category name
        self._read_category_file()
        self._load_dataset()
        self.fallback_sample = {"image_path":"placeholder.jpg",
                                "bboxes":[[0,0,100,100]],
                                "label":0,
                                "nutrition":NutritionMapper.get_default_nutrition(),
                                "portions":0.0}
    def _load_nutrition_data(self) -> None:
        """Async nutrition data loading with parallel requests"""
        if not self.nutrition_mapper:
            return
        session = aiohttp.ClientSession()
        self.nutrition_mapper.session = session
        tasks = [self.nutrition_mapper.map_food_label_to_nutrition(cat)
                 for cat in self.id_to_category.values()]
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        # Process results
        for cat, result in zip(self.id_to_category.values(), results):
            if isinstance(result, Exception):
                logging.warning(...)
                self.nutrition_cache[cat] = self.nutrition_mapper.get_default_nutrition()
            else:
                self.nutrition_cache[cat] = result

    def _validate_nutrition_data(self) -> None:
        """Validate cached nutrition data"""
        if not self.nutrition_mapper:
            return
        for cat_id, food_name in self.id_to_category.items():
            try:
                nutrition = self.nutrition_cache.get(food_name)
                if not nutrition:
                    logging.warning(f"Missing nutrition data for {food_name}")
                    continue

                required_keys = ["calories", "protein", "fat", "carbohydrates"]
                missing_keys = [k for k in required_keys if k not in nutrition]
                if missing_keys:
                    logging.warning(f"Missing keys {missing_keys} for {food_name}")
            except Exception as e:
                logging.error(f"Error validating nutrition for {food_name}: {e}")

    def _get_fallback_sample(self):
        """Generate dummy sample for error recovery"""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img = self.transform(image=img)["image"]
        return img, {"boxes":torch.zeros((1,4)),
                     "labels":torch.tensor([0]),
                     "nutrition":torch.zeros(4)}

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
            self.id_to_category = {}

    def _load_dataset(self) -> None:
        total_images = 0
        possible_exts = [self.image_ext, self.image_ext.lower(),
                         self.image_ext.upper()]
        possible_exts = list(dict.fromkeys(possible_exts))
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
                image_path = None
                for ext in possible_exts:
                    candidate = os.path.join(category_path, f"{img_id}{ext}")
                    if os.path.exists(candidate):
                        image_path = candidate
                        break
                if image_path is None:
                    logging.warning(f"Warning: Image {img_id} not found with extensions {possible_exts}. Skipping...")
                    continue
                self.data.append({"image_path":image_path,
                                  "bboxes":bboxes, "label":label})
                total_images += 1
        logging.info(f"Total images loaded: {total_images}")

    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Mask loading verification
        try:
            item:dict = self.data[idx]
            mask_path = Path(item["image_path"]).with_suffix(".png")
            mask = None
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = (mask>127).astype(np.float32) # Convert to binary float mask
                item["mask"] = mask
            # Load the image using OpenCV and convert BGR to RGB
            image = cv2.imread(item["image_path"])
            if image is None:
                raise ValueError(f"⚠️ Image not found: {item['image_path']}")
            if "nutrition" not in item:
                logging.warning(f"Missing nutrition data for {item['label']}")
                item["nutrition"] = self.nutrition_mapper.get_default_nutrition()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = item["bboxes"]
            labels = [item["label"]]*len(bboxes)

            if self.transform:
                if "mask" in item:
                    transformed = self.transform(image=image, bboxes=bboxes, labels=labels, mask=item["mask"])
                    image = transformed["image"]
                    bboxes = transformed["bboxes"] # Transformed PascalVoc
                    labels = transformed["labels"]
                    mask = transformed["mask"]
                else:
                    transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
                    image = transformed["image"]
                    bboxes = transformed["bboxes"]
                    labels = transformed["labels"]
            else:
                mask = item.get("mask")
            # Convert to YOLO format (normalized x,y,w,h)
            valid_bboxes = []
            valid_labels = []
            valid_portions = []
            height, width = image.shape[1], image.shape[2] # Transformed image shape (C,H,W)
            # Nutrition data
            food_name = self.id_to_category.get(item["label"], "unknown")
            nutrition = self.nutrition_cache.get(food_name,
                                                 self.nutrition_mapper.get_default_nutrition())
            nutrition_data = torch.tensor([nutrition.get("calories",0),
                                           nutrition.get("protein",0),
                                           nutrition.get("fat",0),
                                           nutrition.get("carbohydrates",0)],
                                          dtype=torch.float32)
            # Portion calculation after transformation
            portions = []
            for i,(x1, y1, x2, y2) in enumerate(bboxes):
                if x1 >= x2 or y1 >= y2:
                    logging.warning(f"Invalid bounding box ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}) for sample {idx}. Skipping...")
                    continue
                valid_bboxes.append([x1, y1, x2, y2])
                valid_labels.append(labels[i])
                if mask is not None:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    mask_crop = mask[y1:y2, x1:x2]
                    mask_area = np.sum(mask_crop)
                    portion = self._estimate_portion(food_name, mask_area)
                else:
                    bbox_area = (x2-x1)*(y2-y1)
                    portion = self._estimate_portion(food_name, bbox_area)
                    valid_portions.append(portion)
            height, width = None, None
            if isinstance(image, torch.Tensor):
                height, width = image.shape[1], image.shape[2]
            else:
                height, width = image.shape[0], image.shape[1]

            yolo_boxes = []
            for (x1, y1, x2, y2) in bboxes:
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                yolo_boxes.append([x_center, y_center, w, h])

            target = {"bboxes":torch.tensor(yolo_boxes, dtype=torch.float32),
                      "labels":torch.tensor(valid_labels, dtype=torch.int64),
                      "portions":torch.tensor(valid_portions, dtype=torch.float32),
                      "nutritions":nutrition_data}

            return image, target

        except Exception as e:
            logging.error(f"Failed to load sample {idx}: {str(e)}")
            return self._get_fallback_sample()

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
        yolotarget = torch.cat([target["labels"].unsqueeze(1),      # [num_boxes, 1]
                                target["bboxes"],                            # [num_boxes, 4]
                                target["portions"].unsqueeze(1)], dim = 1)  # [num_boxes, 1]
        detection_targets.append(yolotarget)
        nutrition_targets.append(target["nutritions"])
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
