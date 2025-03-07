import json
import asyncio
import os
from functools import lru_cache
from typing import Callable, Any, Optional, Iterable, List, Tuple
import aiohttp
import nest_asyncio
from shape_mapping import UEC256ShapeMapper
import cv2
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from typing import Dict
from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
from ml_pipeline.utils.transforms import get_train_transforms

from pathlib import Path
import logging

nest_asyncio.apply()

class UECFoodDataset(Dataset):
    def __init__(self, root_dir:Optional[str]=None,
                 transform:Optional[Callable]=None, image_ext:str=".jpg",
                 nutrition_mapper:Optional[NutritionMapper]=None, mask_threshold:int = 127) -> None:
        #TODO: adjust parameters before cloud GPU usage
        """
        Args:
            root_dir (str): Root directory containing subfolders for each food category.
            transform (callable, optional): Albumentations transformation pipeline.
            image_ext (str): Extension of the image files (default '.jpg').
            nutrition_mapper (NutritionMapper, optional) : Mapper for nutrition data
            mask_threshold (int): Threshold for binarizing masks (127 by default)
        """

        if root_dir is None:
            raise ValueError("root_dir must be provided")
        self.root_dir = Path(root_dir)
        self.transform = transform or get_train_transforms()
        self.image_ext = image_ext
        self.nutrition_mapper = nutrition_mapper
        self.mask_threshold = mask_threshold
        self.nutrition_cache: Dict[str, Dict[str, float]] = {}
        self.shape_mapper = UEC256ShapeMapper()
        self.data: List[Dict] = []  # For each dict correspond to one image
        self.id_to_category:Dict[int, str] = {} # Mapping numerical id to category name
        # Load category mapping
        self._read_category_file()
        # Pre-populate nutrition cache with defaults if mapper is provided
        if self.nutrition_mapper:
            for cat_name in self.id_to_category.values():
                self.nutrition_cache[cat_name] = self.nutrition_mapper.get_default_nutrition()
            asyncio.run(self._load_nutrition_data_async())
            self._validate_nutrition_data()
        self._load_dataset()

    async def _load_nutrition_data_async(self) -> None:
        """Async nutrition data loading with parallel requests"""
        if not self.nutrition_mapper:
            return
        async with aiohttp.ClientSession() as session:
            self.nutrition_mapper.session = session
            tasks = [self.nutrition_mapper.map_food_label_to_nutrition(cat) for cat in self.id_to_category.values()]
            results = await asyncio.gather(*tasks, return_exceptions = True)
            for cat, result in zip(self.id_to_category.values(), results):
                if isinstance(result, Exception):
                    logging.warning(f"Error mapping nutrition for {cat}: {result}")
                else:
                    self.nutrition_cache[cat] = result

    def _validate_nutrition_data(self) -> None:
        """Validate cached nutrition data"""
        if not self.nutrition_mapper:
            return
        required_keys = ["calories", "protein", "fat", "carbohydrates"]
        for food_name in self.id_to_category.values():
            nutrition = self.nutrition_cache.get(food_name, {})
            missing_keys = [k for k in required_keys if k not in nutrition]
            if missing_keys:
                logging.warning(f"Missing keys ({missing_keys}) for ({food_name})")

    def _get_fallback_sample(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate dummy sample for error recovery"""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, {"boxes":torch.zeros((1,4), dtype = torch.float32),
                     "labels":torch.tensor([0], dtype = torch.int64),
                     "portions": torch.tensor([0.0], dtype = torch.float32),
                     "nutrition":torch.tensor([0.0, 0.0, 0.0, 0.0], dtype = torch.float32)}

    def _read_category_file(self) -> None:
        categories_file = self.root_dir / "category.txt"
        if categories_file.exists():
            with open(categories_file, "r") as f:
                lines = f.readlines()[1:]
            # Skip the header line
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts)<2:
                    continue
                try:
                    cat_id = int(parts[0])
                    cat_name = ' '.join(parts[1:])
                    self.id_to_category[cat_id] = cat_name
                except ValueError:
                    continue
        else:
            logging.warning(f"⚠️ Warning: category.txt not found in the root directory.")

    def _load_dataset(self) -> None:
        total_images = 0
        possible_exts = [self.image_ext, self.image_ext.lower(), self.image_ext.upper()]
        possible_exts = list(dict.fromkeys(possible_exts))
        for category in sorted(os.listdir(self.root_dir)):
            category_path = self.root_dir / category
            if not category_path.is_dir():
                continue
            try:
                label = int(category)
            except ValueError:
                logging.warning(f"⚠️ Warning: Folder name '{category} is not numeric . Skipping...'")
                continue
            # The annotation file in the folder
            bb_info_file = category_path / "bb_info.txt"
            if not bb_info_file.exists():
                logging.warning(f"⚠️ Warning: {bb_info_file} not found. Skipping folder {category}...")
                continue

            # Read bounding box and group by image id
            image_to_bboxes = defaultdict(list)
            with open(bb_info_file, "r") as f:
                lines = f.readlines()[1:]
                # Skip header (e.g, "img, x1, y1, x2, y2")
                for line in lines:
                    parts = line.strip().split()
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
                    candidate = category_path / f"{img_id}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                if image_path is None:
                    logging.warning(f"Warning: Image {img_id} not found with extensions {possible_exts}. Skipping...")
                    continue
                self.data.append({"image_path":str(image_path), "bboxes":bboxes, "label":label})
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
                if mask is None:
                    logging.warning(f"Failed to load mask at {mask_path}")
                else:
                    mask = (mask > self.mask_threshold).astype(np.float32)
            # Load the image using OpenCV and convert BGR to RGB
            image = cv2.imread(item["image_path"])
            if image is None:
                raise ValueError(f"⚠️ Image not found: {item['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = item["bboxes"]
            labels = [item["label"]]*len(bboxes)

            if self.transform:
                if mask is not None:
                    transformed = self.transform(image=image, bboxes=bboxes, labels=labels, mask=mask)
                    mask = transformed["mask"]
                else:
                    transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
                image = transformed["image"]
                bboxes = transformed["bboxes"]
                labels = transformed["labels"]
            # Process bounding boxes and portions
            height, width = image.shape[1], image.shape[2] if isinstance(image, torch.Tensor) else image.shape[:2]
            food_name = self.id_to_category.get(item["label"], "unknown")
            nutrition = self.nutrition_cache.get(food_name, self.nutrition_mapper.get_default_nutrition())
            nutrition_data = torch.tensor([nutrition.get(k, 0.0) for k in ["calories", "protein", "fat", "carbohydrates"]], dtype=torch.float32)
            # Convert to YOLO format (normalized x,y,w,h)
            valid_bboxes, valid_labels, valid_portions = [], [], []
            invalid_boxes = []
            for i,(x1, y1, x2, y2) in enumerate(bboxes):
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    invalid_boxes.append((x1, y1, x2, y2))
                    continue
                valid_bboxes.append([x1, y1, x2, y2])
                valid_labels.append(labels[i])
                if mask is not None:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    x1, y2 = max(0, x1), min(width, x2)
                    y1, y2 = max(0, y1), min(height, y2)
                    if x1 < x2 and y1 < y2:
                        area = mask[y1:y2, x1:x2].sum()
                    else:
                        area = (x2 - x1) * (y2 - y1)
                else:
                    area = (x2 - x1) * (y2 - y1)
                portion = self._estimate_portion(food_name, area)
                valid_portions.append(portion)

            if invalid_boxes:
                logging.warning(f"Sample {idx}: Invalid boxes {invalid_boxes}")
            if not valid_bboxes:
                logging.warning(f"No valid boxes for sample {idx}, using fallback")
                return self._get_fallback_sample()
            yolo_boxes = [
                [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height, (x2 - x1) / width, (y2 - y1) / height]
                for x1, y1, x2, y2 in valid_bboxes
            ]

            target = {"bboxes":torch.tensor(yolo_boxes, dtype=torch.float32),
                      "labels":torch.tensor(valid_labels, dtype=torch.int64),
                      "portions":torch.tensor(valid_portions, dtype=torch.float32),
                      "nutritions":nutrition_data}

            return image, target

        except Exception as e:
            logging.error(f"Failed to load sample {idx}: {str(e)}", exc_info = True)
            return self._get_fallback_sample()

    def _estimate_portion(self, food_name:str, bbox_area:float) -> float:
        """Estimate portion (volume in ml) using food-specific density per pixel area"""
        shape_prior = self.shape_mapper.get_shape_prior(food_name)
        if not isinstance(shape_prior, dict):
            logging.error(f"Invalid shape_prior type for {food_name}: {type(shape_prior)}")
            return bbox_area * 2.5 # fallback
        shape_type = shape_prior.get("type", "domed")
        height = shape_prior.get("height", 2.5)
        volume_modifier = shape_prior.get("volume_modifier", 0.85)

        if shape_type == "cylindrical":
            return bbox_area* height* volume_modifier
        elif shape_type == "spherical":
            radius = np.sqrt(bbox_area/np.pi)
            return (4/3) * np.pi * radius ** 3 * volume_modifier
        elif shape_type == "domed":
            return (bbox_area ** 1.5) * height * volume_modifier
        else:
            logging.warning(f"Unknown shape type: {shape_type}")
            return bbox_area*2.5 # Fallback

# Custom Collate Function
def collate_fn(batch:Iterable[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    images, detection_targets, nutrition_targets = [], [], []
    for img, target in batch:
        images.append(img)
        # Combine labels and boxes into [class_id, x, y, w, h]
        yolo_target = torch.cat([target["labels"].unsqueeze(1),      # [num_boxes, 1]
                                target["bboxes"],                            # [num_boxes, 4]
                                target["portions"].unsqueeze(1)], dim = 1)  # [num_boxes, 1]
        detection_targets.append(yolo_target)
        nutrition_targets.append(target["nutritions"])
    return torch.stack(images, dim = 0), {"detection": detection_targets,
                                          "nutrition": torch.stack(nutrition_targets, dim = 0)}


class FoodMaskDataset(Dataset):
    """Dataset class for FoodMask format"""
    def __init__(self, root_dir:str, transform:Optional[Callable] = None, nutrition_mapper:NutritionMapper = None,  mask_threshold:int = 127) -> None:
        """
        Args:
            root_dir (str): Root directory containing images and masks.
            transform (callable, optional): Albumentations transformation pipeline.
            nutrition_mapper (NutritionMapper, optional) : Mapper for nutrition data
            mask_threshold (int): Threshold for binarizing masks (127 by default)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or get_train_transforms()
        self.nutrition_mapper = nutrition_mapper
        self.mask_threshold = mask_threshold
        self.nutrition_cache: Dict[str, Dict[str, float]] = {}
        self.data: List[Dict] = []  # For each dict correspond to one image
        self.id_to_category = {} # Mapping numerical id to category name
        # Load category mapping
        self._load_categories()
        if self.nutrition_mapper:
            UECFoodDataset._load_nutrition_data_async()
        self._load_dataset()

    def _load_categories(self):
        categories_file = self.root_dir/"categories.json"
        if categories_file.exists():
            with open(categories_file, "r") as f:
                categories = json.load(f)
            for cat in categories:
                self.id_to_category[cat["id"]] = cat["name"]
        else:
            logging.warning("Categories file not found")

    def  _load_dataset(self):
        annotation_file = self.root_dir/"annotations.json"
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotations file not found at {annotation_file}")
        with open(annotation_file, "r") as f:
            data = json.load(f)
        # Process FoodMask annotations
        image_dict = {img["id"]: img for img in data["images"]}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_dict:
                continue
            image_info  = image_dict[img_id]
            image_path = self.root_dir/"images"/image_info["file_name"]
            mask_path = self.root_dir/"mask"/image_info["file_name"].replace(".jpg", ".png")
            # Extract bbox
            x, y, w, h = ann["bbox"]
            bbox = [x,y, x+w, y+h]
            self.data.append({
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "bbox": bbox,
                "label": ann["category_id"]
            })
    def _process_mask(self, mask_path:str = None) -> Optional[np.ndarray]:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logging.warning(f"Failed to load mask at {mask_path}")
                return None
            mask = (mask > self.mask_threshold).astype(np.float32)
            return mask
    @lru_cache(maxsize=128)
    def _load_image(self, image_path:str = None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
