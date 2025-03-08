import json
import asyncio
from functools import lru_cache
from typing import Callable, Any, Optional, Iterable, List, Tuple
import aiohttp
import nest_asyncio
from shape_mapping import UEC256ShapeMapper
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
from ml_pipeline.data_processing.nutrition_mapper import NutritionMapper
from ml_pipeline.utils.transforms import get_train_transforms

from pathlib import Path
import logging

nest_asyncio.apply()

class FoodDataset(Dataset):
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
        self._load_categories()
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

    @staticmethod
    def _get_fallback_sample():
        """Generate dummy sample for error recovery"""
        image = torch.zeros((3, 224, 224), dtype=torch.float32)
        target = {"bboxes": torch.zeros((1, 4), dtype=torch.float32),
                  "labels": torch.zeros((1), dtype=torch.int64),
                  "portions": torch.zeros((1), dtype=torch.float32),
                  "nutritions": torch.zeros((1, 4), dtype=torch.float32)}
        return image, target

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

    def _load_categories(self):
        """Load FoodMask categories"""
        categories_file = self.root_dir / "categories.json"
        if categories_file.exists():
            with open(categories_file, "r") as f:
                categories = json.load(f)
            for cat in categories:
                self.id_to_category[cat["id"]] = cat["name"]
        else:
            logging.warning("Categories file not found")

    def _load_dataset(self):
        """Load FoodMask dataset"""
        annotation_file = self.root_dir / "annotations.json"
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotations file not found at {annotation_file}")

        with open(annotation_file, "r") as f:
            data = json.load(f)

        # Process FoodMask annotations
        image_dict = {img["id"]: img for img in data["images"]}
        annotations_by_image = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_dict:
                continue
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            x, y, w, h = ann["bbox"]
            annotations_by_image[img_id].append({
                "bbox": [x, y, x + w, y + h],
                "label": ann["category_id"],
                "segmentation": ann["segmentation"]})

        self.data = []
        for img_id, img_info in image_dict.items():
            image_path = self.root_dir/"images"/img_info["file_name"]
            objects = annotations_by_image.get(img_id, [])
            bboxes = [obj["bbox"] for obj in objects]
            labels = [obj["label"] for obj in objects]
            self.data.append({
                "image_path": str(image_path),
                "bbox": bboxes,
                "label": labels,
                "masks":[obj["segmentation"] for obj in objects]
            })

    def _process_mask(self, mask_path):
        """Process mask from FoodMask format"""
        if not mask_path:
            return None

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.warning(f"Failed to load mask at {mask_path}")
            return None

        mask = (mask > self.mask_threshold).astype(np.float32)
        return mask


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        try:
            item = self.data[idx]
            image_path = item["image_path"]
            bboxes = item["bboxes"]
            labels = item["labels"]

            image = self._load_image(image_path)
            height, width = image.shape[:2]

            if not bboxes:  # Handle empty images
                return self._get_fallback_sample()

            yolo_boxes = []
            portions = []
            nutritions = []
            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = bbox
                if x1 >= x2 or y1 >= y2 or x2 > width or y2 > height:
                    continue  # Skip invalid boxes
                center_x = (x1 + x2) / 2 / width
                center_y = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                yolo_box = [center_x, center_y, bbox_width, bbox_height]

                food_name = self.id_to_category.get(label, f"unknown_{label}")
                area = (x2 - x1) * (y2 - y1)  # Replace with mask area if available
                portion = self._estimate_portion(food_name, area)

                nutrition_data = [0, 0, 0, 0]
                if self.nutrition_mapper:
                    nutrition_dict = self.nutrition_mapper.get_cached_nutrition(food_name)
                    nutrition_data = [nutrition_dict.get(k, 0) for k in ["calories", "protein", "fat", "carbohydrates"]]

                yolo_boxes.append(yolo_box)
                portions.append(portion)
                nutritions.append(nutrition_data)

            if not yolo_boxes:
                return self._get_fallback_sample()

            if self.transform:
                transformed = self.transform(image=image, bboxes=yolo_boxes, labels=labels)
                image = transformed["image"]
                yolo_boxes = transformed["bboxes"]
                labels = transformed["labels"]
                if not isinstance(image, torch.Tensor):
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

            target = {
                "bboxes": torch.tensor(yolo_boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "portions": torch.tensor(portions, dtype=torch.float32),
                "nutritions": torch.tensor(nutritions, dtype=torch.float32)
            }
            return image, target
        except Exception as e:
            logging.error(f"Failed to load sample {idx}: {str(e)}", exc_info=True)
            return self._get_fallback_sample()

    @lru_cache(maxsize=128)
    def _load_image(self, image_path):
        """Load image with caching for better performance"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Image not found: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    @staticmethod
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

