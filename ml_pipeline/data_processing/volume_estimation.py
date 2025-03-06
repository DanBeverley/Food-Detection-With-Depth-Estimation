import logging
from pathlib import Path
import hashlib
import cv2
import numpy as np
import torch
from scipy.spatial import ConvexHull
from shape_mapping import ShapePrior
from cachetools import LRUCache
from typing import List, Optional, Dict
from ml_pipeline.data_processing.shape_mapping import UEC256ShapeMapper

class HybridPortionEstimator:
    def __init__(self, device=None, nutrition_mapper=None, fallback_scale:float= 22/800):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nutrition_mapper = nutrition_mapper
        self.fallback_scale = fallback_scale
        self.depth_cache = LRUCache(maxsize=100) # For a larger, more efficient caching
        self.models_loaded = False
        # Food type to density mapping (g/cm³)
        self.density_db = {
            'rice': 0.9,  # Cooked rice
            'bread': 0.25,  # Sliced bread
            'meat': 1.1,  # Cooked beef
            'vegetables': 0.6
        }
        self._load_models()

    def _load_models(self):
        """Load both reference object detector and depth estimation model"""
        if self.models_loaded:
            return
        try:
            # Load reference object detector (YOLOv8)
            yolov8_path = Path("models/yolov8n.pt")
            if yolov8_path.exists() and yolov8_path.is_file():
                self.ref_detector = torch.hub.load('ultralytics/yolov8', 'custom', path=str(yolov8_path))
            else:
                self.ref_detector = torch.hub.load("ultralytics/yolov8", 'yolov8n', pretrained=True)
            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to(self.device)
            self.midas_transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
            self.models_loaded = True
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            logging.warning("Using fallback mode without pre-trained models")
            self.ref_detector = None  # Minimal fallback; adjust as needed
            self.midas = None
            self.models_loaded = False

    def get_scale(self, image):
        """
        Compute the pixel-to-centimeter scale factor for an image.
        This uses YOLOv8 to detect reference objects and then calculates
        the scale based on a known width (e.g., a credit card is ~8.5 cm).
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ref_objects = self._detect_reference_objects(img_rgb)
        scale_ratio = self._calculate_scale(ref_objects, known_width_cm=8.5)
        return scale_ratio

    @staticmethod
    def _get_region_depth(depth_map, mask, box=None):
        """Get average depth in masked region"""
        if mask is not None:
            valid_depths = depth_map[mask]
        else:  # Fallback to bounding box
            x1, y1, x2, y2 = map(int, box)
            valid_depths = depth_map[y1:y2,x1:x2].flatten()
        return np.mean(valid_depths) if valid_depths.size > 0 else .1

    @staticmethod
    def _scale_depth_map(depth_map, ref_objects, known_width_cm):
        if not ref_objects:
            return depth_map
        best_ref = max(ref_objects, key=lambda x: x["confidence"])
        ref_width_pixels = best_ref["box"][2] - best_ref["box"][0]
        scale_ratio = known_width_cm/ref_width_pixels
        return depth_map*scale_ratio

    def _get_reference_scale(self, image:np.ndarray):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ref_objects = self._detect_reference_objects(img_rgb)
        return self._calculate_scale(ref_objects, known_width_cm = 8.5)

    def _get_scaled_depth_map(self, image, scale_ratio):
        depth_map = self._get_depth_map(image)
        return depth_map * scale_ratio

    def estimate_portion(self, image: np.ndarray, food_boxes: List[List[float]],
                        food_labels: List[str], masks: Optional[List[np.ndarray]] = None) -> List[Dict]:
        scale_ratio = self._get_reference_scale(image)
        depth_map = self._get_scaled_depth_map(image, scale_ratio)
        portions = []
        for idx, (box, label) in enumerate(zip(food_boxes, food_labels)):
            if masks and idx < len(masks):
                area_cm2 = np.sum(masks[idx]) * (scale_ratio ** 2)
                depth_cm = self._get_region_depth(depth_map, masks[idx])
            else:
                w, h = box[2] - box[0], box[3] - box[1]
                area_cm2 = (w*h) * (scale_ratio ** 2)
                depth_cm = self._get_region_depth(depth_map, masks[idx])
            density = self._get_food_density(label)
            volume_cm3 = area_cm2 * depth_cm
            weight_g = volume_cm3 * density
            calories = self._estimate_calories(label, weight_g)
            portions.append({"weight":weight_g, "calories": calories, "volume": volume_cm3, "bounding_box": box})
        return portions

    def _detect_reference_objects(self, image):
        """Detect known reference objects using YOLOv8"""
        results = self.ref_detector(image)
        reference_classes = ["plate", "credit card", "bowl"]
        ref_objects = []
        for box in results.boxes:
            cls = int(box.cls)
            label = self.ref_detector.names[cls]
            if label in reference_classes:
                ref_objects.append({
                    'box': box.xyxy[0].tolist(),
                    'label': label,
                    'confidence': float(box.conf)
                })
        return ref_objects

    def _calculate_scale(self, ref_objects, known_width_cm):
        """calculate scale using best reference object"""
        if not ref_objects:
            return self.fallback_scale # Fallback default

        # Select most confident reference object
        best_ref = max(ref_objects, key=lambda x:x["confidence"]*(x["box"][2] - x["box"][0]))
        ref_width = best_ref["box"][2] - best_ref["box"][0]

        # Known physical dimensions for common references
        reference_sizes = {
            'credit card': 8.56,  # cm
            'plate': 22,  # cm (standard dinner plate)
            'bowl': 15,  # cm
            'chopsticks': 24,  # cm (average length)
            'fork': 19,  # cm (standard dinner fork)
            'spoon': 17,  # cm (standard tablespoon)
            'knife': 21,  # cm (dinner knife)
            'smartphone': 14.7,  # cm (average smartphone length)
            'tennis ball': 6.7,  # cm (diameter)
            'soda can': 12.2,  # cm (height, 330ml)
            'water bottle': 23,  # cm (500ml standard bottle)
            'apple': 8,  # cm (diameter, medium apple)
            'banana': 18,  # cm (average banana length)
            'orange': 7.5,  # cm (diameter, medium orange)
            'egg': 6,  # cm (height, large egg)
            'slice of bread': 12,  # cm (width of standard loaf slice)
            'burger': 10,  # cm (diameter of typical fast-food burger)
            'pizza slice': 12,  # cm (length from crust to tip)
            'pizza (whole)': 30,  # cm (large pizza diameter)
            'cupcake': 6.5,  # cm (diameter of standard cupcake)
            'mug': 9.5,  # cm (height of a coffee mug)
            'cereal box': 30,  # cm (height of standard box)
            'milk carton': 20,  # cm (height of 1-liter carton)
            'carrot': 15,  # cm (average carrot length)
            'strawberry': 4,  # cm (diameter of medium strawberry)
            'grape': 2,  # cm (diameter of medium grape)
            'spaghetti (uncooked)': 25,  # cm (length of standard spaghetti)
        }
        physical_size = reference_sizes.get(best_ref["label"], known_width_cm)
        return physical_size/ref_width

    def _get_depth_map(self, image:np.ndarray) -> np.ndarray:
        """Get depth map using MiDaS with caching for more performance"""
        img_hash = hashlib.sha256(image.tobytes()).hexdigest()
        # Check if having a cached result
        if img_hash in self.depth_cache:
            return self.depth_cache[img_hash]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.midas_transform(img_rgb).to(self.device)
        with torch.no_grad():
            depth = self.midas(img_tensor).squeeze().cpu().numpy()
        self.depth_cache[img_hash] = depth
        return depth

    @staticmethod
    def _calculate_volume_confidence(depth_quality:float, mask_quality:float,
                                     reference_confidence:float)->float:
        """Calculate overall confidence score for volume estimation"""
        weights = {"depth":0.4, "mask":0.4, "reference":0.2}
        confidence = (depth_quality * weights["depth"] +
                      mask_quality * weights["mask"] +
                      reference_confidence * weights["reference"])
        return min(max(confidence, 0.0), 1.0)

    def _get_food_density(self, label):
        """Get density from database or nutrition mapper"""
        if self.nutrition_mapper:
            nutrition = self.nutrition_mapper.map_food_label_to_nutrition(label)
            if nutrition and "density" in nutrition:
                return nutrition["density"]
        return self.density_db.get(label.lower(), 0.8) # Default

    def _estimate_calories(self, label, weight_g):
        # Estimate calories with nutrition mapper
        if self.nutrition_mapper:
            nutrition = self.nutrition_mapper.map_food_label_to_nutrition(label)
            if nutrition and nutrition["calories_per_100g"]:
                return (weight_g/100)*nutrition["calories_per_100g"]
        return weight_g*1.5 # Fallback 1.5 cal/g

class UECVolumeEstimator:
    def __init__(self, food_shape_priors = None, priors_file = None, confidence_weights = None):
        self.shape_mapper = UEC256ShapeMapper()
        self.food_shape_priors = self.shape_mapper.category_map
        self.confidence_weights = confidence_weights or {"mask":0.4, "depth":0.4, "shape":0.2}
        # self.food_shape_priors = food_shape_priors or self._get_default_shape_prior()

    def load_shape_priors(self, file_path:str):
        import json
        with open(file_path, "r") as f:
            self.food_shape_priors.update(json.load(f))

    @staticmethod
    def _get_default_shape_prior():
        """Define shape priors for common UEC-256 food categories
        Based on typical geometrical shapes of Japanese foods"""
        return {
            'rice': {
                'shape': 'dome',
                'height_ratio': 0.5,  # height to width ratio
                'volume_modifier': 0.7  # accounts for air gaps
            },
            'ramen': {
                'shape': 'bowl_content',
                'liquid_ratio': 0.7,  # ratio of bowl typically filled
                'solid_ratio': 0.3  # ratio of solid ingredients
            },
            'sushi': {
                'shape': 'cylinder',
                'height_ratio': 0.4
            },
            'tempura': {
                'shape': 'irregular',
                'volume_modifier': 0.8
            },
            'default': {
                'shape': 'irregular',
                'volume_modifier': 0.85
            }
        }
    @staticmethod
    def _calculate_base_area(mask, reference_scale):
        area_pixels = np.sum(mask)
        return area_pixels * (reference_scale ** 2)

    def estimate_volume(self, depth_map, mask, food_class, reference_scale):
        """
        Estimate food volume using depth map, segmentation mask, and food-specific priors.

        Args:
            depth_map: 2D numpy array of depth values
            mask: Binary segmentation mask for the food item
            food_class: String indicating UEC food class
            reference_scale: Pixels to real-world units conversion factor

        Returns:
            volume_cm3: Estimated volume in cubic centimeters
            confidence: Confidence score of the estimation
        """
        # Shape priors
        # shape_prior = self.food_shape_priors.get(food_class, self.food_shape_priors["default"])
        shape_prior = self.shape_mapper.get_shape_prior(food_class)
        # Calculate base metrics
        area_cm2 = self._calculate_base_area(mask, reference_scale)

        if shape_prior.shape== 'dome':
            volume = self._estimate_dome_volume(depth_map, mask, area_cm2, shape_prior)
        elif shape_prior.shape == 'bowl_content':
            volume = self._estimate_bowl_content_volume(depth_map, mask, area_cm2, shape_prior)
        elif shape_prior.shape == 'cylinder':
            volume = self._estimate_cylinder_volume(depth_map, mask, area_cm2, shape_prior)
        else:  # irregular shape
            volume = self._estimate_irregular_volume(depth_map, mask, area_cm2, shape_prior, reference_scale)
        # Calculate confidence based on factors
        confidence = self._calculate_confidence(mask, depth_map, shape_prior)
        return volume, confidence
    @staticmethod
    def _estimate_dome_volume(depth_map, mask, area_cm2, prior):
        """
        Estimate volume for dome-shaped foods (e.g., rice portions)
        """
        masked_depth = depth_map*mask
        height = np.max(masked_depth[mask > 0]) if np.any(mask) else 0.1
        radius = np.sqrt(area_cm2 / np.pi)
        volume = (1 / 3) * np.pi * height * (3 * radius ** 2 + height ** 2)
        return volume*prior.volume_modifier

    @staticmethod
    def _estimate_bowl_content_volume(depth_map, mask, area_cm2, prior):
        """Estimate volume for food served in bowls"""
        masked_depth = depth_map*mask
        mean_depth = np.mean(masked_depth[mask>0]) if np.sum(mask)>0 else 0.1
        # Default fallback if missing
        liquid_ratio = prior.liquid_ratio or 0.5
        solid_ratio = prior.solid_ratio or 0.5
        # Calculate bowl volume and apply content ratio
        bowl_volume = area_cm2*mean_depth
        liquid_volume = bowl_volume*liquid_ratio
        solid_volume = bowl_volume*solid_ratio

        return (liquid_volume + solid_volume)*prior.volume_modifier

    def _estimate_cylinder_volume(self, depth_map, mask, area_cm2, prior):
        """Estimate volume for cylindrical foods e.g. sushi roll"""
        # Use contour analysis to get the length
        contours = self._get_contours(mask)
        if contours.size>0:
            x_min, y_min = np.min(contours, axis = 0)
            x_max, y_max = np.max(contours, axis = 0)
            length = max(x_max - x_min, y_max - y_min)
        else:
            length = np.sqrt(area_cm2)
        # Avoid division by zero
        if length <= 0:
            length = np.sqrt(area_cm2)
        radius = max(area_cm2/(2*length), 0.1)
        height_ratio = getattr(prior, "height_ratio", 0.5)
        height = 2*radius*height_ratio

        volume = np.pi*(radius**2)*height

        volume_modifier = getattr(prior, "volume_modifier", 0.8)
        return volume * volume_modifier

    @staticmethod
    def _estimate_irregular_volume(depth_map:np.ndarray, mask:np.ndarray,
                                   area_cm2: float, prior:ShapePrior,
                                   pixel_scale:float) -> float:
        """
        Estimate volume for irregularly shaped foods using depth integration.

        Args:
            depth_map: 2D array of depth values
            mask: Binary mask indicating food region
            area_cm2: Area in square centimeters
            prior: ShapePrior object containing shape information
            pixel_scale: Scale factor to convert pixels to centimeters

        Returns:
            float: Estimated volume in cubic centimeters
        """
        # Apply mask to depth map
        masked_depth = depth_map * mask
        # Convert depth values to real-world units (centimeters)
        scaled_depth = masked_depth * pixel_scale
        # Calculate volume by integrating depth values
        # Since area_cm2 is already in cm², multiply by scaled depth to get cm³
        pixel_area = np.sum(mask)
        avg_depth_cm = np.sum(scaled_depth) / pixel_area if pixel_area > 0 else 0
        volume_cm3 = area_cm2 * avg_depth_cm
        # Apply shape-specific modifier from prior
        return volume_cm3 * prior.volume_modifier
    @staticmethod
    def _get_contours(mask):
        """Extract contours from binary mask"""
        mask_points = np.array(np.where(mask>0)).T
        if len(mask_points)<4:
            return mask_points
        try:
            hull = ConvexHull(mask_points)
            return mask_points[hull.vertices]
        except Exception as e:
            return mask_points

    def _calculate_confidence(self, mask, depth_map, prior):
        """Calculate confidence score for volume estimation"""
        # Factors affecting confidence:
        # 1. Mask quality (continuity, size)
        mask_quality = self._assess_mask_quality(mask)
        # 2. Depth consistency
        depth_quality = self._assess_depth_quality(depth_map, mask)
        # 3. Shape prior reliability
        shape_confidence = 0.9 if prior.shape != 'irregular' else 0.7
        return (mask_quality * self.confidence_weights["mask"] +
                depth_quality * self.confidence_weights["depth"] +
                shape_confidence * self.confidence_weights["shape"])

    @staticmethod
    def _assess_mask_quality(mask):
        """Assess quality of segmentation mask"""
        if not mask.any():
            return 0.0
        # Check mask size and continuity
        total_pixels = mask.size
        mask_pixels = np.sum(mask)

        if mask_pixels < 100: # Too small for estimation
            return .3
        # Check mask continuity using connected components
        from scipy.ndimage import label
        labeled, num_features = label(mask)
        if num_features>3:
            return .7      # Multiple disconnected regions
        return .9

    @staticmethod
    def _assess_depth_quality(depth_map, mask):
        """Assess quality of depth map measurements"""
        masked_depth = depth_map * mask
        valid_depths = masked_depth[mask > 0]

        if len(valid_depths) == 0:
            return 0.0
        depth_mean = np.mean(valid_depths)
        if depth_mean == 0:
            return 0.0

        # Check depth variance
        depth_std = np.std(valid_depths)
        relative_std = depth_std/depth_mean

        return 0.6 if relative_std > 0.5 else 0.9

class UnifiedFoodEstimator:
    def __init__(self, category_path:str="category.txt"):
        self.category_path = category_path
        self.hybrid_estimator = HybridPortionEstimator()
        self.uec_estimator = UECVolumeEstimator()
        self.food_category_map = self._load_uec_categories()

    def _load_uec_categories(self):
        """
        Loads category mapping from a file (e.g., 'category.txt')
        If the file doesn't exist, a default mapping is used.
        """
        mapping = {}
        category_file = Path(self.category_path)
        if category_file.exists():
            with open(category_file, "r") as f:
                lines = f.readlines()
                # Skip header
                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            cat_id = int(parts[0])
                            cat_name = " ".join(parts[1:])
                            mapping[cat_id] = cat_name
                        except ValueError:
                            continue
        else:
            # Default mapping if file not found
            mapping = {i:f"Category_{i}" for i in range(1,257)}
        return mapping

    def _calculate_nutrition(self, volume, food_class, density):
        nutrition = {}
        if self.hybrid_estimator.nutrition_mapper:
            try:
                nutrition = self.hybrid_estimator.nutrition_mapper.map_food_label_to_nutrition(food_class) or {}
            except (KeyError, ConnectionError) as e:
                logging.warning(f"Failed to get nutrition for {food_class}: {e}")
        calories_per_100g = nutrition.get("calories_per_100g", 150)
        weight_g = volume * density
        calories = (weight_g / 100) * calories_per_100g
        return weight_g, calories

    def estimate(self, image, detections):
        """Unified estimation pipeline
        Args:
            image
            detections: List of dicts with keys:
                - 'bbox': [x1,y1,x2,y2]
                - 'label': category ID
                - 'mask': binary segmentation mask
                - 'depth': cropped depth map for the food item
        """
        results = []
        depth_map = self.hybrid_estimator._get_depth_map(image)
        for food in detections:
            # Get UEC-specific metadata
            category_id = food['label']
            food_class = self.food_category_map.get(category_id, "unknown")

            if food_class in self.uec_estimator.food_shape_priors:
                # Use UEC-optimized estimation
                volume, confidence = self.uec_estimator.estimate_volume(
                    depth_map=depth_map,
                    mask=food['mask'],
                    food_class=food_class,
                    reference_scale=self.hybrid_estimator.get_scale(image)
                )
                method = "uec_prior"
            else:
                # Fallback to hybrid method with a default confidence
                portion = self.hybrid_estimator.estimate_portion(
                    image=image,
                    food_boxes=[food['bbox']],
                    food_labels=[food_class],
                    masks=[food["mask"]] if "mask" in food else None)[0]
                volume = portion["volume"]
                mask_quality = self.uec_estimator._assess_mask_quality(food["mask"])
                depth_quality = self.uec_estimator._assess_depth_quality(depth_map, food["depth"])
                confidence = (mask_quality * 0.5 + depth_quality * 0.5)
                method = "hybrid"

            density = self.hybrid_estimator._get_food_density(food_class)
            weight_g, calories = self._calculate_nutrition(volume, food_class, density)

            results.append({
                'volume': volume,
                'weight_g': weight_g,
                'calories': calories,
                'confidence': confidence,
                'method': method
            })
        return results