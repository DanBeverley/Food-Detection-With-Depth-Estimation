import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert
from scipy.spatial import ConvexHull
from functools import lru_cache
from shape_mapping import ShapePrior


class HybridPortionEstimator:
    def __init__(self, device=None, nutrition_mapper=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nutrition_mapper = nutrition_mapper
        self.depth_cache = {}
        self._load_models()

        # Food type to density mapping (g/cm³)
        self.density_db = {
            'rice': 0.9,  # Cooked rice
            'bread': 0.25,  # Sliced bread
            'meat': 1.1,  # Cooked beef
            'vegetables': 0.6
        }

    def _load_models(self):
        """Load both reference object detector and depth estimation model"""
        try:
            # Load reference object detector (YOLOv8)
            self.ref_detector = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True)
        except Exception as e:
            logging.error(f"Failed to load YOLOv8: {e}")
            raise RuntimeError("Reference detector loading failed")
        try:
            # Load MiDaS for depth estimation
            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to(self.device)
            self.midas_transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
        except Exception as e:
            logging.error(f"Failed to load MiDaS: {e}")
            raise RuntimeError("Depth estimation model loading failed")
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

    def estimate_portion(self, image, food_boxes, food_labels, masks=None):
        """
        Hybrid estimation pipeline:
        1. Reference scale detection
        2. Food-specific density lookup
        3. Depth-aware volume estimation
        4. Nutritional mapping
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Detect reference objects (plates, credit cards, etc.)
        ref_objects = self._detect_reference_objects(img_rgb)

        # 2. Calculate pixel-to-cm ratio
        scale_ratio = self._calculate_scale(ref_objects, known_width_cm=8.5)

        # 3. Get depth map
        depth_map = self._get_depth_map(img_rgb)

        portions = []
        for idx,(box, label) in enumerate(zip(food_boxes, food_labels)):
            if masks and idx<len(masks):
                mask_area = np.sum(masks[idx])
                area_cm2 = mask_area*(scale_ratio**2)
            else:
                # Fallback to bbox area
                w = box[2]-box[0]
                h = box[3]-box[1]
                area_cm2 = (w*h)*(scale_ratio**2)

            # Convert to xywh format
            x, y, w, h = box_convert(box, 'xyxy', 'xywh').tolist()

            # 4. Food specific density
            density = self._get_food_density(label)

            # 5. Calculate physical dimensions
            area_cm2 = (w*h)*(scale_ratio**2)
            depth_cm = self._get_region_depth(depth_map, box)*scale_ratio

            volume_cm3 = area_cm2*depth_cm

            # 6. Weights and calories
            weight_g = volume_cm3*density
            calories = self._estimate_calories(label, weight_g)

            portions.append({
                "weight":weight_g,
                "calories":calories,
                "volume":volume_cm3,
                "bounding_box":box
            })
        return portions

    def _detect_reference_objects(self, image):
        """Detect known reference objects using YOLOv8"""
        results = self.ref_detector(image)
        reference_classes = ["plate", "credit card", "bowl"]

        ref_objects = []
        for *box, conf, cls in results.pred[0]:
            label = self.ref_detector.name[int(cls)]
            if label in reference_classes:
                ref_objects.append({'box':box,'label':label,'confidence':conf})
        return ref_objects

    @staticmethod
    def _calculate_scale(ref_objects, known_width_cm):
        """calculate scale using best reference object"""
        if not ref_objects:
            return 22/640 # Fallback default

        # Select most confident reference object
        best_ref = max(ref_objects, key=lambda x:x["confidence"])
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
    @lru_cache(maxsize=128)
    def _get_depth_map(self, image:np.ndarray) -> np.ndarray:
        """Get depth map using MiDaS"""
        if hasattr(image, "filename"):
            cache_key = Path(image.filename).stem
        else:
            cache_key = hash(image.tobytes())
        # Check cache first
        if cache_key in self.depth_cache:
            return self.depth_cache[cache_key]
        # Compute and cache
        depth = self.midas(image)
        self.depth_cache[cache_key] = depth
        return depth

    @staticmethod
    def _calculate_volume_confidence(depth_quality:float, mask_quality:float,
                                     reference_confidence:float)->float:
        """Calculate overall confidence score for volume estimation"""
        weights = {"depth":0.4, "mask":0.4, "reference":0.2}
        confidence = (depth_quality * weights["depth"] +
                      depth_quality * weights["mask"] +
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
    def __init__(self, food_shape_priors = None):
        self.food_shape_priors = food_shape_priors or self._get_default_shape_prior()
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
        shape_prior = self.food_shape_priors.get(food_class, self.food_shape_priors["default"])

        # Calculate base metrics
        area_pixels = np.sum(mask)
        area_cm2 = area_pixels*(reference_scale**2)

        if shape_prior['shape'] == 'dome':
            volume = self._estimate_dome_volume(depth_map, mask, area_cm2, shape_prior)
        elif shape_prior['shape'] == 'bowl_content':
            volume = self._estimate_bowl_content_volume(depth_map, mask, area_cm2, shape_prior)
        elif shape_prior['shape'] == 'cylinder':
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
        max_height = np.max(masked_depth(mask>0))

        # Spherical cap formula with modification
        radius = np.sqrt(area_cm2/np.pi)
        height = max_height*prior["height_ratio"]
        volume = (1/6)*np.pi*height*(3*radius**2+height**2)
        return volume*prior.volume_modifier
    @staticmethod
    def _estimate_bowl_content_volume(depth_map, mask, area_cm2, prior):
        """Estimate volume for food served in bowls"""
        masked_depth = depth_map*mask
        mean_depth = np.mean(masked_depth[mask>0])

        # Calculate bowl volume and apply content ratio
        bowl_volume = area_cm2*mean_depth
        liquid_volume = bowl_volume*prior["liquid_ratio"]
        solid_volume = bowl_volume*prior["solid_ratio"]

        return liquid_volume + solid_volume

    def _estimate_cylinder_volume(self, depth_map, mask, area_cm2, prior):
        """Estimate volume for cylindrical foods e.g. sushi roll"""
        # Use contour analysis to get the length
        contours = self._get_contours(mask)
        length = np.max(np.ptp(contours, axis=0))

        # Calculate radius from area and length
        radius = area_cm2/(2*length)

        # Use height from prior
        height = 2*radius*prior["height_ratio"]
        return np.pi*(radius**2)*height
    @staticmethod
    def _estimate_irregular_volume(self, depth_map:np.ndarray, mask:np.ndarray,
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
        # Each depth value needs to be scaled by pixel_scale since it's in the same
        # coordinate system as the x,y dimensions
        scaled_depth = masked_depth * pixel_scale

        # Calculate volume by integrating depth values
        # Since area_cm2 is already in cm², multiply by scaled depth to get cm³
        pixel_area = np.sum(mask)  # Count of pixels in mask
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
        shape_confidence = 0.9 if prior['shape'] != 'irregular' else 0.7

        return mask_quality * 0.4 + depth_quality * 0.4 + shape_confidence * 0.2

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
    def __init__(self):
        self.hybrid_estimator = HybridPortionEstimator()
        self.uec_estimator = UECVolumeEstimator()
        self.food_category_map = self._load_uec_categories()
    @staticmethod
    def _load_uec_categories():
        """
        Loads category mapping from a file (e.g., 'category.txt')
        If the file doesn't exist, a default mapping is used.
        """
        category_file = "/category.txt"
        mapping = {}
        if os.path.exists(category_file):
            with open(category_file, "r") as f:
                lines = f.readlines()
                # Skip header
                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts)>=2:
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
                    depth_map=food['depth'],
                    mask=food['mask'],
                    food_class=food_class,
                    reference_scale=self.hybrid_estimator.get_scale(image)
                )
            else:
                # Fallback to hybrid method with a default confidence
                volume = self.hybrid_estimator.estimate_portion(
                    image=image,
                    food_boxes=[food['bbox']],
                    food_labels=[food_class]
                )[0]['volume']
                confidence = 1.0

            # Add nutrition data from hybrid system
            nutrition = self.hybrid_estimator.nutrition_mapper.map_food_label_to_nutrition(food_class)
            results.append({
                'volume': volume,
                'calories': volume * nutrition['calories_per_cm3'],
                'confidence': confidence,
                'method': 'uec_prior' if food_class in self.uec_estimator.food_shape_priors else 'hybrid'
            })

        return results