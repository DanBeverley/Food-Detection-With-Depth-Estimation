import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_convert
from PIL import Image
from scipy.spatial import ConvexHull
from dataset_loader import UECFoodDataset
from nutrition_mapper import NutritionMapper


class HybridPortionEstimator:
    def __init__(self, device=None, nutrition_mapper=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nutrition_mapper = nutrition_mapper
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

            # Load MiDaS for depth estimation
            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to(self.device)
            self.midas_transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def estimate_portion(self, image, food_boxes, food_labels):
        """
        Hybrid estimation pipeline:
        1. Reference scale detection
        2. Food-specific density lookup
        3. Depth-aware volume estimation
        4. Nutritional mapping
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Detect reference objects (plates, credit cards, etc)
        ref_objects = self._detect_reference_objects(img_rgb)

        # 2. Calculate pixel-to-cm ratio
        scale_ratio = self._calculate_scale(ref_objects, known_width_cm=8.5)

        # 3. Get depth map
        depth_map = self._get_depth_map(img_rgb)

        portions = []
        for box, label in zip(food_boxes, food_labels):
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

    def _calculate_scale(self, ref_objects, known_width_cm):
        """calculate scale using best reference object"""
        if not ref_objects:
            return 0.1 # Fallback default

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

    def _get_depth_map(self, image):
        """Get depth map using MiDaS"""
        input_tensor = self.midas_transform(Image.fromarray(image)).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_tensor.unsqueeze(0))
            prediction = F.interpolate(prediction.unsqueeze(1),
                                       size = image.shape[:2],
                                       mode = "bicubic",
                                       align_corners = False,)
        return prediction.squeeze().cpu().numpy()

    def _get_food_density(self, label):
        """Get density from database or nutrition mapper"""
        if self.nutrition_mapper:
            nutrition = self.nutrition_mapper.map_food_label_to_nutrion(label)
            if nutrition and "density" in nutrition:
                return nutrition["density"]
        return self.density_db.get(label.lower(), 0.8) # Default

    def _estimate_calories(self, label, weight_g):
        # Estimate calories with nutrition mapper
        if self.nutrition_mapper:
            nutrition = self.nutrition_mapper.map_food_label_to_nutrion(label)
            if nutrition and nutrition["calories_per_100g"]:
                return (weight_g/100)*nutrition["calories_per_100g"]
        return weight_g*1.5 # Fallback 1.5 cal/g

    def _get_region_depth(self, depth_map, box):
        """Get average depth in food region"""
        x1, y1, x2, y2 = map(int, box)
        region = depth_map[y1:y2, x1:x2]
        return np.mean(region) if region.size>0 else .1


class UECVolumeEstimator:
    def __init__(self, food_shape_priors = None):
        self.food_shape_priors = food_shape_priors or self._get_default_shape
    def _get_default_shape_prior(self):
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
