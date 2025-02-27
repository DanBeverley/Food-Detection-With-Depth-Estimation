import logging
import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List

class FoodShape(Enum):
    DOME = "dome"  # For mounded foods like rice
    CYLINDER = "cylinder"  # For rolls, sushi
    RECTANGULAR = "rectangular"  # For blocks of food, tempura
    BOWL_CONTENT = "bowl_content"  # For soups, noodles
    LAYERED = "layered"  # For sandwiches, layer cakes
    IRREGULAR = "irregular"  # For non-standard shapes
    SPHERE = "sphere"  # For round fruits, meatballs
    ELLIPSOID = "ellipsoid"  # For eggs, croquettes

@dataclass
class ShapePrior:
    shape:FoodShape
    height_ratio:float  # Height to width ratio
    volume_modifier:float # Accounts for density, air gaps
    typical_serving_cm3:Optional[float]= None # Reference volume
    sub_components:Optional[List[str]] = None # For composite dishes
    special_handling:Optional[str] = None # special processing instructions
    liquid_ratio:Optional[float] = None # Added for bowl_content
    solid_ratio:Optional[float] = None # Added for bowl_content
    def __post_init__(self):
        if self.shape != FoodShape.BOWL_CONTENT and (self.liquid_ratio is not None or self.solid_ratio is not None):
            raise ValueError("liquid_ratio and solid_ratio are only valid for BOWL_CONTENT shapes")
        if self.liquid_ratio is not None and self.solid_ratio is not None:
            if not(0<=self.liquid_ratio<=1 and 0<=self.solid_ratio<=1):
                raise ValueError("liquid_ratio and solid_ratio must be between 0 and 1")
            if abs(self.liquid_ratio + self.solid_ratio - 1.0)>1e-6:
                raise ValueError("liquid_ratio and solid_ratio must sum to 1.0")

class UEC256ShapeMapper:
    def __init__(self):
        self.category_map = self._initialize_category_map()
        if not isinstance(self.category_map, dict):
            raise ValueError("Category map must be a dictionary")
        if "default" not in self.category_map:
            logging.warning("Default shape prior is missing in category_map")
            self.category_map["default"] = ShapePrior(shape=FoodShape.IRREGULAR,
                                                      height_ratio=0.5, volume_modifier=0.85,
                                                      typical_serving_cm3 = None)
    @staticmethod
    def _initialize_category_map()->Dict[str, ShapePrior]:
        """Initialize shape mapping for UEC-256 food categories.
        Based on typical Japanese food serving styles and shapes"""
        return {
            # Rice dishes
            "rice": ShapePrior(
                shape=FoodShape.DOME,
                height_ratio=0.6,
                volume_modifier=1.0,
                typical_serving_cm3=180
            ),
            "fried_rice": ShapePrior(
                shape=FoodShape.DOME,
                height_ratio=0.4,
                volume_modifier=0.85,
                typical_serving_cm3=250
            ),
            "onigiri": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.6,
                volume_modifier=0.95,
                typical_serving_cm3=120
            ),
            "donburi": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.8,
                volume_modifier=1.0,
                typical_serving_cm3=380,
                liquid_ratio=0.15,
                solid_ratio=0.85,
                sub_components=["rice", "toppings", "sauce"]
            ),

            # Noodle dishes
            "ramen": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.6,
                volume_modifier=0.95,
                typical_serving_cm3=550,
                liquid_ratio=0.7,
                solid_ratio=0.3,
                sub_components=["broth", "noodles", "toppings"]
            ),
            "udon": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.8,
                volume_modifier=0.95,
                typical_serving_cm3=450,
                liquid_ratio=0.6,
                solid_ratio=0.4,
                sub_components=["broth", "noodles", "toppings"]
            ),
            "soba": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.7,
                volume_modifier=0.95,
                typical_serving_cm3=400,
                liquid_ratio=0.6,
                solid_ratio=0.4,
                sub_components=["broth", "noodles", "toppings"]
            ),
            "yakisoba": ShapePrior(
                shape=FoodShape.DOME,
                height_ratio=0.45,
                volume_modifier=0.9,
                typical_serving_cm3=300
            ),

            # Sushi and rolls
            "sushi_roll": ShapePrior(
                shape=FoodShape.CYLINDER,
                height_ratio=1.0,
                volume_modifier=0.85,
                typical_serving_cm3=35
            ),
            "nigiri_sushi": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.4,
                volume_modifier=0.95,
                typical_serving_cm3=25
            ),
            "temaki": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.9,
                volume_modifier=0.8,
                typical_serving_cm3=70
            ),
            "chirashi": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.6,
                volume_modifier=0.9,
                typical_serving_cm3=350,
                liquid_ratio=0.05,
                solid_ratio=0.95,
                sub_components=["rice", "fish", "vegetables"]
            ),

            # Fried foods
            "tempura": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.6,
                volume_modifier=0.7,  # Accounts for air gaps in batter
                typical_serving_cm3=120
            ),
            "tonkatsu": ShapePrior(
                shape=FoodShape.RECTANGULAR,
                height_ratio=0.3,
                volume_modifier=0.9,
                typical_serving_cm3=200
            ),
            "karaage": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.7,
                volume_modifier=0.8,
                typical_serving_cm3=150
            ),
            "korokke": ShapePrior(
                shape=FoodShape.ELLIPSOID,
                height_ratio=0.6,
                volume_modifier=0.85,
                typical_serving_cm3=120
            ),
            "takoyaki": ShapePrior(
                shape=FoodShape.SPHERE,
                height_ratio=1.0,
                volume_modifier=0.95,
                typical_serving_cm3=20
            ),

            # Soups and stews
            "miso_soup": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.9,
                volume_modifier=1.0,
                typical_serving_cm3=200,
                liquid_ratio=0.8,
                solid_ratio=0.2
            ),
            "curry": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.6,
                volume_modifier=0.92,
                typical_serving_cm3=350,
                liquid_ratio=0.7,
                solid_ratio=0.3
            ),
            "oden": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.8,
                volume_modifier=0.9,
                typical_serving_cm3=350,
                liquid_ratio=0.6,
                solid_ratio=0.4,
                sub_components=["broth", "various_items"]
            ),
            "sukiyaki": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.7,
                volume_modifier=0.9,
                typical_serving_cm3=400,
                liquid_ratio=0.5,
                solid_ratio=0.5,
                sub_components=["broth", "meat", "vegetables", "tofu"]
            ),
            "shabu_shabu": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.8,
                volume_modifier=0.95,
                typical_serving_cm3=450,
                liquid_ratio=0.55,
                solid_ratio=0.45,
                sub_components=["broth", "meat", "vegetables"]
            ),

            # Side dishes
            "tamagoyaki": ShapePrior(
                shape=FoodShape.RECTANGULAR,
                height_ratio=0.5,
                volume_modifier=0.95,
                typical_serving_cm3=100
            ),
            "gyoza": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.3,
                volume_modifier=0.9,
                typical_serving_cm3=20
            ),
            "edamame": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.4,
                volume_modifier=0.7,
                typical_serving_cm3=80
            ),
            "tsukemono": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.2,
                volume_modifier=0.9,
                typical_serving_cm3=50
            ),
            "chawanmushi": ShapePrior(
                shape=FoodShape.CYLINDER,
                height_ratio=1.2,
                volume_modifier=0.95,
                typical_serving_cm3=150
            ),

            # Grilled foods
            "yakitori": ShapePrior(
                shape=FoodShape.CYLINDER,
                height_ratio=0.8,
                volume_modifier=0.9,
                typical_serving_cm3=60
            ),
            "teriyaki_chicken": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.4,
                volume_modifier=0.9,
                typical_serving_cm3=180
            ),
            "grilled_fish": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.3,
                volume_modifier=0.85,
                typical_serving_cm3=150
            ),
            "yakiniku": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.3,
                volume_modifier=0.9,
                typical_serving_cm3=200
            ),

            # Desserts
            "mochi": ShapePrior(
                shape=FoodShape.SPHERE,
                height_ratio=1.0,
                volume_modifier=0.95,
                typical_serving_cm3=30
            ),
            "dorayaki": ShapePrior(
                shape=FoodShape.CYLINDER,
                height_ratio=0.4,
                volume_modifier=0.9,
                typical_serving_cm3=80
            ),
            "taiyaki": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.6,
                volume_modifier=0.8,
                typical_serving_cm3=120
            ),
            "dango": ShapePrior(
                shape=FoodShape.SPHERE,
                height_ratio=1.0,
                volume_modifier=0.95,
                typical_serving_cm3=15,
                special_handling="Typically comes in groups of 3-5"
            ),
            "anmitsu": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.7,
                volume_modifier=0.9,
                typical_serving_cm3=200,
                liquid_ratio=0.4,
                solid_ratio=0.6,
                sub_components=["jelly", "fruits", "mochi", "red_bean"]
            ),

            # Bread and pastries
            "melon_pan": ShapePrior(
                shape=FoodShape.DOME,
                height_ratio=0.6,
                volume_modifier=0.7,  # Accounts for air in bread
                typical_serving_cm3=200
            ),
            "anpan": ShapePrior(
                shape=FoodShape.SPHERE,
                height_ratio=0.9,
                volume_modifier=0.8,
                typical_serving_cm3=120
            ),
            "curry_bread": ShapePrior(
                shape=FoodShape.ELLIPSOID,
                height_ratio=0.7,
                volume_modifier=0.8,
                typical_serving_cm3=150
            ),

            # Vegetables and salads
            "hijiki": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.4,
                volume_modifier=0.7,
                typical_serving_cm3=60
            ),
            "goma_ae": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.5,
                volume_modifier=0.8,
                typical_serving_cm3=80
            ),
            "sunomono": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.5,
                volume_modifier=0.9,
                liquid_ratio=0.2,
                solid_ratio=0.8,
                typical_serving_cm3=100
            ),

            # Tofu dishes
            "agedashi_tofu": ShapePrior(
                shape=FoodShape.RECTANGULAR,
                height_ratio=0.8,
                volume_modifier=0.9,
                typical_serving_cm3=150,
                sub_components=["tofu", "broth"]
            ),
            "hiyayakko": ShapePrior(
                shape=FoodShape.RECTANGULAR,
                height_ratio=0.8,
                volume_modifier=0.95,
                typical_serving_cm3=120
            ),
            "mapo_tofu": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.7,
                volume_modifier=0.95,
                typical_serving_cm3=250,
                liquid_ratio=0.65,
                solid_ratio=0.35,
                sub_components=["tofu", "sauce", "meat"]
            ),
            "okonomiyaki": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.5,
                volume_modifier=1.0,
                typical_serving_cm3=220,
                sub_components=["batter", "cabbage", "meat", "negi"]
            ),
            # Default for unknown categories
            "default": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.5,
                volume_modifier=0.85,
                typical_serving_cm3=None
            )
        }
    def get_shape_prior(self, food_category:str) -> ShapePrior:
        if not food_category:
            logging.warning("Empty food category provided")
            return self.category_map["default"]
        """Get shape prior for a given food category"""
        # Clean the input category string
        category = re.sub(r'\s+', '_', food_category.lower().strip())
        prior = self.category_map.get(category)
        if prior is not None:
            return prior
        keywords = category.split("_")
        for keyword in keywords:
            if len(keyword) < 3: # For skipping short words
                continue
            matches  = [cat for cat in self.category_map.keys()
                        if keyword in cat.split("_") and cat != "default"]
            if matches:
                best_match = matches[0]
                logging.info(f"No exact match for '{food_category}', using '{best_match}' based on keyword '{keyword}'")
                return self.category_map[best_match]
        logging.info(f"No shape prior found for '{food_category}', using default")
        return self.category_map["default"]

    def add_custom_category(self, category:str, shape:FoodShape,
                            height_ratio:float, volume_modifier:float,
                            typical_serving_cm3:Optional[float] = None,
                            sub_components:Optional[List[str]] = None) -> None:
        """Add a custom food category with its shape prior"""
        if height_ratio <= 0 or volume_modifier <= 0:
            raise ValueError("height_ratio and volume_modifier must be positive values")
        category = category.lower().strip().replace(" ", "_")
        if category in self.category_map:
            logging.warning(f"Overwritting existing category '{category}'")
        self.category_map[category] = ShapePrior(shape = shape,
                                                 height_ratio = height_ratio,
                                                 volume_modifier = volume_modifier,
                                                 typical_serving_cm3 = typical_serving_cm3,
                                                 sub_components = sub_components)

    def get_similar_categories(self, food_category:str, threshold:float=0.2)->List[str]:
        """Find similar food categories based on shape categories"""
        target_prior = self.get_shape_prior(food_category)
        similar_categories = []
        for category, prior in self.category_map.items():
            if (prior.shape == target_prior.shape and
                    abs(prior.height_ratio-target_prior.height_ratio)<threshold):
                similar_categories.append(category)
        return similar_categories
