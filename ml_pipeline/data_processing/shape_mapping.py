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

class UEC256ShapeMapper:
    def __init__(self):
        self.category_map = self._initialize_category_map()
    @staticmethod
    def _initialize_category_map()->Dict[str, ShapePrior]:
        """Initialize shape mapping for UEC-256 food categories.
        Based on typical Japanese food serving styles and shapes"""
        return {
            # Rice dishes
            "rice": ShapePrior(
                shape=FoodShape.DOME,
                height_ratio=0.5,
                volume_modifier=0.9,
                typical_serving_cm3=200
            ),
            "fried_rice": ShapePrior(
                shape=FoodShape.DOME,
                height_ratio=0.4,
                volume_modifier=0.85,
                typical_serving_cm3=250
            ),

            # Noodle dishes
            "ramen": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.8,
                volume_modifier=0.95,
                typical_serving_cm3=500,
                sub_components=["broth", "noodles", "toppings"]
            ),
            "udon": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.8,
                volume_modifier=0.95,
                typical_serving_cm3=450
            ),

            # Sushi and rolls
            "sushi_roll": ShapePrior(
                shape=FoodShape.CYLINDER,
                height_ratio=1.0,
                volume_modifier=0.9,
                typical_serving_cm3=30
            ),
            "nigiri_sushi": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.4,
                volume_modifier=0.95,
                typical_serving_cm3=25
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

            # Soups and stews
            "miso_soup": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.9,
                volume_modifier=1.0,
                typical_serving_cm3=200
            ),
            "curry": ShapePrior(
                shape=FoodShape.BOWL_CONTENT,
                height_ratio=0.7,
                volume_modifier=0.95,
                typical_serving_cm3=300
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

            # Default for unknown categories
            "default": ShapePrior(
                shape=FoodShape.IRREGULAR,
                height_ratio=0.5,
                volume_modifier=0.85,
                typical_serving_cm3=None
            )
        }
    def get_shape_prior(self, food_category:str)->ShapePrior:
        """Get shape prior for a given food category"""
        # Clean the input category string
        category = food_category.lower().strip().replace(" ","_")
        return self.category_map.get(category, self.category_map["default"])

    def add_custom_category(self, category:str, shape:FoodShape,
                            height_ratio:float, volume_modifier:float,
                            typical_serving_cm3:Optional[float] = None,
                            sub_components:Optional[List[str]] = None)->None:
        """Add a custom food category with its shape prior"""
        self.category_map[category] = ShapePrior(shape = shape,
                                                 height_ratio = height_ratio,
                                                 volume_modifier = volume_modifier,
                                                 typical_serving_cm3 = typical_serving_cm3,
                                                 sub_components = sub_components)

    def get_similar_categories(self, food_category:str)->List[str]:
        """Find similar food categories based on shape categories"""
        target_prior = self.get_shape_prior(food_category)
        similar_categories = []

        for category, prior in self.category_map.items():
            if (prior.shape == target_prior.shape and
                    abs(prior.height_ratio-target_prior.height_ratio)<0.2):
                similar_categories.append(category)
        return similar_categories
