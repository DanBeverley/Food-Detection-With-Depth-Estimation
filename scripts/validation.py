import logging
import cv2
from pathlib import Path

import yaml
from typing import Tuple, Union, List, Dict, Any
from ml_pipeline.models.food_detector import FoodDetector
from ml_pipeline.models.food_classifier import FoodClassifier
from ml_pipeline.data_processing.volume_estimation import HybridPortionEstimator

class FoodSystemValidator:
    def __init__(self, config) -> None:
        self.detector = FoodDetector(**config["detector"])
        self.classifier = FoodClassifier(**config["classifier"])
        self.estimator = HybridPortionEstimator(**config["estimator"])
        self.logger = logging.getLogger("Validation")
    def validate_image(self, img_path:str)  -> Tuple[bool, Union[List[Dict[str, Any]], str]]:
        """Full pipeline validation for single step"""
        try:
            # Load and preprocess
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Invalid image file")
            # Detection
            detections = self.detector.detect(img, return_masks=True)
            self.logger.info(f"Detected {len(detections)} items")

            results = []
            for det in detections:
                # Classification
                cls_result = self.classifier.classify_crop(img, det["bbox"])
                # Nutrition estimation
                portion = self.estimator.estimate_portion(image=img, food_boxes=[det["bbox"]],
                                                          food_labels=[cls_result["class_id"]],
                                                          masks=[det.get("mask")])[0]
                results.append({"bbox":det["bbox"],
                                "class":cls_result,
                               "nutrition":portion})
            return True, results
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return False, str(e)

def run_validation_suite(config:yaml, test_dir:str="test_images") -> None:
    test_images = list(Path(test_dir).glob("*.*"))
    validator = FoodSystemValidator(config) # Load from unified config

    for img_path in test_images:
        success, result = validator.validate_image(str(img_path))
        if success:
            print(f"Validation passed for {img_path.name}")
            print(f"Results: {result}")
        else:
            print(f"Validation failed for {img_path.name}")





