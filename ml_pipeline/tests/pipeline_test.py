from ml_pipeline.models.food_detector import FoodDetector
from ml_pipeline.models.food_classifier import FoodClassifier
from ml_pipeline.data_processing.volume_estimation import HybridPortionEstimator

import cv2
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PipelineResult:
    class_id: int
    confidence: float
    portion_size: float
    nutrition_values: Dict[str, float]
    bbox: np.ndarray
    mask: np.ndarray = None


class FoodPipeline:
    def __init__(self, detector, classifier, estimator):
        self.detector = detector
        self.classifier = classifier
        self.estimator = estimator

    def process_image(self, image_path: str) -> List[PipelineResult]:
        """Process single image through entire pipeline"""
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return self.process_array(image)

    def process_array(self, image: np.ndarray) -> List[PipelineResult]:
        """Process numpy array through pipeline"""
        # Get detections with masks
        detections = self.detector.detect(image, return_masks=True)

        results = []
        for det in detections:
            # Get classification
            cls_result = self.classifier.classify_crop(image, det["bbox"])

            # Estimate portion size
            portion = self.estimator.estimate_portion(
                image=image,
                food_boxes=[det["bbox"]],
                food_labels=[cls_result["class_id"]],
                masks=[det.get("mask")]
            )

            # Combine results
            results.append(PipelineResult(
                class_id=cls_result["class_id"],
                confidence=cls_result["confidence"],
                portion_size=portion[0],
                nutrition_values=cls_result.get("nutrition", {}),
                bbox=det["bbox"],
                mask=det.get("mask")
            ))

        return results


def test_pipeline(image:np.array):
    # Initialize components
    detector = FoodDetector()
    classifier = FoodClassifier()
    estimator = HybridPortionEstimator()

    # Create pipeline
    pipeline = FoodPipeline(detector, classifier, estimator)

    # Test processing
    results = pipeline.process_image("test_image.jpg")

    # Assertions
    assert len(results) > 0, "Pipeline failed to detect any food items"
    for result in results:
        assert result.confidence > 0, "Invalid confidence score"
        assert result.portion_size > 0, "Invalid portion size"
        assert all(0 <= coord <= max(image.shape) for coord in result.bbox.flatten()), "Invalid bbox"

