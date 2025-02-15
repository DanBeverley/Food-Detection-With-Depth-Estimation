import cv2
from ml_pipeline.models.food_detector import FoodDetector
from ml_pipeline.models.food_classifier import FoodClassifier
from ml_pipeline.data_processing.volume_estimation import HybridPortionEstimator

def test_pipeline():
    detector = FoodDetector()
    classifier = FoodClassifier()
    estimator = HybridPortionEstimator()

    image = cv2.imread("test_image.jpg")
    detections = detector.detect(image, return_masks=True)

    results = []
    for det in detections:
        cls_result = classifier.classify_crop(image, det["bbox"])
        portion = estimator.estimate_portion(image=image, food_boxes=[det["bbox"]],
                                             food_labels=[cls_result["class_id"]],
                                             masks=[det.get("mask")])
        results.append(portion)
    assert len(results)>0, "Pipeline failed"

