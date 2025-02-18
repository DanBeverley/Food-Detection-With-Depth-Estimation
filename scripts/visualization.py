import matplotlib.pyplot as plt
import cv2

class FoodVisualizer:
    @staticmethod
    def plot_detection(image, detections, nutrition_info):
        plt.figure(figsize=(12,8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for det, nutrition in zip(detections, nutrition_info):
            x1, y1, x2, y2 = det["bbox"]
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                              fill=False, edgecolor="red", linewidth=2))
            text = f"{det['class_name']}\nCal: {nutrition['calories']:.1f}kcal"
            plt.text(x1, y1-10, text, bbox=dict(facecolor="yellow", alpha=0.8))
        plt.axis("off")
        plt.tight_layout()
        plt.show()