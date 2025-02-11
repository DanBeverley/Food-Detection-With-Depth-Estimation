import numpy as np
import torch
import cv2
import csv

class DepthProcessor:
    def __init__(self, depth_dir:str, focal_strength:float=525.0):
        """
        Args
        :param depth_dir(str): Root directory containing depth maps
        :param focal_strength:
        """