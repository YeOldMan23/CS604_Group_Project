from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm

from .globals import *

class MedicalDataLoader(DataLoader):
    def __init__(self, data_path, masks_path):
        self.data_path  = data_path
        self.mask_path  = masks_path

        # List the paths down and their specific masks
        self.cwd         = os.getcwd()
        self.images_list = [cv2.imread(os.path.join(self.cwd, img_dir)) for img_dir in os.listdir(self.data_path)]

        masks_list  = os.listdir(self.mask_path)
        # Need to convert the mask to 1 hot encoding
        self.one_hot_mask_list = []

        for mask_dir in masks_list:
            converted_mask = self.amend_mask(mask_dir)
            self.one_hot_mask_list.append(converted_mask)
        
    def amend_mask(self, number_based_mask : np.ndarray) -> np.ndarray:
        """
        Conver the number based mask to one hot encoding
        """
        output_channels = len(CLASS_ENCODING)

        input_mask  = cv2.imread(os.path.join(self.cwd, number_based_mask), cv2.IMREAD_GRAYSCALE)
        output_mask = np.zeros((input_mask.shape[0], input_mask.shape[1], output_channels), dtype=np.int8)

        # Convert the masks to one hot encoding
        for class_index in CLASS_ENCODING.keys():
            class_mask = np.where(input_mask == class_index, 1, 0)
            output_mask[:, :, class_index] = class_mask

        return output_mask
    
    def __len__(self):
        return len(os.listdir(self.data_path))
    
    def __getitem__(self, idx):
        chosen_image = self.images_list[idx]
        chosen_mask  = self.one_hot_mask_list[idx]

        return chosen_image, chosen_mask