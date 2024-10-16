"""
Test script for dataloader, to ensure that the images and masks are loaded properly
"""

import numpy as np
from tqdm import tqdm
import os
import cv2

import torch
from torch.utils.data import DataLoader

# Get the training data
images_dir = os.path.join(os.getcwd(), "data/train_images")
masks_dir  = os.path.join(os.getcwd(), "data/train_masks")

from utils.dataset_prep import medical_train_eval_split
from utils.globals import *
from utils.metrics import calculate_metrics_by_pixel, print_metrics

# Get the train and test dataset
train_dataset, test_dataset = medical_train_eval_split(images_dir, masks_dir)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=1, shuffle=True)

for inputs, targets in train_dataloader:
    # Convert the inputs back to images
    inputs = inputs.squeeze(0).cpu().numpy()
    image = np.transpose(inputs, (1, 2, 0))

    # Convert the masks to 255
    targets = targets.squeeze(0).cpu().numpy()
    targets = np.transpose(targets, (1, 2, 0)) * int(255 / 9)

    # Show the results
    for i in range(9):
        print(f"Mask {i} max : {np.max(targets[:, :, i])} min ; {np.min(targets[:, :, i])}")
        cv2.imshow(f"mask_{i}", targets[:, :, i])
    cv2.imshow("Image", image)
    cv2.waitKey(10)
