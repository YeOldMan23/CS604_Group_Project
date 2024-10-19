from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from tqdm import tqdm

from .globals import *
from .transforms import *

class MedicalDataset(Dataset):
    def __init__(self, data_path, masks_path, transforms : ImageAndMasksTransforms):
        self.data_path  = data_path
        self.mask_path  = masks_path
        self.transforms = transforms

        # List the paths down and their specific masks
        self.images_list = [cv2.imread(img_dir) for img_dir in self.data_path]

        # Need to convert the mask to 1 hot encoding
        self.one_hot_mask_list = []

        for mask_dir in self.mask_path:
            converted_mask = self.amend_mask(mask_dir)
            self.one_hot_mask_list.append(converted_mask)
        
    def amend_mask(self, number_based_mask : np.ndarray) -> np.ndarray:
        """
        Conver the number based mask to one hot encoding
        """
        base_channels = len(CLASS_ENCODING) # Remove the class index 
        output_channels = len(NEW_CLASS_ENCODING)

        input_mask  = cv2.imread(number_based_mask, cv2.IMREAD_GRAYSCALE)
        base_mask = np.zeros((input_mask.shape[0], input_mask.shape[1], base_channels), dtype=np.int8)
        output_mask = np.zeros((input_mask.shape[0], input_mask.shape[1], output_channels), dtype=np.int8)

        # Convert the masks to one hot encoding
        for class_index in CLASS_ENCODING.keys():
            class_mask = (input_mask == class_index).astype(int)
            base_mask[:, :, class_index] = class_mask # Adjust for the class encoding

        # Combine the left and right kidney into one class kidney, drop background mask
        output_mask[:, :, 0] = base_mask[:, :, 1]
        output_mask[:, :, 1] = base_mask[:, :, 2] + base_mask[:, :, 3]
        output_mask[:, :, 2] = base_mask[:, :, 4]
        output_mask[:, :, 3] = base_mask[:, :, 5]
        output_mask[:, :, 4] = base_mask[:, :, 6]
        output_mask[:, :, 5] = base_mask[:, :, 7]
        output_mask[:, :, 6] = base_mask[:, :, 8]

        return output_mask
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        chosen_image = self.images_list[idx]
        chosen_mask  = self.one_hot_mask_list[idx]

        if self.transforms:
            chosen_image, chosen_mask = self.transforms.transform_image_and_mask(chosen_image, chosen_mask)

        return chosen_image, chosen_mask
    

def medical_train_eval_split(images_dir, masks_dir, train_split : float = 0.9):
    """
    Do regular train test split
    """
    assert os.path.exists(images_dir), f"{images_dir} image path does not exist!"
    assert os.path.exists(masks_dir), f"{masks_dir} masks path does not exist!"
    assert 0.0 < train_split < 1.0, f"{train_split} split not between 1.0 and 0"

    images_list = os.listdir(images_dir)
    masks_list  = os.listdir(masks_dir)

    # Make the train test split
    length_train  = int(len(images_list) * train_split)
    indexes       = [i for i in range(len(images_list))]

    train_indexes = np.random.choice(indexes, length_train, replace=False)
    test_indexes  = [j for j in indexes if j not in set(train_indexes)]

    # Split the train and test directories
    train_images_list = [os.path.join(images_dir, images_list[k]) for k in train_indexes]
    train_masks_list  = [os.path.join(masks_dir, masks_list[k]) for k in train_indexes]
    test_images_list  = [os.path.join(images_dir, images_list[k]) for k in test_indexes]
    test_masks_list   = [os.path.join(masks_dir, masks_list[k]) for k in test_indexes]
    
    # Make the train and test dataset
    train_dataset = MedicalDataset(train_images_list, train_masks_list, ImageAndMasksTransforms())
    test_dataset  = MedicalDataset(test_images_list, test_masks_list, ImageAndMasksTransforms())

    return train_dataset, test_dataset 

    