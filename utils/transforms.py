import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch import manual_seed

import numpy as np
from PIL import Image
import random

from .globals import SEED

class ImageAndMasksTransforms():
    def __init__(self, mean : float, std_dev : float, resize_shape = (224, 224), hflip_prob : float = 0.5, vflip_prob : float = 0.5, is_train = False) -> None:
        self.resize_shape = resize_shape

        # transforms and their probs
        self.mean            = mean
        self.std_dev         = std_dev
        self.gaussian_noise  = transforms.GaussianBlur((3, 3))
        self.hflip_prob      = hflip_prob
        self.vflip_prob      = vflip_prob
        self.is_train        = is_train

        # Set the seetds
        np.random.seed(1234)
        manual_seed(1234)
        random.seed(1234)

    def transform_image_and_mask(self, image, mask):
        hflip_prob = random.random()
        vflip_prob = random.random()

        # Test 
        if not self.is_train:
            image = Image.fromarray(image)
            image = transforms.Grayscale()(image)

            mask = transforms.ToTensor()(mask)
            image = transforms.ToTensor()(image)

            # Random Crop the image to thatsize
            # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.resize_shape)
            # image = F.crop(image, i, j, h, w)
            # mask  = F.crop(mask, i, j, h, w)

            # Resize to 224, 224
            # image = transforms.Resize(self.resize_shape)(image)
            # mask  = transforms.Resize(self.resize_shape)(mask)

            return image, mask

        # Transform the image to PIL Image first, mask to tensor
        image = Image.fromarray(image)
        image = transforms.Grayscale()(image)
        mask = transforms.ToTensor()(mask)

        # Resize to 224, 224
        image = transforms.Resize(self.resize_shape)(image)
        mask  = transforms.Resize(self.resize_shape)(mask)

        # Random horizontal flip
        if hflip_prob > self.hflip_prob :
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Random vertical flip
        if vflip_prob > self.vflip_prob:
            image = F.vflip(image)
            mask = F.vflip(mask)

        # Make the image and mask both Tensors
        image = transforms.ToTensor()(image)

        # Normalize
        # image = transforms.Normalize(mean = self.mean, std= self.std_dev)(image)

        return image, mask