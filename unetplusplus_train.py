"""
Training Script
"""
import numpy as np
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

from utils.dataset_prep import medical_train_eval_split
from utils.globals import *
from utils.metrics import calculate_metrics_by_pixel, print_metrics

# Import the mode
from models.UNetPlusPlus import unet_plus_plus_model

# import the losses
from losses.FocalTverskyLoss import FocalTverskyLoss
from losses.DiceLoss import DiceLoss
from losses.FocalLoss import FocalLoss

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the training data
images_dir = os.path.join(os.getcwd(), "data/train_images")
masks_dir  = os.path.join(os.getcwd(), "data/train_masks")

# Get the train and test dataset
train_dataset, test_dataset = medical_train_eval_split(images_dir, masks_dir, train_split=0.95)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load the model
model = unet_plus_plus_model
model.to(device)

# Load the optimzer and loss
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
# ce_loss = torch.nn.CrossEntropyLoss()
# dice_loss = DiceLoss()
combo_loss = [torch.nn.CrossEntropyLoss(), DiceLoss()]
combo_weights = [1.0, 1.0]
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# We also need to initialize the other losses
loss_pt_2 = FocalLoss(gamma=2)
loss_pt_3 = FocalLoss(gamma=3)
loss_pt_4 = FocalLoss(gamma=5)

# Loss List to keep
train_loss_list = []
test_loss_list  = []
train_metrics_dict_list = []
test_metrics_dict_list  = []

print("Beginning Training!")
for num_epoch in range(1, NUM_EPOCHS):
    train_pbar = tqdm(train_dataloader)
    test_pbar  = tqdm(test_dataloader)

    # Prepare Training 
    epoch_train_loss = 0
    epoch_test_loss  = 0
    epoch_train_dict = {}
    epoch_test_dict  = {}

    # Initialize the metrics dictionary
    for class_name in CLASS_ENCODING.keys():
        epoch_test_dict[class_name]  = {"IoU": 0, "Precision": 0, "Recall": 0, "Accuracy": 0}
        epoch_train_dict[class_name] = {"IoU": 0, "Precision": 0, "Recall": 0, "Accuracy": 0}

    model.train()
    count = 0
    # Do the training part first
    for inputs, targets in train_pbar:
        inputs, targets = inputs.to(device), targets.to(torch.float32).to(device)

        predictions, class_preds = model(inputs)

        cur_loss = 0
        optimizer.zero_grad()
        
        if num_epoch <= 30:
            for loss, weight in zip(combo_loss, combo_weights):
                cur_loss += weight * loss(predictions, targets)
            cur_loss.backward()
        elif num_epoch > 30 and num_epoch <= 40:
            cur_loss = loss_pt_2(predictions, targets)
            cur_loss.backward()
        elif num_epoch > 40 and num_epoch <= 50:
            cur_loss = loss_pt_3(predictions, targets)
            cur_loss.backward()
        elif num_epoch > 50 and num_epoch <= 70:
            cur_loss = loss_pt_4(predictions, targets)
            cur_loss.backward()
        optimizer.step()

        # Add the loss
        epoch_train_loss += cur_loss.item() if type(cur_loss) != int else cur_loss
        count += 1

        # Calculate the metrics
        cur_metrics = calculate_metrics_by_pixel(predictions, targets)

        train_pbar.set_description(f"Epoch {num_epoch} | Training Loss : {epoch_train_loss / count:.5f}")

        # append the metrics to the epoch list
        for i in CLASS_ENCODING.keys():
            epoch_train_dict[i]["IoU"] += cur_metrics[i]["IoU"]
            epoch_train_dict[i]["Precision"] += cur_metrics[i]["Precision"]
            epoch_train_dict[i]["Recall"] += cur_metrics[i]["Recall"]

    # Normalize to the size of dataloader
    epoch_train_loss /= len(train_pbar)
    for i in CLASS_ENCODING.keys():
        epoch_train_dict[i]["IoU"] /= len(train_pbar)
        epoch_train_dict[i]["Precision"] /= len(train_pbar)
        epoch_train_dict[i]["Recall"] /= len(train_pbar)

    print("Training Metrics :")
    print_metrics(epoch_train_dict)
        
    model.eval()
    count = 0
    # Test part
    for inputs, targets in test_pbar:
        inputs, targets = inputs.to(device), targets.to(torch.float32).to(device)

        predictions, class_preds = model(inputs)
        cur_loss = 0
        if num_epoch <= 30:
            for loss, weight in zip(combo_loss, combo_weights):
                cur_loss += weight * loss(predictions, targets)
        elif num_epoch > 30 and num_epoch <= 40:
            cur_loss = loss_pt_2(predictions, targets)
        elif num_epoch > 40 and num_epoch <= 50:
            cur_loss = loss_pt_3(predictions, targets)
        elif num_epoch > 50 and num_epoch <= 70:
            cur_loss = loss_pt_4(predictions, targets)

        epoch_test_loss += cur_loss.item() if type(cur_loss) != int else cur_loss
        count += 1

        # Calculate the metrics
        cur_metrics = calculate_metrics_by_pixel(predictions, targets)

        test_pbar.set_description(f"Epoch {num_epoch} | Testing Loss : {epoch_test_loss / count:.5f}")

        # append the metrics to the epoch list
        for i in CLASS_ENCODING.keys():
            epoch_test_dict[i]["IoU"] += cur_metrics[i]["IoU"]
            epoch_test_dict[i]["Precision"] += cur_metrics[i]["Precision"]
            epoch_test_dict[i]["Recall"] += cur_metrics[i]["Recall"]

    # Normalize to the size of dataloader
    epoch_test_loss /= len(test_pbar)
    for i in CLASS_ENCODING.keys():
        epoch_test_dict[i]["IoU"] /= len(test_pbar)
        epoch_test_dict[i]["Precision"] /= len(test_pbar)
        epoch_test_dict[i]["Recall"] /= len(test_pbar)

    print("Testing Metrics :")
    print_metrics(epoch_test_dict)

    # End of Epoch Updates
    train_loss_list.append(epoch_train_loss)
    test_loss_list.append(epoch_test_loss)
    train_metrics_dict_list.append(epoch_train_dict)
    test_metrics_dict_list.append(epoch_test_dict)

    scheduler.step(num_epoch)
    scheduler.get_last_lr()
