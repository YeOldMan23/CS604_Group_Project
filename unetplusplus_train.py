"""
Training Script
"""
import numpy as np
from tqdm import tqdm
import os
import shutil
import io
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

from utils.dataset_prep import medical_train_eval_split
from utils.globals import *
from utils.metrics import calculate_metrics_by_pixel, print_metrics

# Import the mode
from models.UNetPlusPlus import unet_plus_plus_model
from models.U2Net import U2Net_model, U2NetP_model

# import the losses
from losses.FocalTverskyLoss import FocalTverskyLoss
from losses.U2NetCustomLoss import U2NetCustomLoss
from losses.DiceLoss import DiceLoss
from losses.FocalLoss import FocalLoss

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the training data
images_dir = os.path.join(os.getcwd(), "data/train_images")
masks_dir  = os.path.join(os.getcwd(), "data/train_masks")

if os.path.exists(os.path.join(os.getcwd(), "U2NetPredictions")):
    shutil.rmtree(os.path.join(os.getcwd(), "U2NetPredictions"))
os.mkdir(os.path.join(os.getcwd(), "U2NetPredictions"))
u2_net_pred_images = os.path.join(os.getcwd(), "U2NetPredictions")

# Get the train and test dataset
train_dataset, test_dataset = medical_train_eval_split(images_dir, masks_dir, train_split=0.95)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load the model
model = U2NetP_model
model.to(device)

# Load the optimzer and loss
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
# ce_loss = torch.nn.CrossEntropyLoss()
dice_loss = DiceLoss()
focal_tversky = FocalTverskyLoss()
u2_loss = U2NetCustomLoss()

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
    for class_name in NEW_CLASS_ENCODING.keys():
        epoch_test_dict[class_name]  = {"IoU": 0, "Precision": 0, "Recall": 0, "Accuracy": 0}
        epoch_train_dict[class_name] = {"IoU": 0, "Precision": 0, "Recall": 0, "Accuracy": 0}

    model.train()
    # Do the training part first
    count = 0
    for inputs, targets in train_pbar:
        inputs, targets = inputs.to(device), targets.to(torch.float32).to(device)

        predictions = model(inputs)

        d0, d1, d2, d3, d4, d5, d6 = model.forward(inputs) # Final Prediction is d1

        optimizer.zero_grad()

        # loss = dice_loss(predictions, targets)
        loss2, loss = u2_loss(d0, d1, d2, d3, d4, d5, d6, targets)

        # Take only end of the model
        pred = d0

        # Normalize prediction 
        ma = torch.max(pred)
        mi = torch.min(pred)
        predictions = (pred - mi) / (ma - mi)

        # print(predictions.shape)        
        # print(predictions)

        loss.backward()

        optimizer.step()

        # Add the loss
        epoch_train_loss += loss.item()
        count += 1

        # Calculate the metrics
        cur_metrics = calculate_metrics_by_pixel(predictions, targets)

        train_pbar.set_description(f"Epoch {num_epoch} | Training Loss : {epoch_train_loss / count:.5f}")

        # append the metrics to the epoch list
        for i in NEW_CLASS_ENCODING.keys():
            epoch_train_dict[i]["IoU"] += cur_metrics[i]["IoU"]
            epoch_train_dict[i]["Precision"] += cur_metrics[i]["Precision"]
            epoch_train_dict[i]["Recall"] += cur_metrics[i]["Recall"]

    # Normalize to the size of dataloader
    epoch_train_loss /= len(train_pbar)
    for i in NEW_CLASS_ENCODING.keys():
        epoch_train_dict[i]["IoU"] /= len(train_pbar)
        epoch_train_dict[i]["Precision"] /= len(train_pbar)
        epoch_train_dict[i]["Recall"] /= len(train_pbar)

    print("Training Metrics :")
    print_metrics(epoch_train_dict)
        
    model.eval()
    # Test part
    count = 0
    print_first = False
    for inputs, targets in test_pbar:
        inputs, targets = inputs.to(device), targets.to(torch.float32).to(device)

        predictions = model(inputs)

        # loss = dice_loss(predictions, targets)

        d0, d1, d2, d3, d4, d5, d6 = model.forward(inputs) # Final Prediction is d1

        loss2, loss = u2_loss(d0, d1, d2, d3, d4, d5, d6, targets)

        # Take only end of the model
        pred = d0

        # Normalize prediction 
        ma = torch.max(pred)
        mi = torch.min(pred)
        predictions = (pred - mi) / (ma - mi)

        epoch_test_loss += loss.item() if type(loss) != int else loss
        count += 1

        # Calculate the metrics
        cur_metrics = calculate_metrics_by_pixel(predictions, targets)

        # if not print_first:
        #     sample_tensor = predictions[0, :, :, :]

        #     print_first = True

        test_pbar.set_description(f"Epoch {num_epoch} | Testing Loss : {epoch_test_loss / count:.5f}")

        # append the metrics to the epoch list
        for i in NEW_CLASS_ENCODING.keys():
            epoch_test_dict[i]["IoU"] += cur_metrics[i]["IoU"]
            epoch_test_dict[i]["Precision"] += cur_metrics[i]["Precision"]
            epoch_test_dict[i]["Recall"] += cur_metrics[i]["Recall"]

    # Normalize to the size of dataloader
    epoch_test_loss /= len(test_pbar)
    for i in NEW_CLASS_ENCODING.keys():
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
