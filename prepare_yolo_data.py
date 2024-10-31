#!/usr/bin/env python3
"""
YOLO only accepts their format
"""
import os
import cv2
import numpy as np
import sys

from utils.globals import CLASS_ENCODING, NEW_CLASS_ENCODING

YOLO_DATA_LOC = os.path.join(os.getcwd(), "data/yolo")
ORG_DATA_LOC = os.path.join(os.getcwd(), "data")

def adjust_class_no(class_no : int):
    if class_no in [0, 3]:
        raise Exception("Class not acceptedd")
    
    return class_no - 1 if class_no == 1 else class_no - 2

def convert_to_yolo_format(mask : np.ndarray) -> list:
    mask_list = []
    mask_width, mask_height = mask.shape[0], mask.shape[1]

    # Slice the mask, get all contours of that mask, turn into text
    for class_no in CLASS_ENCODING.keys():
        # Ignore BG, combine left and right = 1
        if class_no in [0, 3]:
            continue
        
        # Combine left and right kidney into 1
        if class_no == 2:
            class_mask = np.where(mask == 2, 255, 0) + np.where(mask == 3, 255, 0)
        else:
            class_mask = np.where((mask == class_no), 255, 0)

        # Get the contours out of mask
        contours, hierarchy = cv2.findContours(class_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            mask_dict = {}
            adjusted_contour = []
            # Adjust the contours to 0 to 1
            for idx, coord in enumerate(contour.flatten().tolist()):
                if idx%2 == 0:
                    adjusted_contour.append(str(coord / mask_width))
                else:
                    adjusted_contour.append(str(coord / mask_height))
            
            mask_dict[adjust_class_no(class_no)] = adjusted_contour
            mask_list.append(mask_dict)

    return mask_list

def write_yolo_file_format(mask_list : list) -> list:
    write_list = []
    for detection in mask_list:
        detect_class = list(detection.keys())[0]
        class_contour = detection[detect_class]

        detect_line = [str(detect_class)]
        detect_line.extend(class_contour)

        write_list.append(detect_line)

    return write_list

if not os.path.exists(YOLO_DATA_LOC):
    os.mkdir(YOLO_DATA_LOC)
    os.mkdir(os.path.join(YOLO_DATA_LOC, "train"))
    os.mkdir(os.path.join(YOLO_DATA_LOC, "test"))

# Find all the images in the data
images_list = os.listdir(os.path.join(ORG_DATA_LOC, "train_images"))
masks_list  = os.listdir(os.path.join(ORG_DATA_LOC, "train_masks"))

# Make the train test split
length_train  = int(len(images_list) * 0.9)
indexes       = [i for i in range(len(images_list))]

train_indexes = np.random.choice(indexes, length_train, replace=False)
test_indexes  = [j for j in indexes if j not in set(train_indexes)]

# Make the data from the train and test
for indexes in train_indexes:
    image_loc, mask_loc = images_list[indexes], masks_list[indexes]
    image_loc = os.path.join(ORG_DATA_LOC, "train_images", image_loc)
    mask_loc = os.path.join(ORG_DATA_LOC, "train_masks", mask_loc)

    # Read the image using cv2
    train_image = cv2.imread(image_loc)
    mask_image  = cv2.imread(mask_loc, cv2.IMREAD_GRAYSCALE)


    yolo_detections = convert_to_yolo_format(mask_image)
    yolo_file_format = write_yolo_file_format(yolo_detections)

    img_basename = os.path.basename(image_loc)
    yolo_detection_file_name = os.path.join(YOLO_DATA_LOC, "train", img_basename.rstrip(".png") + ".txt")

    # Write the new img and file format into data folder
    cv2.imwrite(os.path.join(YOLO_DATA_LOC, "train", os.path.basename(image_loc)), train_image)

    with open(yolo_detection_file_name, "w") as yolo_file:
        for det in yolo_file_format:
            line_to_write = " ".join(det)

            line_to_write += "\n"

            yolo_file.write(line_to_write)
    
    

for indexes in test_indexes:
    image_loc, mask_loc = images_list[indexes], masks_list[indexes]
    image_loc = os.path.join(ORG_DATA_LOC, "train_images", image_loc)
    mask_loc = os.path.join(ORG_DATA_LOC, "train_masks", mask_loc)

    # Read the image using cv2
    train_image = cv2.imread(image_loc)
    mask_image  = cv2.imread(mask_loc, cv2.IMREAD_GRAYSCALE)

    yolo_detections = convert_to_yolo_format(mask_image)
    yolo_file_format = write_yolo_file_format(yolo_detections)

    img_basename = os.path.basename(image_loc)
    yolo_detection_file_name = os.path.join(YOLO_DATA_LOC, "test", img_basename.rstrip(".png") + ".txt")

    # Write the new img and file format into data folder
    cv2.imwrite(os.path.join(YOLO_DATA_LOC, "test", os.path.basename(image_loc)), train_image)

    with open(yolo_detection_file_name, "w") as yolo_file:
        for det in yolo_file_format:
            line_to_write = " ".join(det)

            line_to_write += "\n"

            yolo_file.write(line_to_write)




