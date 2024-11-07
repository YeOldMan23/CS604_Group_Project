"""
File containing all metrics calculation
Assume torch.tensor input and output
"""
import torch
from tabulate import tabulate
from typing import Union

from .globals import NEW_CLASS_ENCODING, CLASS_ENCODING

def calculate_metrics_by_pixel(pred : torch.Tensor, actual : torch.Tensor, threshold: float = 0.7):
    metrics_dict = {}

    for class_name in CLASS_ENCODING.keys():
        metrics_dict[class_name] = {}

    # Calculate IoU by class
    for i in range(len(CLASS_ENCODING.keys())):
        pred_class = pred[:, i, :, :]
        acutal_class = actual[:, i, :, :]

        # Convert the predictionto 1 and 0
        pred_class = (pred_class >= threshold)

        pred_class = (pred_class.view(-1)).int()
        acutal_class = (acutal_class.view(-1)).int()

        # Calculate by Pixel
        TP = (pred_class & acutal_class).sum().item()
        TN = ((~pred_class) & (~acutal_class)).sum().item()
        FP = (pred_class & (~acutal_class)).sum().item()
        FN = ((~pred_class) & acutal_class).sum().item()

        # Calculate the value
        IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        # Put in dictionary
        metrics_dict[i]["IoU"] = IoU
        metrics_dict[i]["Recall"] = recall
        metrics_dict[i]["Precision"] = precision

    return metrics_dict

def print_metrics(metrics_table : dict) -> None:
    metrics_list = []
    for i in metrics_table.keys():
        class_name = CLASS_ENCODING[i]

        class_metrics = [class_name,
                         metrics_table[i]["IoU"],
                         metrics_table[i]["Recall"],
                         metrics_table[i]["Precision"]]
        
        metrics_list.append(class_metrics)

    print(tabulate(metrics_list, headers=["Class Name", "IoU", "Recall", "Precision"]))

