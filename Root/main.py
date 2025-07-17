import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from src.model.model import Net , RNN_Net
from src.utils.train import train
from src.utils.test import test
import matplotlib.pyplot as plt
from src.utils.misc_utils import build_classification_dataset, resize_images

import os
import re




def main():
    
    # # Replace this with your actual path
    # rename_image_files_with_underscores("/data2/users/koushani/chbmit/Eye_ML/RV_images_final")

    # patient_folder = "/data2/users/koushani/chbmit/Eye_ML/RV_images_final/CME/ACB_OU"
    # save_folder = "/data2/users/koushani/chbmit/Root/plots"
    # show_eye_pair(patient_folder, save_dir=save_folder)
    
    df_resized = build_classification_dataset("/data2/users/koushani/chbmit/Eye_ML/RV_images_resized")
    resize_images(df_resized, output_root="/data2/users/koushani/chbmit/Eye_ML/RV_images_resized", target_size=(320, 320))`
    print(df_resized.head())
    df_resized.to_csv("faa_image_classification_dataset_resized.csv", index=False)
    


if __name__ == "__main__":
    main()