import pydicom
import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm

import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def normalize_to_uint8(dcm) -> np.ndarray:
    data = dcm.pixel_array.astype("float32")
    window_center = dcm.WindowCenter
    window_width = dcm.WindowWidth
    
    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = window_center[0]
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = window_width[0]
    
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    
    normalized = np.clip((data - lower) / (upper - lower), 0, 1)
    normalized = (normalized * 255).astype('uint8')  # Scale to 0-255 and convert to uint8
    return normalized


def read_images(dir="data/dicoms/", ids=None, use_cache=True) -> np.ndarray:
    if use_cache and os.path.exists("cached_images.pkl") and ids is None:
        with open("cached_images.pkl", 'rb') as f:
            images = pickle.load(f)
        print("Loaded images from cache.")
        return images

    if use_cache and os.path.exists("cached_images_test.pkl") and ids == "test":
        with open("cached_images_test.pkl", 'rb') as f:
            images = pickle.load(f)
        print("Loaded test images from cache.")
        return images
        
    dcm_files = os.listdir(dir)
    dcm_files = [f for f in dcm_files if f.endswith(".dcm")]
    test_split = False
    if ids is not None:
        if ids=="train":
            ids = set(pd.read_csv("testimgs.csv").ID)
            dcm_files = [x for x in dcm_files if x.replace(".dcm", "") not in ids]
        elif ids=="test":
            test_split = True
            ids = set(pd.read_csv("testimgs.csv").ID)
            dcm_files = [x for x in dcm_files if x.replace(".dcm", "") in ids]
        elif type(ids)==list:
            dcm_files = [x for x in dcm_files if x.replace(".dcm", "") in ids]
        else:
            raise ValueError()
        
            
    images = {}
    for filename in tqdm(dcm_files, desc="Loading imgs"):
        dcm = pydicom.dcmread(dir + filename)
        px_array = dcm.pixel_array
        if px_array.dtype != np.uint8:
            px_array = normalize_to_uint8(dcm)
        if px_array.ndim == 2:
            px_array = px_array[None]

        images[filename.replace(".dcm", "")] = px_array

    if use_cache and ids is None:
        with open("cached_images.pkl", 'wb') as f:
            pickle.dump(images, f)
        print("Saved images to cache.")

    if use_cache and test_split:
        with open("cached_images_test.pkl", 'wb') as f:
            pickle.dump(images, f)
        print("Saved images to cache.")
        
    return images


def plot_wandb_training_curves(project_name, metric=None):
    api = wandb.Api()
    last_run = api.runs(f"{project_name}")[0]
    history = last_run.history()

    data = history[["epoch", "train_loss", "test_loss"]]

    if metric is not None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    else:
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs = [axs]

    loss_data = data.melt(id_vars=["epoch"], value_vars=["train_loss", "test_loss"], 
                          var_name="Loss Type", value_name="Loss")
    sns.lineplot(ax=axs[0], data=loss_data, x="epoch", y="Loss", hue="Loss Type")
    axs[0].set_ylabel("Loss", fontsize=14)
    axs[0].legend(title="Loss")
    axs[0].grid(True)
    
    if metric is not None:
        data = history[["epoch", f"train_{metric}", f"test_{metric}"]]
        metric_data = data.melt(id_vars=["epoch"], value_vars=[f"train_{metric}", f"test_{metric}"], 
                        var_name="Metric Type", value_name=metric.capitalize())
        
        sns.lineplot(ax=axs[1], data=metric_data, x="epoch", y=metric.capitalize(), hue="Metric Type")
        axs[1].set_xlabel("Epoch", fontsize=14)
        axs[1].set_ylabel(metric.capitalize(), fontsize=14)
        axs[1].legend(title="Metric")
        axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def first_conv_to_1_channel(model):
    layer_name, layer = list(model.named_children())[0]
    if not isinstance(layer, torch.nn.Conv2d):
        raise ValueError("First layer of the model is not torch.nn.Conv2d")
    
    average_weights = layer.weight.sum(dim=1, keepdim=True)
    
    new_conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=layer.bias is None
    )
    with torch.no_grad():
        new_conv.weight.copy_(average_weights)
        
        # if layer.bias is not None:
        #     new_conv.bias.copy_(layer.bias)
    
    setattr(model, layer_name, new_conv)
    return model
