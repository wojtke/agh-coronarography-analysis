import pydicom
import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm


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
        
    dcm_files = os.listdir(dir)
    dcm_files = [f for f in dcm_files if f.endswith(".dcm")]
    if ids is not None:
        if ids=="train":
            ids = set(pd.read_csv("testimgs.csv").ID)
            dcm_files = [x for x in dcm_files if x.replace(".dcm", "") not in ids]
        elif ids=="test":
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
        
    return images