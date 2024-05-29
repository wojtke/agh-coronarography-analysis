import pydicom
import numpy as np

def normalize_dicom(dcm) -> np.ndarray:
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
    return normalized