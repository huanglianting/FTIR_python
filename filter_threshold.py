import numpy as np

def filter_threshold(spectrum, x, threshold1, threshold2):
    valid_idx = (x >= threshold1) & (x <= threshold2)
    filtered_x = x[valid_idx]
    filtered_spectrum = spectrum[valid_idx, :]
    return filtered_x, filtered_spectrum

