import numpy as np
import matplotlib.pyplot as plt
import os

def vector_normalized(spectrum, x, save_path):
    normalized_spectrum = np.zeros_like(spectrum)
    for i in range(spectrum.shape[1]):
        norm = np.linalg.norm(spectrum[:, i])
        normalized_spectrum[:, i] = spectrum[:, i] / norm
    return normalized_spectrum
