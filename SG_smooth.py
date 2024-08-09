import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

def SG_smooth(spectrum, x, order, framelen, save_path):
    smoothed_spectrum = savgol_filter(spectrum, framelen, order, axis=0)
    return smoothed_spectrum
