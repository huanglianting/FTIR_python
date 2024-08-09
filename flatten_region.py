import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

def flatten_region(spectrum, x, start_wavenumber, end_wavenumber, save_path):
    region_idx = (x >= start_wavenumber) & (x <= end_wavenumber)
    flattened_spectrum = np.copy(spectrum)

    x_outside = x[~region_idx]
    spectrum_outside = spectrum[~region_idx, :]

    for i in range(spectrum.shape[1]):
        interp_func = interp1d(x_outside, spectrum_outside[:, i], kind='linear', fill_value="extrapolate")
        flattened_spectrum[region_idx, i] = interp_func(x[region_idx])

    return flattened_spectrum
