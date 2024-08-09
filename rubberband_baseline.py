import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import os


def rubberband_baseline(spectrum, x, save_path):
    baseline_corrected_spectrum = np.zeros_like(spectrum)
    baselines = np.zeros_like(spectrum)

    for i in range(spectrum.shape[1]):
        y = spectrum[:, i]
        hull = ConvexHull(np.column_stack((x, y)))
        k = hull.vertices
        k = k[np.argsort(x[k])]
        k = k[y[k] <= y[k[0]]]

        if x[0] != x[k[0]]:
            k = np.insert(k, 0, 0)
        if x[-1] != x[k[-1]]:
            k = np.append(k, len(x) - 1)

        baseline = interp1d(x[k], y[k], kind='linear', fill_value="extrapolate")(x)
        baselines[:, i] = baseline
        baseline_corrected_spectrum[:, i] = y - baseline

    return baseline_corrected_spectrum
