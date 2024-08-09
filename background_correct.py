import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def background_correct(Sc, x_sample, y_bg):
    background_corrected_spectrum = np.zeros((Sc.shape[0], Sc.shape[1] - 1))
    for i in range(1, Sc.shape[1]):
        y_sample = Sc[:, i]
        background_corrected_spectrum[:, i - 1] = -np.log10(
            (y_sample + np.finfo(float).eps) / (y_bg + np.finfo(float).eps))

    return background_corrected_spectrum
