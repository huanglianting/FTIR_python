import numpy as np
import matplotlib.pyplot as plt


def derivative(spectrum, x_sample, save_path):
    # derivative_spectrum = np.zeros((spectrum.shape[0] - 1, spectrum.shape[1]))
    derivative_spectrum = np.zeros((spectrum.shape[0] - 2, spectrum.shape[1]))
    for i in range(spectrum.shape[1]):
        # derivative_spectrum[:, i] = np.diff(spectrum[:, i])  # 一阶求导
        derivative_spectrum[:, i] = np.diff(spectrum[:, i], n=2)  # 二阶求导

    # 调整波数数组，使其与导数光谱对齐
    # x_sample = x_sample[:-1]
    x_sample = x_sample[:-2]

    return x_sample, derivative_spectrum