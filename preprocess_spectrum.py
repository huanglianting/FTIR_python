import numpy as np
from scipy.signal import savgol_filter


def preprocess_spectrum(x, AB, threshold1, threshold2, order, frame_len, save_path):
    # 只处理threshold1-threshold2之间的光谱
    x, filtered_spectrum = filter_threshold(AB, x, threshold1, threshold2)
    # 矢量归一化
    normalized_spectrum = vector_normalized(filtered_spectrum)
    # Savitzky-Golay平滑
    smoothed_spectrum = savgol_filter(normalized_spectrum, frame_len, order, axis=0)
    # 计算二阶导数。求了导感觉效果没那么好；另外，如果求了二阶导数，那么k-means那里，纵坐标还是AB就不太对了
    # x, derivative_spectrum = derivative(smoothed_spectrum, x)
    return x, smoothed_spectrum


def filter_threshold(spectrum, x, threshold1, threshold2):
    valid_idx = (x >= threshold1) & (x <= threshold2)
    filtered_x = x[valid_idx]
    filtered_spectrum = spectrum[valid_idx, :]
    return filtered_x, filtered_spectrum


def vector_normalized(spectrum):
    normalized_spectrum = np.zeros_like(spectrum)
    for i in range(spectrum.shape[1]):
        norm = np.linalg.norm(spectrum[:, i])
        normalized_spectrum[:, i] = spectrum[:, i] / norm
    return normalized_spectrum


def derivative(spectrum, x):
    # derivative_spectrum = np.zeros((spectrum.shape[0] - 1, spectrum.shape[1]))
    derivative_spectrum = np.zeros((spectrum.shape[0] - 2, spectrum.shape[1]))
    for i in range(spectrum.shape[1]):
        # derivative_spectrum[:, i] = np.diff(spectrum[:, i])  # 一阶求导
        derivative_spectrum[:, i] = np.diff(spectrum[:, i], n=2)  # 二阶求导
    # 调整波数数组，使其与导数光谱对齐
    # x = x[:-1]
    x = x[:-2]
    return x, derivative_spectrum