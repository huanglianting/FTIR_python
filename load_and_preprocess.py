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


def load_and_preprocess(data_files, threshold1, threshold2, order, frame_len, save_path):
    x_all, spectrum_all = [], []
    for data_file in data_files:
        x, AB = load_data(data_file)
        x, spectrum = preprocess_spectrum(x, AB, threshold1, threshold2, order, frame_len, save_path)
        x_all.append(x)
        spectrum_all.append(spectrum)

    # Check if dimensions match before concatenation
    print("Checking spectrum dimensions...")
    for i, spec in enumerate(spectrum_all):
        print(f"Spectrum {i}: shape = {spec.shape}")

    # Combine replicates by concatenating along the column axis (axis=1)
    x_combined = x_all[0]  # 所有x值相同，取第一个即可
    spectrum_combined = np.concatenate(spectrum_all, axis=1)
    return x_combined, spectrum_combined


def load_data(data):
    if 'AB' in data:
        AB = data['AB'][:, 1:]
        x = data['AB'][:, 0]
    else:
        TR = data['TR'][:, 1:]
        x = data['TR'][:, 0]
        AB = -np.log10(TR)
    return x, AB

