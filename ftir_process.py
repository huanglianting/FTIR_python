import os
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def load_and_preprocess(data_files, threshold1, threshold2, order, frame_len, save_path):
    x_all, spectrum_all = [], []
    for data in data_files:
        # TR 转 AB
        if 'AB' in data:
            AB = data['AB'][:, 1:]
            x = data['AB'][:, 0]
        else:
            TR = data['TR'][:, 1:]
            x = data['TR'][:, 0]
            AB = -np.log10(TR)
        # 只处理threshold1-threshold2之间的光谱
        valid_idx = (x >= threshold1) & (x <= threshold2)
        x = x[valid_idx]
        spectrum = AB[valid_idx, :]
        # Savitzky-Golay平滑并求二阶导数
        spectrum = savgol_filter(spectrum, frame_len, order, axis=0, deriv=2)
        x_all.append(x)
        spectrum_all.append(spectrum)
    x_combined = x_all[0]  # 所有x值相同，取第一个即可
    spectrum_combined = np.concatenate(spectrum_all, axis=1)
    return x_combined, spectrum_combined
