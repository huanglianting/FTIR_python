import numpy as np
from preprocess_spectrum import preprocess_spectrum


def load_and_preprocess(data_files, threshold1, threshold2, order, frame_len, save_path):
    x_all, spectrum_all = [], []
    for data_file in data_files:
        x, AB = load_data(data_file)
        x, spectrum = preprocess_spectrum(x, AB, threshold1, threshold2, order, frame_len, save_path)
        x_all.append(x)
        spectrum_all.append(spectrum)
    # Combine replicates by concatenating along the column axis (axis=1)
    x_combined = x_all[0]
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
