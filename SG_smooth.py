from scipy.signal import savgol_filter

def SG_smooth(spectrum, x, order, framelen, save_path):
    smoothed_spectrum = savgol_filter(spectrum, framelen, order, axis=0)
    return smoothed_spectrum
