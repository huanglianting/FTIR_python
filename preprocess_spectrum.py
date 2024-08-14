from SG_smooth import SG_smooth
from derivative import derivative
from filter_threshold import filter_threshold
from flatten_region import flatten_region
from vector_normalized import vector_normalized


def preprocess_spectrum(x, AB, threshold1, threshold2, order, frame_len, save_path):
    # 只处理threshold1-threshold2之间的光谱
    x, filtered_spectrum = filter_threshold(AB, x, threshold1, threshold2)
    # 矢量归一化
    normalized_spectrum = vector_normalized(filtered_spectrum, x, save_path)
    # Savitzky-Golay平滑
    smoothed_spectrum = SG_smooth(normalized_spectrum, x, order, frame_len, save_path)
    # 计算二阶导数。求了导感觉效果没那么好；另外，如果求了二阶导数，那么k-means那里，纵坐标还是AB就不太对了
    # x, derivative_spectrum = derivative(smoothed_spectrum, x, save_path)
    return x, smoothed_spectrum
