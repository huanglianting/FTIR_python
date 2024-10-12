import os
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.stats as stats
import matplotlib.pyplot as plt
from load_data import load_data
from kmeans_clustering_and_plot import kmeans_clustering_and_plot
from perform_pca_rf_analysis import perform_pca_rf_analysis
from perform_svm_analysis import perform_svm_analysis
from preprocess_spectrum import preprocess_spectrum
from perform_pca_lda_analysis import perform_pca_lda_analysis
from plot_individual_spectrum_with_marked_peaks import plot_individual_spectrum_with_marked_peaks

# 确保使用正确的Matplotlib后端
matplotlib.use('TkAgg')
os.environ["OMP_NUM_THREADS"] = "1"
# 设置Matplotlib使用的字体为SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 读取数据
benign_data_files = [f'N:\\hlt\\FTIR\\benign-441-520-1299\\benign\\benign_{i}.0.mat' for i in range(1, 13)]
cancer_441_data_files = [f'N:\\hlt\\FTIR\\benign-441-520-1299\\441\\441_{i}.0.mat' for i in range(1, 13)]
cancer_520_data_files = [f'N:\\hlt\\FTIR\\benign-441-520-1299\\520\\520_{i}.0.mat' for i in range(1, 13)]
cancer_1299_data_files = [f'N:\\hlt\\FTIR\\benign-441-520-1299\\1299\\1299_{i}.0.mat' for i in range(1, 13)]
benign_data = [sio.loadmat(file) for file in benign_data_files]
cancer_441_data = [sio.loadmat(file) for file in cancer_441_data_files]
cancer_520_data = [sio.loadmat(file) for file in cancer_520_data_files]
cancer_1299_data = [sio.loadmat(file) for file in cancer_1299_data_files]
save_path = 'N:\\hlt\\FTIR\\result\\benign-441-520-1299'  # 保存图片的路径

# Ensure the save_path directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)


# Load and preprocess data for each concentration and replicate
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


# 光谱预处理
threshold1 = 900  # 过滤掉小于threshold1的噪声
threshold2 = 1800  # 过滤掉大于threshold2的噪声
order = 2  # 多项式阶数
frame_len = 13  # 窗口长度（帧长度）

x_benign, spectrum_benign = load_and_preprocess(benign_data, threshold1, threshold2, order, frame_len, save_path)
x_441, spectrum_441 = load_and_preprocess(cancer_441_data, threshold1, threshold2, order, frame_len, save_path)
x_520, spectrum_520 = load_and_preprocess(cancer_520_data, threshold1, threshold2, order, frame_len, save_path)
x_1299, spectrum_1299 = load_and_preprocess(cancer_1299_data, threshold1, threshold2, order, frame_len, save_path)

# 指定需要标注的波数点
peak_wavenumbers = [1030, 1080, 1243, 1310, 1403, 1455, 1555, 1652]
# 分别绘制四组数据的光谱图
'''
plot_individual_spectrum_with_marked_peaks(x_benign, spectrum_benign, 'benign', 'green', save_path, peak_wavenumbers)
plot_individual_spectrum_with_marked_peaks(x_441, spectrum_441, '441', 'orange', save_path, peak_wavenumbers)
plot_individual_spectrum_with_marked_peaks(x_520, spectrum_520, '520', 'red', save_path, peak_wavenumbers)
plot_individual_spectrum_with_marked_peaks(x_1299, spectrum_1299, '1299', 'blue', save_path, peak_wavenumbers)
'''

# 进行PCA-LDA分析
perform_pca_lda_analysis(
    spectrum_benign=spectrum_benign,
    spectrum_441=spectrum_441,
    spectrum_520=spectrum_520,
    spectrum_1299=spectrum_1299,
    save_path=save_path,
    n_pca_components=20, test_size=0.3, random_state=42
)

# 进行PCA-RF分析
perform_pca_rf_analysis(
    spectrum_benign=spectrum_benign,
    spectrum_441=spectrum_441,
    spectrum_520=spectrum_520,
    spectrum_1299=spectrum_1299,
    save_path=save_path,
    n_pca_components=20,
    test_size=0.3,
    random_state=42,
    n_estimators=200,  # 例如，使用200棵树
    max_depth=10        # 例如，设置最大深度为10
)

perform_svm_analysis(
    spectrum_benign=spectrum_benign,
    spectrum_441=spectrum_441,
    spectrum_520=spectrum_520,
    spectrum_1299=spectrum_1299,
    save_path=save_path,
    test_size=0.3,                   # 可根据需要调整
    random_state=42,           # 数据划分的随机种子
    kernel='rbf',                    # 可选择 'linear', 'rbf', 'poly' 等
    C=1.0,                           # 正则化参数，默认1.0
    gamma='scale'                    # 核函数系数，默认 'scale'
)

# K-means聚类分析
# kmeans_clustering_and_plot(spectrum_1, spectrum_2, x_1, save_path, n_clusters=7)
