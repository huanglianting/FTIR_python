import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

os.environ["OMP_NUM_THREADS"] = "1"
from load_data import load_data
from kmeans_clustering_and_plot import kmeans_clustering_and_plot
from preprocess_spectrum import preprocess_spectrum
from perform_pca_analysis import perform_pca_analysis

# 设置Matplotlib使用的字体为SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 读取数据
data_1 = sio.loadmat('N:\\hlt\\FTIR\\data\\cancer-1.0.mat')
data_2 = sio.loadmat('N:\\hlt\\FTIR\\data\\benign-1.0.mat')
save_path = 'N:\\hlt\\FTIR\\result\\cancer1_benign1'  # 保存图片的路径
x_1, AB_1 = load_data(data_1)
x_2, AB_2 = load_data(data_2)

# Ensure the save_path directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 光谱预处理
threshold1 = 900  # 过滤掉小于threshold1的噪声
threshold2 = 1800  # 过滤掉大于threshold2的噪声
order = 2  # 多项式阶数
frame_len = 13  # 窗口长度（帧长度）

x_1, original_spectrum_1, spectrum_1 = preprocess_spectrum(x_1, AB_1, threshold1, threshold2, order, frame_len,
                                                           save_path)  # cancer
x_2, original_spectrum_2, spectrum_2 = preprocess_spectrum(x_2, AB_2, threshold1, threshold2, order, frame_len,
                                                           save_path)  # benign

# 计算均值和标准差
mean_original_cancer = np.mean(original_spectrum_1, axis=1)  # 计算每一行（每一波数）的均值
std_original_cancer = np.std(original_spectrum_1, axis=1)
mean_original_benign = np.mean(original_spectrum_2, axis=1)
std_original_benign = np.std(original_spectrum_2, axis=1)

mean_cancer = np.mean(spectrum_1, axis=1)
std_cancer = np.std(spectrum_1, axis=1)
mean_benign = np.mean(spectrum_2, axis=1)
std_benign = np.std(spectrum_2, axis=1)

# 创建1x2子图
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(x_1, mean_original_cancer, label='Cancer', color='blue')
plt.fill_between(x_1, mean_original_cancer - std_original_cancer, mean_original_cancer + std_original_cancer, color='blue', alpha=0.2)
plt.plot(x_1, mean_original_benign, label='Benign', color='green')
plt.fill_between(x_1, mean_original_benign - std_original_benign, mean_original_benign + std_original_benign, color='green', alpha=0.2)
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Absorbance')
plt.title('Original Spectrum (mean ± SD)')
plt.legend()

# 创建1x2子图用于smoothed_spectrum
plt.subplot(1, 2, 2)
plt.plot(x_1, mean_cancer, label='Cancer', color='red')
plt.fill_between(x_1, mean_cancer - std_cancer, mean_cancer + std_cancer, color='red', alpha=0.2)
plt.plot(x_1, mean_benign, label='Benign', color='orange')
plt.fill_between(x_1, mean_benign - std_benign, mean_benign + std_benign, color='orange', alpha=0.2)
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Absorbance')
plt.title('Final Spectrum (mean ± SD)')
plt.legend()

plt.tight_layout()
plt.show()


# 以下是PCA和K-means聚类分析
# 确保 spectrum_1234 的大小一致
min_length = min(spectrum_1.shape[0], spectrum_2.shape[0])
spectrum_1 = spectrum_1[:min_length, :]
spectrum_2 = spectrum_2[:min_length, :]
x_1 = x_1[:min_length]

# 主成分分析
perform_pca_analysis(spectrum_1, spectrum_2, x_1, save_path)
# K-means聚类分析
kmeans_clustering_and_plot(spectrum_1, spectrum_2, x_1, save_path, n_clusters=7)
