import os
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from kmeans_clustering_and_plot import kmeans_clustering_and_plot
from load_and_preprocess import load_and_preprocess
from plot_individual_spectrum_with_marked_peaks import plot_individual_spectrum_with_marked_peaks
from plot_spectrum_with_marked_peaks import plot_spectrum_with_marked_peaks
from split_dataset import split_dataset
from perform_pca_lda_analysis import train_pca_lda_model, test_pca_lda_model
from perform_pca_rf_analysis import train_pca_rf_model, test_pca_rf_model
from perform_svm_analysis import train_svm_model, test_svm_model
from perform_cnn_analysis import train_cnn_model, test_cnn_model

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
if not os.path.exists(save_path):
    os.makedirs(save_path)

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
"""
plot_spectrum_with_marked_peaks(x_benign, spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, save_path, peak_wavenumbers)
# 分别绘制四组数据的光谱图
plot_individual_spectrum_with_marked_peaks(x_benign, spectrum_benign, 'benign', 'green', save_path, peak_wavenumbers)
plot_individual_spectrum_with_marked_peaks(x_441, spectrum_441, '441', 'orange', save_path, peak_wavenumbers)
plot_individual_spectrum_with_marked_peaks(x_520, spectrum_520, '520', 'red', save_path, peak_wavenumbers)
plot_individual_spectrum_with_marked_peaks(x_1299, spectrum_1299, '1299', 'blue', save_path, peak_wavenumbers)
"""

# 划分训练集和测试集，只需run一次，就保存到save_path里储存为npy了
# split_dataset(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, save_path, test_size=0.3, random_state=42)
# 读取训练集和测试集
X_train_scaled = np.load(f"{save_path}/X_train_scaled.npy")
X_test_scaled = np.load(f"{save_path}/X_test_scaled.npy")
y_train = np.load(f"{save_path}/y_train.npy")
y_test = np.load(f"{save_path}/y_test.npy")

"""
# 训练 PCA-LDA 模型并保存模型参数，只需run一次，后续测试会自己读取train保存的参数
# train_pca_lda_model(X_train_scaled, y_train, save_path, n_pca_components=20)
# 测试 PCA-LDA 模型
test_pca_lda_model(X_test_scaled, y_test, save_path, show_plot=True)

# train_pca_rf_model(X_train_scaled, y_train, save_path, random_state=42, n_pca_components=20, n_estimators=200, max_depth=10)
test_pca_rf_model(X_test_scaled, y_test, save_path, show_plot=True)

# train_svm_model(X_train_scaled, y_train, save_path, kernel='rbf', C=1.0, gamma='scale')
test_svm_model(X_test_scaled, y_test, save_path, show_plot=False)

# train_cnn_model(X_train_scaled, y_train, save_path, epochs=100, batch_size=32, lr=0.001)
test_cnn_model(X_test_scaled, y_test, save_path, batch_size=32, show_plot=False)
"""

# K-means聚类分析
# kmeans_clustering_and_plot(spectrum_1, spectrum_2, x_1, save_path, n_clusters=7)
