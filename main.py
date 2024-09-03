import os
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.stats as stats
import matplotlib.pyplot as plt

# 确保使用正确的Matplotlib后端
matplotlib.use('TkAgg')

os.environ["OMP_NUM_THREADS"] = "1"
from load_data import load_data
from kmeans_clustering_and_plot import kmeans_clustering_and_plot
from preprocess_spectrum import preprocess_spectrum
from perform_pca_analysis import perform_pca_analysis
from analyze_and_plot_spectrum import analyze_and_plot_spectrum
from plot_spectrum_with_mean_std import plot_spectrum_with_mean_std
from plot_spectrum_with_marked_peaks import plot_spectrum_with_marked_peaks

# 设置Matplotlib使用的字体为SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 读取数据
data_1_rep1 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\1000-1.0.mat')
data_1_rep2 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\1000-2.0.mat')
data_1_rep3 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\1000-3.0.mat')

data_2_rep1 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\2000-1.0.mat')
data_2_rep2 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\2000-2.0.mat')
data_2_rep3 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\2000-3.0.mat')

data_3_rep1 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\3000-1.0.mat')
data_3_rep2 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\3000-2.0.mat')

data_4_rep1 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\4000-1.0.mat')
data_4_rep2 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\4000-2.0.mat')
data_4_rep3 = sio.loadmat('N:\\hlt\\FTIR\\data_baseline\\4000-3.0.mat')

save_path = 'N:\\hlt\\FTIR\\result\\concentration'  # 保存图片的路径

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

x_1, spectrum_1 = load_and_preprocess(
    [data_1_rep1, data_1_rep2, data_1_rep3],
    threshold1, threshold2, order, frame_len, save_path
)
x_2, spectrum_2 = load_and_preprocess(
    [data_2_rep1, data_2_rep2, data_2_rep3],
    threshold1, threshold2, order, frame_len, save_path
)
x_3, spectrum_3 = load_and_preprocess(
    [data_3_rep1, data_3_rep2],
    threshold1, threshold2, order, frame_len, save_path
)
x_4, spectrum_4 = load_and_preprocess(
    [data_4_rep1, data_4_rep2, data_4_rep3],
    threshold1, threshold2, order, frame_len, save_path
)

# 绘制出spectrum_1、spectrum_2的最终光谱图（未标注显著差异波数段）
# plot_spectrum_with_mean_std(x_1, spectrum_1, spectrum_2, save_path)
# 绘制出spectrum_1、spectrum_2的最终光谱图，把p<0.001的波数段都标注出来
# analyze_and_plot_spectrum(x_1, spectrum_1, spectrum_2, save_path, 0.001)
# 指定需要标注的波数点
peak_wavenumbers = [1030, 1080, 1239, 1313, 1407, 1451, 1550, 1577, 1656]
# 调用函数
plot_spectrum_with_marked_peaks(x_1, spectrum_1, spectrum_2, spectrum_3, spectrum_4, save_path, peak_wavenumbers)

# 主成分分析
# perform_pca_analysis(spectrum_1, spectrum_2, x_1, save_path)
# K-means聚类分析
# kmeans_clustering_and_plot(spectrum_1, spectrum_2, x_1, save_path, n_clusters=7)
