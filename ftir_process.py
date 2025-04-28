import os
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from load_and_preprocess import load_and_preprocess
from plot_spectrum_with_marked_peaks import plot_spectrum_with_marked_peaks, plot_individual_spectrum_with_marked_peaks
from split_dataset import split_dataset_ftir
from perform_individual_model_analysis import train_pca_lda_model, test_pca_lda_model, train_pca_rf_model, \
    test_pca_rf_model, train_svm_model, test_svm_model, train_cnn_model, test_cnn_model


# 生成文件列表的通用函数
def generate_file_lists(prefixes, num_files, ftir_file_path):
    all_file_lists = {}
    for prefix in prefixes:
        file_list = [f'{ftir_file_path}{prefix}_{i}.mat' for i in range(1, num_files + 1)]
        all_file_lists[prefix] = file_list
    return all_file_lists


# 确保使用正确的Matplotlib后端
matplotlib.use('TkAgg')
os.environ["OMP_NUM_THREADS"] = "1"
# 设置Matplotlib使用的字体为SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 单独处理FTIR
save_path = 'N:\\hlt\\FTIR\\result\\FNA'  # 保存图片的路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 读取数据
cancer_data_files = [f'N:\\hlt\\FTIR\\data\\cancer{i}_{j}.mat' for i in range(1, 8) for j in range(1, 4)]
normal_data_files = [f'N:\\hlt\\FTIR\\data\\normal{i}_{j}.mat' for i in range(1, 8) for j in range(1, 4)]
normal_data = [sio.loadmat(file) for file in normal_data_files]
cancer_data = [sio.loadmat(file) for file in cancer_data_files]

# 光谱预处理
threshold1 = 900  # 过滤掉小于threshold1的噪声
threshold2 = 1800  # 过滤掉大于threshold2的噪声
order = 2  # 多项式阶数
frame_len = 13  # 窗口长度（帧长度）
x_cancer, spectrum_cancer = load_and_preprocess(cancer_data, threshold1, threshold2, order, frame_len, save_path)
x_normal, spectrum_normal = load_and_preprocess(normal_data, threshold1, threshold2, order, frame_len, save_path)

# 指定需要标注的波数点
# peak_wavenumbers = [1030, 1075, 1103, 1148, 1198, 1238, 1316, 1409, 1453, 1546, 1653, 1740]
peak_wavenumbers = [987, 1032, 1078, 1105, 1151, 1198, 1257, 1313, 1361, 1414, 1466, 1589, 1637, 1740]
plot_spectrum_with_marked_peaks(x_normal, spectrum_normal, spectrum_cancer, save_path, peak_wavenumbers)
# plot_individual_spectrum_with_marked_peaks(x_normal, spectrum_normal, 'normal', 'green', save_path, peak_wavenumbers)
# plot_individual_spectrum_with_marked_peaks(x_cancer, spectrum_cancer, 'cancer', 'red', save_path, peak_wavenumbers)

"""
# 划分训练集和测试集，只需run一次，储存为npy
split_dataset_ftir(spectrum_normal, spectrum_cancer, save_path, test_size=0.3, random_state=42)

# 读取训练集和测试集
X_train_ftir = np.load(f"{save_path}/X_train_ftir.npy")
X_test_ftir = np.load(f"{save_path}/X_test_ftir.npy")
y_train_ftir = np.load(f"{save_path}/y_train_ftir.npy")
y_test_ftir = np.load(f"{save_path}/y_test_ftir.npy")

train_pca_lda_model(X_train_ftir, y_train_ftir, save_path, n_pca_components=20) # 训练模型并保存模型参数，只需run一次，后续测试会读取train保存的参数
test_pca_lda_model(X_test_ftir, y_test_ftir, save_path, show_plot=True)
train_pca_rf_model(X_train_ftir, y_train_ftir, save_path, random_state=42, n_pca_components=20, n_estimators=200, max_depth=10)
test_pca_rf_model(X_test_ftir, y_test_ftir, save_path, show_plot=True)
train_svm_model(X_train_ftir, y_train_ftir, save_path, kernel='rbf', C=1.0, gamma='scale')
test_svm_model(X_test_ftir, y_test_ftir, save_path, show_plot=False)
train_cnn_model(X_train_ftir, y_train_ftir, save_path, epochs=100, batch_size=32, lr=0.001)
test_cnn_model(X_test_ftir, y_test_ftir, save_path, batch_size=32, show_plot=False)
"""