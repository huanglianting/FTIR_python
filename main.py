import os
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
from load_and_preprocess import load_and_preprocess
from torch.utils.data import DataLoader, TensorDataset
from multimodal_model import CMACF, train_model, test_model


save_path = 'N:\\hlt\\FTIR\\result\\FNA_supernatant'  # 保存图片的路径
if not os.path.exists(save_path):
    os.makedirs(save_path)


# 多模态的FTIR及mz预处理
# 读取FTIR数据
normal1_files = [f'N:\\hlt\\FTIR\\FNA预实验\\上清液\\control1_{i}.0.mat' for i in range(1, 4)]
normal2_files = [f'N:\\hlt\\FTIR\\FNA预实验\\上清液\\control2_1.0.mat',
                 f'N:\\hlt\\FTIR\\FNA预实验\\上清液\\control2_3.0.mat']
cancer1_files = [f'N:\\hlt\\FTIR\\FNA预实验\\上清液\\cancer1_{i}.0.mat' for i in range(1, 4)]
cancer2_files = [f'N:\\hlt\\FTIR\\FNA预实验\\上清液\\cancer2_1.0.mat',
                 f'N:\\hlt\\FTIR\\FNA预实验\\上清液\\cancer2_3.0.mat']
# 加载数据
normal1_data = [sio.loadmat(file) for file in normal1_files]
normal2_data = [sio.loadmat(file) for file in normal2_files]
cancer1_data = [sio.loadmat(file) for file in cancer1_files]
cancer2_data = [sio.loadmat(file) for file in cancer2_files]

# FTIR光谱预处理
threshold1 = 900  # 过滤掉小于threshold1的噪声
threshold2 = 1800  # 过滤掉大于threshold2的噪声
order = 2  # 多项式阶数
frame_len = 13  # 窗口长度（帧长度）
x_ftir, spectrum_normal1 = load_and_preprocess(normal1_data, threshold1, threshold2, order, frame_len, save_path)
_, spectrum_normal2 = load_and_preprocess(normal2_data, threshold1, threshold2, order, frame_len, save_path)
_, spectrum_cancer1 = load_and_preprocess(cancer1_data, threshold1, threshold2, order, frame_len, save_path)
_, spectrum_cancer2 = load_and_preprocess(cancer2_data, threshold1, threshold2, order, frame_len, save_path)

# 打印每个样品的形状
print("x_ftir shape:", x_ftir.shape)    # 形状为(467,)
print("spectrum_normal1 shape:", spectrum_normal1.shape)    # 形状为(467, 1716)
print("spectrum_normal2 shape:", spectrum_normal2.shape)
print("spectrum_cancer1 shape:", spectrum_cancer1.shape)    # 形状为(467, 1421)
print("spectrum_cancer2 shape:", spectrum_cancer2.shape)


# 处理mz
file_path = r'N:\hlt\FTIR\result\FNA_supernatant\compound measurements.xlsx'
df = pd.read_excel(file_path, header=1)  # 从第二行读取数据
cancer_columns = [col for col in df.columns if 'cancer' in col.lower()]  # 分别提取每个癌症和正常样本
normal_columns = [col for col in df.columns if 'normal' in col.lower()]
cancer_samples = {col: df[col].values for col in cancer_columns}
normal_samples = {col: df[col].values for col in normal_columns}
mz = df['m/z'].values  # 提取 m/z 列，形状为 (12572,)

for col, values in cancer_samples.items():  # 癌症样本数据，每个样本形状为(12572,)
    print(f"{col} shape:", values.shape)
for col, values in normal_samples.items():  # 正常样本数据，每个样本形状为(12572,)
    print(f"{col} shape:", values.shape)
print("mz shape:", mz.shape)




'''
# 以下开始构建多模态模型
# 读取训练集和测试集，务必先运行mz_process、ftir_process
X_train_ftir = np.load(f"{save_path}/X_train_ftir.npy")
X_test_ftir = np.load(f"{save_path}/X_test_ftir.npy")
y_train_ftir = np.load(f"{save_path}/y_train_ftir.npy")
y_test_ftir = np.load(f"{save_path}/y_test_ftir.npy")
X_train_mz = np.load(f"{save_path}/X_train_mz.npy")
X_test_mz = np.load(f"{save_path}/X_test_mz.npy")
y_train_mz = np.load(f"{save_path}/y_train_mz.npy")
y_test_mz = np.load(f"{save_path}/y_test_mz.npy")


print("y_train_ftir shape:", y_train_ftir.shape)
print("y_train_mz shape:", y_train_mz.shape)
print("前 5 行数据:\n", y_train_ftir[:5])
'''
