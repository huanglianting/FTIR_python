import os
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from load_and_preprocess import load_and_preprocess
from torch.utils.data import DataLoader, TensorDataset
from multimodal_model import CMACF, train_model, test_model

# 定义基础路径
ftir_file_path = 'N:\\hlt\\FTIR\\FNA预实验\\code_test\\'
mz_file_path = r'N:\\hlt\\FTIR\\FNA预实验\\code_test\\compound measurements.xlsx'
save_path = 'N:\\hlt\\FTIR\\FNA预实验\\code_test\\resul'  # 保存图片的路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 创建训练集和测试集保存文件夹
train_folder = os.path.join(save_path, 'train')
test_folder = os.path.join(save_path, 'test')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)


# 先对FTIR进行处理
# 生成文件列表的通用函数
def generate_file_lists(prefixes, num_files):
    all_file_lists = {}
    for prefix in prefixes:
        file_list = [f'{ftir_file_path}{prefix}_{i}.0.mat' for i in range(1, num_files + 1)]
        all_file_lists[prefix] = file_list
    return all_file_lists


# 生成cancer和control的文件列表
cancer_prefixes = [f'cancer{i}' for i in range(1, 11)]  # 一共有cancer1到cancer10，总计10个样品
control_prefixes = [f'control{i}' for i in range(1, 11)]
cancer_file_lists = generate_file_lists(cancer_prefixes, 3)  # 对于FTIR，每个样品重复三次滴加到基底
control_file_lists = generate_file_lists(control_prefixes, 3)

# 加载数据
cancer_ftir_data = {k: [sio.loadmat(f) for f in v] for k, v in cancer_file_lists.items()}
control_ftir_data = {k: [sio.loadmat(f) for f in v] for k, v in control_file_lists.items()}

# FTIR光谱预处理相关参数
threshold1 = 900  # 过滤掉小于threshold1的噪声
threshold2 = 1800  # 过滤掉大于threshold2的噪声
order = 2  # 多项式阶数
frame_len = 13  # 窗口长度（帧长度）

# 进行预处理
control_ftir, cancer_ftir = {}, {}
x_ftir, control_ftir['control1'] = load_and_preprocess(control_ftir_data['control1'], threshold1, threshold2,
                                                       order, frame_len, save_path)
for key in list(control_ftir_data.keys())[1:]:
    _, control_ftir[key] = load_and_preprocess(control_ftir_data[key], threshold1, threshold2, order, frame_len,
                                               save_path)
for key in cancer_ftir_data.keys():
    _, cancer_ftir[key] = load_and_preprocess(cancer_ftir_data[key], threshold1, threshold2, order, frame_len,
                                              save_path)

# 打印每个样品的形状
print("x_ftir shape:", x_ftir.shape)
for key in control_ftir.keys():
    print(f"spectrum_control{key[len('control'):]} shape:",
          control_ftir[key].shape)  # 形状均为(467, xxxx)。例如：(467, 1517) (467, 1716)
for key in cancer_ftir.keys():
    print(f"spectrum_cancer{key[len('cancer'):]} shape:",
          cancer_ftir[key].shape)  # 形状均为(467, xxxx)。例如：(467, 1165) (467, 1260)

# 处理mz
df = pd.read_excel(mz_file_path, header=1)  # 从第二行读取数据
cancer_columns = [col for col in df.columns if 'cancer' in col.lower()]  # 分别提取每个癌症和正常样本
control_columns = [col for col in df.columns if 'normal' in col.lower()]
cancer_mz = {col: df[col].values for col in cancer_columns}
control_mz = {col: df[col].values for col in control_columns}
mz = df['m/z'].values  # 提取 m/z 列，形状为 (12572,)

for col, values in cancer_mz.items():  # 癌症样本数据，每个样本形状为(12572,)
    print(f"{col} shape:", values.shape)
for col, values in control_mz.items():  # 正常样本数据，每个样本形状为(12572,)
    print(f"{col} shape:", values.shape)
print("mz shape:", mz.shape)

# 特征压缩
N = 128  # 设定压缩后的特征维度
pca_ftir = PCA(n_components=N)
pca_mz = PCA(n_components=N)

# 分别对每个样本的FTIR和mz数据进行处理
combined_features_list = []
labels_list = []

# 处理癌症样本
for i in range(1, 11):
    cancer_ftir_key = f'cancer{i}'
    cancer_mz_key = f'cancer_{i} [1]'
    ftir_samples = cancer_ftir[cancer_ftir_key].T  # shape：(xxxx, 467)，如 (1421, 467)
    mz_sample = cancer_mz[cancer_mz_key].reshape(1, -1)  # shape：(1, 12572)
    # 将m/z从1补齐到xxxx（FTIR的采样点数），以使得二者对齐
    mz_sample = np.repeat(mz_sample, ftir_samples.shape[0], axis=0)  # shape：(xxxx, 12572)，如 (1421, 12572)
    # 将特征都用pca压缩到128
    ftir_reduced = pca_ftir.fit_transform(ftir_samples)  # shape：(xxxx, 128)，如 (1421, 128)
    mz_reduced = pca_mz.fit_transform(mz_sample)  # shape：(xxxx, 128)，如 (1421, 128)
    # 在特征维度上拼接FTIR及m/z
    combined_features = np.hstack((ftir_reduced, mz_reduced))  # shape：(xxxx, 256)，如 (1421, 256)
    combined_features_list.append(combined_features)
    labels_list.extend([1] * ftir_samples.shape[0])  # 癌症的标签标记为1

# 处理正常样本
for i in range(1, 11):
    control_ftir_key = f'control{i}'
    control_mz_key = f'normal_{i} [1]'
    ftir_samples = control_ftir[control_ftir_key].T
    mz_sample = control_mz[control_mz_key].reshape(1, -1)
    # 将m/z从1补齐到xxxx（FTIR的采样点数），以使得二者对齐
    mz_sample = np.repeat(mz_sample, ftir_samples.shape[0], axis=0)
    # 将特征都用pca压缩到128
    ftir_reduced = pca_ftir.fit_transform(ftir_samples)
    mz_reduced = pca_mz.fit_transform(mz_sample)
    # 在特征维度上拼接FTIR及m/z
    combined_features = np.hstack((ftir_reduced, mz_reduced))
    combined_features_list.append(combined_features)
    labels_list.extend([0] * ftir_samples.shape[0])  # 对照组（正常）的标签标记为0

# 合并所有样本的特征和标签
combined_features = np.vstack(combined_features_list)  # combined_features shape after stacking: (29210, 256)
labels = np.array(labels_list)  # labels shape after conversion: (29210,) ？

# 划分训练集和测试集7:3，记得打乱一下random，不要 1111100000，要0110010001这样的
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.3, random_state=41)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# 数据增强：对训练集添加高斯噪声。因为只有10个样本，很容易过拟合。
noise_std = 0.1  # 噪声的标准差，可以根据需要调整
noise = np.random.normal(0, noise_std, X_train.shape)
X_train = X_train + noise

# 保存训练集和测试集
np.save(os.path.join(train_folder, 'X_train.npy'), X_train)
np.save(os.path.join(train_folder, 'y_train.npy'), y_train)
np.save(os.path.join(test_folder, 'X_test.npy'), X_test)
np.save(os.path.join(test_folder, 'y_test.npy'), y_test)


# 接下来把他们放进 MLP 里
# 搭建 MLP 模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(2 * N,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 看那几篇论文，学习一下常用的多模态都有哪些网络框架，我们这里也搭建一下不同的常用网络框架，然后计算几个评价指标进行比较

# softmax，例如输出：0.8（80%的概率）为是癌症，0.2为不是癌症
