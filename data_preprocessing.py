import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from load_and_preprocess import load_and_preprocess
import torch
import torch.nn as nn


# 定义FTIR模态特异性特征提取的MLP分支网络
class FTIRMLP(nn.Module):
    def __init__(self, input_dim):
        super(FTIRMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x


# 定义MZ模态特异性特征提取的MLP分支网络
class MZMLP(nn.Module):
    def __init__(self, input_dim):
        super(MZMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x


def preprocess_data(ftir_file_path, mz_file_path1, mz_file_path2, train_folder, test_folder, save_path):
    # ===============================处理FTIR=================================================
    # 生成文件列表的通用函数
    def generate_file_lists(prefixes, num_files, ftir_file_path):
        all_file_lists = {}
        for prefix in prefixes:
            file_list = [f'{ftir_file_path}{prefix}_{i}.mat' for i in range(1, num_files + 1)]
            all_file_lists[prefix] = file_list
        return all_file_lists

    # 生成cancer和normal的文件列表
    cancer_prefixes = [f'cancer{i}' for i in range(1, 12)]  # 一共有cancer1到cancer11，总计11个样品
    normal_prefixes = [f'normal{i}' for i in range(1, 12)]
    cancer_file_lists = generate_file_lists(cancer_prefixes, 3, ftir_file_path)  # 对于FTIR，每个样品重复三次滴加到基底
    normal_file_lists = generate_file_lists(normal_prefixes, 3, ftir_file_path)
    # 加载数据
    cancer_ftir_data = {k: [sio.loadmat(f) for f in v] for k, v in cancer_file_lists.items()}
    normal_ftir_data = {k: [sio.loadmat(f) for f in v] for k, v in normal_file_lists.items()}

    # FTIR光谱预处理相关参数
    threshold1 = 900  # 过滤掉小于threshold1的噪声
    threshold2 = 1800  # 过滤掉大于threshold2的噪声
    order = 2  # 多项式阶数
    frame_len = 13  # 窗口长度（帧长度）
    # 进行预处理
    normal_ftir, cancer_ftir = {}, {}
    x_ftir, normal_ftir['normal1'] = load_and_preprocess(normal_ftir_data['normal1'], threshold1, threshold2,
                                                           order, frame_len, save_path)
    for key in list(normal_ftir_data.keys())[1:]:
        _, normal_ftir[key] = load_and_preprocess(normal_ftir_data[key], threshold1, threshold2, order, frame_len,
                                                   save_path)
    for key in cancer_ftir_data.keys():
        _, cancer_ftir[key] = load_and_preprocess(cancer_ftir_data[key], threshold1, threshold2, order, frame_len,
                                                  save_path)

    # 打印每个样品的形状
    print("x_ftir shape:", x_ftir.shape)
    for key in normal_ftir.keys():
        print(f"spectrum_normal{key[len('normal'):]} shape:",
              normal_ftir[key].shape)  # 形状均为(467, xxxx)。例如：(467, 1517) (467, 1716)
    for key in cancer_ftir.keys():
        print(f"spectrum_cancer{key[len('cancer'):]} shape:",
              cancer_ftir[key].shape)  # 形状均为(467, xxxx)。例如：(467, 1165) (467, 1260)

    # ===================================处理mz===========================================================
    df1 = pd.read_excel(mz_file_path1, header=1)  # 从第二行读取数据
    df2 = pd.read_excel(mz_file_path2, header=1)
    # 两个文件的m/z列不一致
    # 从第一个文件提取样本 (1-7)
    cancer_columns1 = [col for col in df1.columns if 'cancer' in col.lower()]  # 分别提取每个癌症和正常样本
    normal_columns1 = [col for col in df1.columns if 'normal' in col.lower()]
    cancer_mz1 = {col: df1[col].values for col in cancer_columns1}
    normal_mz1 = {col: df1[col].values for col in normal_columns1}
    # 从第二个文件提取样本 (8-11)
    cancer_columns2 = [col for col in df2.columns if 'cancer' in col.lower()]
    normal_columns2 = [col for col in df2.columns if 'normal' in col.lower()]
    cancer_mz2 = {col: df2[col].values for col in cancer_columns2}
    normal_mz2 = {col: df2[col].values for col in normal_columns2}
    # m/z列不同，保留两个不同的特征集（横坐标），形状为 (12572,)
    mz1 = df1['m/z'].values
    mz2 = df2['m/z'].values

    for col, values in cancer_mz1.items():  # 癌症样本数据，每个样本形状为(12572,)
        print(f"mz1:{col} shape:", values.shape)
    for col, values in normal_mz1.items():  # 正常样本数据，每个样本形状为(12572,)
        print(f"mz1:{col} shape:", values.shape)
    print("mz1 shape:", mz1.shape)
    for col, values in cancer_mz2.items():  # 癌症样本数据，每个样本形状为(12572,)
        print(f"mz2:{col} shape:", values.shape)
    for col, values in normal_mz2.items():  # 正常样本数据，每个样本形状为(12572,)
        print(f"mz2:{col} shape:", values.shape)
    print("mz2 shape:", mz2.shape)

    # =============================分别对每个样本的FTIR和mz数据进行处理======================================
    # 合并所有样本的特征和标签
    ftir_features_list = []
    mz_features_list = []
    labels_list = []

    # 初始化特征提取器
    ftir_extractor = FTIRMLP(input_dim=len(x_ftir))  # FTIR特征提取器
    mz1_extractor = MZMLP(input_dim=len(mz1))  # 第一批质谱特征提取器
    mz2_extractor = MZMLP(input_dim=len(mz2))  # 第二批质谱特征提取器
    # 设置为评估模式
    ftir_extractor.eval()
    mz1_extractor.eval()
    mz2_extractor.eval()

    # 处理癌症样本
    for i in range(1, 12):
        cancer_ftir_key = f'cancer{i}'
        cancer_mz_key = f'cancer_{i} [1]'
        ftir_samples = cancer_ftir[cancer_ftir_key].T  # shape：(xxxx, 467)，如 (1421, 467)
        # 使用MLP提取FTIR特征
        with torch.no_grad():
            ftir_feat = ftir_extractor(torch.tensor(ftir_samples, dtype=torch.float32)).numpy()
        if i <= 7:
            mz_sample = cancer_mz1[cancer_mz_key].reshape(1, -1)   # shape：(1, 12572)
            with torch.no_grad():
                mz_feat = mz1_extractor(torch.tensor(mz_sample, dtype=torch.float32)).numpy()
        else:
            mz_sample = cancer_mz2[cancer_mz_key].reshape(1, -1)
            with torch.no_grad():
                mz_feat = mz2_extractor(torch.tensor(mz_sample, dtype=torch.float32)).numpy()
        ftir_features_list.append(ftir_feat)
        # 复制代谢组学数据，使其样本数量和 FTIR 数据的样本数量相同
        mz_feat_repeated = np.repeat(mz_feat, ftir_samples.shape[0], axis=0)
        mz_features_list.append(mz_feat_repeated)
        labels_list.extend([1] * ftir_samples.shape[0])  # 癌症的标签标记为1

    # 处理正常样本
    for i in range(1, 12):
        normal_ftir_key = f'normal{i}'
        normal_mz_key = f'normal_{i} [1]'
        ftir_samples = normal_ftir[normal_ftir_key].T
        # 使用MLP提取FTIR特征
        with torch.no_grad():
            ftir_feat = ftir_extractor(torch.tensor(ftir_samples, dtype=torch.float32)).numpy()
        if i <= 7:
            mz_sample = normal_mz1[normal_mz_key].reshape(1, -1)   # shape：(1, 12572)
            with torch.no_grad():
                mz_feat = mz1_extractor(torch.tensor(mz_sample, dtype=torch.float32)).numpy()
        else:
            mz_sample = normal_mz2[normal_mz_key].reshape(1, -1)
            with torch.no_grad():
                mz_feat = mz2_extractor(torch.tensor(mz_sample, dtype=torch.float32)).numpy()
        ftir_features_list.append(ftir_feat)
        # 复制代谢组学数据，使其样本数量和 FTIR 数据的样本数量相同
        mz_feat_repeated = np.repeat(mz_feat, ftir_samples.shape[0], axis=0)
        mz_features_list.append(mz_feat_repeated)
        labels_list.extend([0] * ftir_samples.shape[0])  # 对照组（正常）的标签标记为0

    # 合并特征
    ftir_features = np.vstack(ftir_features_list)
    mz_features = np.vstack(mz_features_list)
    labels = np.array(labels_list)
    print("合并后的特征形状:")
    print("ftir_features shape:", ftir_features.shape)
    print("mz_features shape:", mz_features.shape)
    print("labels shape:", labels.shape)

    # ==========================================划分训练、测试集==================================================
    # 划分训练集和测试集6:4，记得打乱一下random，不要 1111100000，要0110010001这样的
    ftir_train, ftir_test, mz_train, mz_test, y_train, y_test = train_test_split(
        ftir_features, mz_features, labels, test_size=0.4, random_state=61
    )

    print("划分后的数据集形状:")
    print("ftir_train shape:", ftir_train.shape)  # (20447, 467)
    print("mz_train shape:", mz_train.shape)  # (20447, 12572)
    print("y_train shape:", y_train.shape)  # (20447,)
    print("ftir_test shape:", ftir_test.shape)  # (8763, 467)
    print("mz_test shape:", mz_test.shape)  # (8763, 12572)
    print("y_test shape:", y_test.shape)  # (8763,)

    # 保存训练集和测试集
    np.save(os.path.join(train_folder, 'ftir_train.npy'), ftir_train)
    np.save(os.path.join(train_folder, 'mz_train.npy'), mz_train)
    np.save(os.path.join(train_folder, 'y_train.npy'), y_train)
    np.save(os.path.join(test_folder, 'ftir_test.npy'), ftir_test)
    np.save(os.path.join(test_folder, 'mz_test.npy'), mz_test)
    np.save(os.path.join(test_folder, 'y_test.npy'), y_test)

    return ftir_train, mz_train, y_train, ftir_test, mz_test, y_test
