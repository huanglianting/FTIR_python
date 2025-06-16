import os
import numpy as np
import pandas as pd
import scipy.io as sio
from load_and_preprocess import load_and_preprocess
import torch
import torch.nn as nn


# 定义FTIR模态特征提取的分支
class FTIRFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FTIRFeatureExtractor, self).__init__()
        # 三层全连接网络
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        # 一维卷积神经网络
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # 三层全连接
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))  # [样本数, 32]
        x = self.dropout(x)
        # 调整维度以适应卷积层输入
        x = x.unsqueeze(1)  # [样本数, 1, 32]
        x = self.relu(self.conv1d(x))  # [样本数, out_channels, 32]
        # 展平卷积输出，重新变成二维
        x = x.view(x.size(0), -1)  # [sample, 32 * out_channels]
        return x


# 定义MZ模态特征提取的分支
class MZFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(MZFeatureExtractor, self).__init__()
        # 三层全连接网络
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        # 一维卷积神经网络
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # 三层全连接
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))  # [样本数, 32]
        x = self.dropout(x)
        # 调整维度以适应卷积层输入
        x = x.unsqueeze(1)  # [样本数, 1, 32]
        x = self.relu(self.conv1d(x))  # [样本数, out_channels, 32]
        # 展平卷积输出，重新变成二维
        x = x.view(x.size(0), -1)  # [sample, 32 * out_channels]
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

    # =============================按患者i处理FTIR和mz数据并划分set======================================
    # 初始化特征提取器
    ftir_extractor = FTIRFeatureExtractor(input_dim=len(x_ftir))
    mz1_extractor = MZFeatureExtractor(input_dim=len(mz1))
    mz2_extractor = MZFeatureExtractor(input_dim=len(mz2))
    # 设置为评估模式
    ftir_extractor.eval()
    mz1_extractor.eval()
    mz2_extractor.eval()

    # 按患者初始化（ 训练(+验证) / 测试 ）列表
    train_ftir, train_mz, train_labels, train_patients = [], [], [], []
    test_ftir, test_mz, test_labels, test_patients = [], [], [], []
    # 随机打乱患者顺序（1-11）
    np.random.seed(4)
    patients = np.arange(1, 12)  # 患者i=1到11
    np.random.shuffle(patients)
    # 划分比例：8训练(+验证)，3测试
    train_patients_list = patients[:8]
    test_patients_list = patients[3:]

    # 遍历每个患者，处理并分配到对应集合
    for i in patients:  # 按打乱后的顺序处理患者
        # 处理癌症样本
        cancer_ftir_key = f'cancer{i}'
        cancer_mz_key = f'cancer_{i} [1]'
        ftir_cancer = cancer_ftir[cancer_ftir_key].T  # shape：(xxxx, 467)，如 (1421, 467)
        # 使用FeatureExtractor网络提取特征
        with torch.no_grad():
            ftir_cancer_feat = ftir_extractor(torch.tensor(ftir_cancer, dtype=torch.float32)).numpy()
        if i <= 7:
            mz_cancer = cancer_mz1[cancer_mz_key].reshape(1, -1)   # shape：(1, 12572)
            with torch.no_grad():
                mz_cancer_feat = mz1_extractor(torch.tensor(mz_cancer, dtype=torch.float32)).numpy()
        else:
            mz_cancer = cancer_mz2[cancer_mz_key].reshape(1, -1)
            with torch.no_grad():
                mz_cancer_feat = mz2_extractor(torch.tensor(mz_cancer, dtype=torch.float32)).numpy()
        # 复制代谢组学数据，使其样本数量和 FTIR 数据的样本数量相同
        mz_cancer_repeated = np.repeat(mz_cancer_feat, ftir_cancer.shape[0], axis=0)
        labels_cancer = np.ones(ftir_cancer.shape[0], dtype=int)  # 癌症的标签标记为1

        # 处理正常样本
        normal_ftir_key = f'normal{i}'
        normal_mz_key = f'normal_{i} [1]'
        ftir_normal = normal_ftir[normal_ftir_key].T
        # 使用FeatureExtractor网络提取特征
        with torch.no_grad():
            ftir_normal_feat = ftir_extractor(torch.tensor(ftir_normal, dtype=torch.float32)).numpy()
        if i <= 7:
            mz_normal = normal_mz1[normal_mz_key].reshape(1, -1)   # shape：(1, 12572)
            with torch.no_grad():
                mz_normal_feat = mz1_extractor(torch.tensor(mz_normal, dtype=torch.float32)).numpy()
        else:
            mz_normal = normal_mz2[normal_mz_key].reshape(1, -1)
            with torch.no_grad():
                mz_normal_feat = mz2_extractor(torch.tensor(mz_normal, dtype=torch.float32)).numpy()
        # 复制代谢组学数据，使其样本数量和 FTIR 数据的样本数量相同
        mz_normal_repeated = np.repeat(mz_normal_feat, ftir_normal.shape[0], axis=0)
        labels_normal = np.zeros(ftir_normal.shape[0], dtype=int)  # 对照组（正常）的标签标记为0

        # 合并患者i的所有样本
        ftir_all = np.vstack([ftir_cancer_feat, ftir_normal_feat])
        mz_all = np.vstack([mz_cancer_repeated, mz_normal_repeated])
        labels_all = np.hstack([labels_cancer, labels_normal])
        patient_ids = np.full_like(labels_all, i)  # 为每个样本添加患者ID

        # 打乱单个患者内的癌症/正常顺序，不然都是010101
        # 生成打乱索引
        np.random.seed(i)  # 用i作种子，固定患者内的随机种子，确保特征和标签同步打乱
        indices = np.arange(ftir_all.shape[0])
        np.random.shuffle(indices)
        # 按索引打乱数据
        ftir_shuffled = ftir_all[indices]
        mz_shuffled = mz_all[indices]
        labels_shuffled = labels_all[indices]
        patient_ids_shuffled = np.full_like(labels_shuffled, i)
        # 根据患者i所在训练/测试集分配打乱后的数据
        if i in train_patients_list:
            train_ftir.append(ftir_shuffled)
            train_mz.append(mz_shuffled)
            train_labels.append(labels_shuffled)
            train_patients.extend(patient_ids_shuffled)
        else:
            test_ftir.append(ftir_shuffled)
            test_mz.append(mz_shuffled)
            test_labels.append(labels_shuffled)
            test_patients.extend(patient_ids_shuffled)

    # 合并训练集数据
    ftir_train = np.vstack(train_ftir) if train_ftir else np.array([])
    mz_train = np.vstack(train_mz) if train_mz else np.array([])
    y_train = np.hstack(train_labels) if train_labels else np.array([])
    patient_indices_train = np.array(train_patients)
    # 合并测试集数据
    ftir_test = np.vstack(test_ftir) if test_ftir else np.array([])
    mz_test = np.vstack(test_mz) if test_mz else np.array([])
    y_test = np.hstack(test_labels) if test_labels else np.array([])
    patient_indices_test = np.array(test_patients)

    # =============================保存数据==================================================
    np.save(os.path.join(train_folder, 'ftir_train.npy'), ftir_train)
    np.save(os.path.join(train_folder, 'mz_train.npy'), mz_train)
    np.save(os.path.join(train_folder, 'y_train.npy'), y_train)
    np.save(os.path.join(train_folder, 'patient_indices_train.npy'), patient_indices_train)
    np.save(os.path.join(test_folder, 'ftir_test.npy'), ftir_test)
    np.save(os.path.join(test_folder, 'mz_test.npy'), mz_test)
    np.save(os.path.join(test_folder, 'y_test.npy'), y_test)
    np.save(os.path.join(test_folder, 'patient_indices_test.npy'), patient_indices_test)

    return ftir_train, mz_train, y_train, patient_indices_train, ftir_test, mz_test, y_test, patient_indices_test
