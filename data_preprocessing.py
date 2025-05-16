import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from load_and_preprocess import load_and_preprocess


def preprocess_data(ftir_file_path, mz_file_path, train_folder, test_folder, save_path):
    # ===============================处理FTIR=================================================
    # 生成文件列表的通用函数
    def generate_file_lists(prefixes, num_files, ftir_file_path):
        all_file_lists = {}
        for prefix in prefixes:
            file_list = [f'{ftir_file_path}{prefix}_{i}.mat' for i in range(1, num_files + 1)]
            all_file_lists[prefix] = file_list
        return all_file_lists

    # 生成cancer和normal的文件列表
    cancer_prefixes = [f'cancer{i}' for i in range(1, 8)]  # 一共有cancer1到cancer7，总计7个样品
    normal_prefixes = [f'normal{i}' for i in range(1, 8)]
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
    df = pd.read_excel(mz_file_path, header=1)  # 从第二行读取数据
    cancer_columns = [col for col in df.columns if 'cancer' in col.lower()]  # 分别提取每个癌症和正常样本
    normal_columns = [col for col in df.columns if 'normal' in col.lower()]
    cancer_mz = {col: df[col].values for col in cancer_columns}
    normal_mz = {col: df[col].values for col in normal_columns}
    mz = df['m/z'].values  # 提取 m/z 列，形状为 (12572,)

    for col, values in cancer_mz.items():  # 癌症样本数据，每个样本形状为(12572,)
        print(f"{col} shape:", values.shape)
    for col, values in normal_mz.items():  # 正常样本数据，每个样本形状为(12572,)
        print(f"{col} shape:", values.shape)
    print("mz shape:", mz.shape)

    # =============================分别对每个样本的FTIR和mz数据进行处理======================================
    # 合并所有样本的特征和标签
    ftir_features_list = []
    mz_features_list = []
    labels_list = []

    # 处理癌症样本
    for i in range(1, 8):
        cancer_ftir_key = f'cancer{i}'
        cancer_mz_key = f'cancer_{i} [1]'
        ftir_samples = cancer_ftir[cancer_ftir_key].T  # shape：(xxxx, 467)，如 (1421, 467)
        mz_sample = cancer_mz[cancer_mz_key].reshape(1, -1)  # shape：(1, 12572)
        # 复制代谢组学数据，使其样本数量和 FTIR 数据的样本数量相同
        mz_sample = np.repeat(mz_sample, ftir_samples.shape[0], axis=0)
        ftir_features_list.append(ftir_samples)
        mz_features_list.append(mz_sample)
        labels_list.extend([1] * ftir_samples.shape[0])  # 癌症的标签标记为1

    # 处理正常样本
    for i in range(1, 8):
        normal_ftir_key = f'normal{i}'
        normal_mz_key = f'normal_{i} [1]'
        ftir_samples = normal_ftir[normal_ftir_key].T
        mz_sample = normal_mz[normal_mz_key].reshape(1, -1)
        # 复制代谢组学数据，使其样本数量和 FTIR 数据的样本数量相同
        mz_sample = np.repeat(mz_sample, ftir_samples.shape[0], axis=0)
        ftir_features_list.append(ftir_samples)
        mz_features_list.append(mz_sample)
        labels_list.extend([0] * ftir_samples.shape[0])  # 对照组（正常）的标签标记为0

    ftir_features = np.vstack(ftir_features_list)
    mz_features = np.vstack(mz_features_list)
    labels = np.array(labels_list)

    # ==========================================划分训练、测试集==================================================
    # 划分训练集和测试集6:4，记得打乱一下random，不要 1111100000，要0110010001这样的
    ftir_train, ftir_test, mz_train, mz_test, y_train, y_test = train_test_split(
        ftir_features, mz_features, labels, test_size=0.3, random_state=57
    )

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
