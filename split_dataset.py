import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_dataset_mz(mz_benign, mz_cancer, save_path, test_size=0.3, random_state=42):
    """
    针对质谱数据的划分函数。
    :param mz_benign: 良性样本的质谱数据，形状为 (n_samples_benign, n_features)
    :param mz_cancer: 癌症样本的质谱数据，形状为 (n_samples_cancer, n_features)
    :param save_path: 数据保存路径
    :param test_size: 测试集比例
    :param random_state: 随机种子
    """
    # 合并和创建标签
    X = np.vstack((mz_benign, mz_cancer))  # 按行合并
    y = np.hstack((np.zeros(len(mz_benign)), np.ones(len(mz_cancer)))).astype(int)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 仅在训练集上拟合
    X_test_scaled = scaler.transform(X_test)  # 使用相同的标准化器转换测试集

    # 保存数据
    np.save(f"{save_path}/X_train_mz.npy", X_train_scaled)
    np.save(f"{save_path}/X_test_mz.npy", X_test_scaled)
    np.save(f"{save_path}/y_train_mz.npy", y_train)
    np.save(f"{save_path}/y_test_mz.npy", y_test)

    print("质谱数据集划分完成并保存。")


def split_dataset_ftir(spectrum_benign, spectrum_cancer, save_path, test_size=0.3, random_state=42):
    """
    针对红外光谱数据的划分函数。
    :param spectrum_benign: 良性样本的红外光谱数据，形状为 (n_features, n_samples_benign)
    :param spectrum_cancer: 癌症样本的红外光谱数据，形状为 (n_features, n_samples_cancer)
    :param save_path: 数据保存路径
    :param test_size: 测试集比例
    :param random_state: 随机种子
    """
    # 合并和创建标签
    combined_spectrum = np.hstack((spectrum_benign, spectrum_cancer))  # 按列合并
    y = np.hstack((np.zeros(spectrum_benign.shape[1]), np.ones(spectrum_cancer.shape[1]))).astype(int)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(combined_spectrum.T, y, test_size=test_size, random_state=random_state, stratify=y)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 仅在训练集上拟合
    X_test_scaled = scaler.transform(X_test)  # 使用相同的标准化器转换测试集

    # 保存数据
    np.save(f"{save_path}/X_train_ftir.npy", X_train_scaled)
    np.save(f"{save_path}/X_test_ftir.npy", X_test_scaled)
    np.save(f"{save_path}/y_train_ftir.npy", y_train)
    np.save(f"{save_path}/y_test_ftir.npy", y_test)

    print("红外光谱数据集划分完成并保存。")


# 古早针对四种亚型的FTIR光谱处理
def split_dataset_4(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, save_path, test_size=0.3, random_state=42):
    # 合并和创建标签
    combined_spectrum = np.hstack((spectrum_benign, spectrum_441, spectrum_520, spectrum_1299))
    y = np.hstack((np.zeros(spectrum_benign.shape[1]), np.ones(spectrum_441.shape[1]),
                   2 * np.ones(spectrum_520.shape[1]), 3 * np.ones(spectrum_1299.shape[1]))).astype(int)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(combined_spectrum.T, y, test_size=test_size, random_state=random_state, stratify=y)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 仅在训练集上拟合
    X_test_scaled = scaler.transform(X_test)  # 使用相同的标准化器转换测试集

    # 保存数据
    np.save(f"{save_path}/X_train_scaled.npy", X_train_scaled)
    np.save(f"{save_path}/X_test_scaled.npy", X_test_scaled)
    np.save(f"{save_path}/y_train.npy", y_train)
    np.save(f"{save_path}/y_test.npy", y_test)

