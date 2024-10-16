from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def split_dataset(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, test_size=0.3,
                  random_state=42):
    """
    划分为训练集和测试集，并标准化数据。
    参数:
    - test_size: 测试集比例，默认0.2
    - random_state: 随机数种子，默认42
    返回:
    - X_train_scaled: 标准化后的训练集
    - X_test_scaled: 标准化后的测试集
    - y_train: 训练集标签
    - y_test: 测试集标签
    """

    # 合并所有光谱数据
    combined_spectrum = np.hstack((spectrum_benign, spectrum_441, spectrum_520, spectrum_1299))

    # 获取每个类别的样本数量
    num_samples_benign = spectrum_benign.shape[1]
    num_samples_441 = spectrum_441.shape[1]
    num_samples_520 = spectrum_520.shape[1]
    num_samples_1299 = spectrum_1299.shape[1]

    # 创建标签
    y = np.hstack((
        np.zeros(num_samples_benign),  # 标签 0: Benign
        np.ones(num_samples_441),  # 标签 1: 441
        2 * np.ones(num_samples_520),  # 标签 2: 520
        3 * np.ones(num_samples_1299)  # 标签 3: 1299
    )).astype(int)

    # 划分训练集和测试集
    X_train_spectrum, X_test_spectrum, y_train, y_test = train_test_split(
        combined_spectrum.T, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_spectrum)  # 仅在训练集上拟合
    X_test_scaled = scaler.transform(X_test_spectrum)  # 使用相同的标准化器转换测试集

    return X_train_scaled, X_test_scaled, y_train, y_test
