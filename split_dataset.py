import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_dataset(spectrum_benign, spectrum_cancer, save_path, test_size=0.3, random_state=42):
    # 合并和创建标签
    combined_spectrum = np.hstack((spectrum_benign, spectrum_cancer))
    y = np.hstack((np.zeros(spectrum_benign.shape[1]), np.ones(spectrum_cancer.shape[1]))).astype(int)

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

