import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap

def perform_rf_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, save_path,
                       test_size=0.3, random_state=42, n_estimators=100, max_depth=None):
    """
    使用随机森林进行四分类分析（无需PCA降维）。

    参数:
    - spectrum_benign, spectrum_441, spectrum_520, spectrum_1299: 各类光谱数据 (numpy.ndarray)
    - save_path: 结果保存路径 (str)
    - test_size: 测试集比例 (float)
    - random_state: 随机种子 (int)
    - n_estimators: 随机森林中树的数量 (int)
    - max_depth: 随机森林中树的最大深度 (int 或 None)
    """
    # 合并所有光谱数据
    combined_spectrum = np.hstack((spectrum_benign, spectrum_441, spectrum_520, spectrum_1299))
    num_samples_benign = spectrum_benign.shape[1]
    num_samples_441 = spectrum_441.shape[1]
    num_samples_520 = spectrum_520.shape[1]
    num_samples_1299 = spectrum_1299.shape[1]

    # 创建标签
    y = np.hstack((
        np.zeros(num_samples_benign),      # 标签 0: Benign
        np.ones(num_samples_441),         # 标签 1: 441
        2 * np.ones(num_samples_520),     # 标签 2: 520
        3 * np.ones(num_samples_1299)     # 标签 3: 1299
    )).astype(int)

    # 划分训练集和测试集，使用分层抽样确保类别比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        combined_spectrum.T, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 仅在训练集上拟合
    X_test_scaled = scaler.transform(X_test)        # 使用相同的标准化器转换测试集

    # 训练随机森林分类器
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train_scaled, y_train)

    # 预测
    y_pred = rf.predict(X_test_scaled)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 将混淆矩阵转换为百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    # 绘制混淆矩阵热力图
    save_confusion_matrix_heatmap(cm_percent, save_path, method_name='RF', show_plot=True)
    # 计算评价指标，保存到excel中
    classification_metrics(cm, y_test, y_pred, save_path, excel_filename='RF_metrics.xlsx')
