import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap


def perform_pca_lda_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, x_1, save_path,
                             n_pca_components=20):
    # n_pca_components: PCA降维后的主成分数量
    # 合并所有光谱数据
    combined_spectrum = np.hstack((spectrum_benign, spectrum_441, spectrum_520, spectrum_1299))
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
        combined_spectrum.T, y, test_size=0.3, random_state=42, stratify=y
    )

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_spectrum)  # 仅在训练集上拟合
    X_test_scaled = scaler.transform(X_test_spectrum)  # 使用相同的标准化器转换测试集

    # PCA降维
    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)  # 在训练集上拟合PCA
    X_test_pca = pca.transform(X_test_scaled)  # 使用相同的PCA模型转换测试集

    # 训练LDA分类器
    lda = LDA(n_components=3)  # 最多 classes - 1 维
    lda.fit(X_train_pca, y_train)

    # 预测
    y_pred = lda.predict(X_test_pca)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵作图出来，更直观，在图片上标注出数值
    # 将混淆矩阵转换为百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    # 绘制混淆矩阵热力图
    save_confusion_matrix_heatmap(cm_percent, save_path, image_filename='PCA-LDA_confusion_matrix_heatmap.png')
    # 计算评价指标，保存到excel中
    classification_metrics(cm, y_test, y_pred, save_path, excel_filename='PCA-LDA_metrics.xlsx')


"""
    # 可视化：使用前两个LDA组件
    # 转换测试集数据到LDA空间
    X_test_lda = lda.transform(X_test_pca)

    plt.figure(figsize=(10, 8))
    colors = ['green', 'orange', 'red', 'blue']
    labels = ['Benign', '441', '520', '1299']
    for class_idx, color, label in zip(range(4), colors, labels):
        plt.scatter(
            X_test_lda[y_test == class_idx, 0],     # LDA第一个组件
            X_test_lda[y_test == class_idx, 1],     # LDA第二个组件
            label=label,
            color=color,
            alpha=0.7,
            edgecolors='w',
            s=50
        )
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.title('PCA-LDA Clustering Results (Testing Set)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # 保存并显示图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'PCA-LDA_result_testing.png'), dpi=300)
    plt.show()
"""
