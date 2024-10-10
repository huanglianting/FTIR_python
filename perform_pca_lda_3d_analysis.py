import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图
from sklearn.preprocessing import StandardScaler
import os


def perform_pca_lda_3d_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, x_1, save_path,
                             n_pca_components=20):
    """
    对四组光谱数据进行PCA-LDA分析，并绘制3D结果。
    """

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

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_spectrum.T)  # 转置为 (samples, features)

    # PCA降维
    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    # LDA降维
    lda = LDA(n_components=3)  # 最多 classes - 1 维，四类数据最多为3维
    X_lda = lda.fit_transform(X_pca, y)

    # 可视化：使用LDA的三个维度
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['green', 'orange', 'red', 'blue']
    labels = ['Benign', '441', '520', '1299']
    for class_idx, color, label in zip(range(4), colors, labels):
        ax.scatter(
            X_lda[y == class_idx, 0],
            X_lda[y == class_idx, 1],
            X_lda[y == class_idx, 2],
            label=label,
            color=color,
            alpha=0.7,
            edgecolors='w',
            s=50
        )

    ax.set_xlabel('LDA 1')
    ax.set_ylabel('LDA 2')
    ax.set_zlabel('LDA 3')
    ax.set_title('PCA-LDA 3D Result')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # 保存并显示图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'PCA_LDA_3D_result.png'), dpi=300)
    plt.show()

