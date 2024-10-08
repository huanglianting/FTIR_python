import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

def perform_pca_lda_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, x_1, save_path, n_pca_components=20):
    # n_pca_components: PCA降维后的主成分数量

    # 合并所有光谱数据
    combined_spectrum = np.hstack((spectrum_benign, spectrum_441, spectrum_520, spectrum_1299))
    num_samples_benign = spectrum_benign.shape[1]
    num_samples_441 = spectrum_441.shape[1]
    num_samples_520 = spectrum_520.shape[1]
    num_samples_1299 = spectrum_1299.shape[1]

    # 创建标签
    y = np.hstack((
        np.zeros(num_samples_benign),       # 标签 0: Benign
        np.ones(num_samples_441),          # 标签 1: 441
        2 * np.ones(num_samples_520),      # 标签 2: 520
        3 * np.ones(num_samples_1299)      # 标签 3: 1299
    )).astype(int)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_spectrum.T)  # 转置为 (samples, features)

    # PCA降维
    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X_scaled)

    # LDA降维
    lda = LDA(n_components=3)  # 最多 classes - 1 维
    X_lda = lda.fit_transform(X_pca, y)

    # 可视化：使用前两个LDA组件
    plt.figure(figsize=(10, 8))
    colors = ['green', 'orange', 'red', 'blue']
    labels = ['Benign', '441', '520', '1299']
    for class_idx, color, label in zip(range(4), colors, labels):
        plt.scatter(
            X_lda[y == class_idx, 0],
            X_lda[y == class_idx, 1],
            label=label,
            color=color,
            alpha=0.7,
            edgecolors='w',
            s=50
        )

    plt.xlabel(f'LDA 1')
    plt.ylabel(f'LDA 2')
    plt.title('PCA-LDA Result')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # 保存并显示图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'PCA_LDA_result.png'), dpi=300)
    plt.show()

    # 绘制加载图（主成分）
    plt.figure(figsize=(12, 6))
    plt.plot(x_1, pca.components_[0], '-r', label='PC 1')
    plt.plot(x_1, pca.components_[1], '-b', label='PC 2')
    plt.gca().invert_xaxis()
    plt.xlabel('Wave-number (cm$^{-1}$)')
    plt.ylabel('Coefficient')
    plt.title('Loading Plot')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'PCA_contribution_result.png'), dpi=300)
    plt.show()

    # 绘制解释方差比例图
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-')
    plt.xticks(ticks=np.arange(0, n_pca_components, 2), labels=np.arange(1, n_pca_components + 1, 2))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Scree_Plot.png'), dpi=300)
    plt.show()
