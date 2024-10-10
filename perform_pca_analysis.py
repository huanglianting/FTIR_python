import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import chi2
import os


def perform_pca_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, x_1, save_path,
                         confidence_level=0.95):
    """
    对四组光谱数据进行PCA分析，并绘制PCA结果、加载图和解释方差比例图。

    参数:
    - spectrum_benign: benign组的光谱数据 (features x samples)
    - spectrum_441: 441组的光谱数据 (features x samples)
    - spectrum_520: 520组的光谱数据 (features x samples)
    - spectrum_1299: 1299组的光谱数据 (features x samples)
    - x_1: 波数轴数据 (features,)
    - save_path: 保存图像的路径
    - confidence_level: 置信水平（当前已移除置信椭圆，可以忽略此参数）
    """

    # 将四组光谱数据进行组合
    combined_spectrum = np.hstack((spectrum_benign, spectrum_441, spectrum_520, spectrum_1299))

    # 执行 PCA
    pca = PCA(n_components=20)  # 选择 20 个主成分
    score = pca.fit_transform(combined_spectrum.T)
    explained = pca.explained_variance_ratio_

    # 计算每组数据的样本数量
    num_samples_benign = spectrum_benign.shape[1]
    num_samples_441 = spectrum_441.shape[1]
    num_samples_520 = spectrum_520.shape[1]
    num_samples_1299 = spectrum_1299.shape[1]

    # 为每组数据打标签
    group_labels = np.hstack((
        np.full(num_samples_benign, 'Benign'),
        np.full(num_samples_441, '441'),
        np.full(num_samples_520, '520'),
        np.full(num_samples_1299, '1299')
    ))

    # 为不同组的数据分配颜色
    color_mapping = {
        'Benign': 'green',
        '441': 'orange',
        '520': 'red',
        '1299': 'blue'
    }

    colors = [color_mapping[label] for label in group_labels]
    labels = ['Benign', '441', '520', '1299']
    unique_labels = np.unique(group_labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制每个类别的数据点
    for label in unique_labels:
        idx = group_labels == label
        ax.scatter(score[idx, 0], score[idx, 1], label=label, color=color_mapping[label], alpha=0.7, edgecolors='w',
                   s=50)

    # 获取主成分1和主成分2的方差贡献
    pc1_variance = explained[0] * 100
    pc2_variance = explained[1] * 100

    # 设置坐标轴和标题
    plt.xlabel(f'PC 1 ({pc1_variance:.2f}%)')
    plt.ylabel(f'PC 2 ({pc2_variance:.2f}%)')
    plt.title('PCA Result')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # 反转x轴方向（可选，根据需求）
    # ax.invert_xaxis()

    # 保存并显示图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'PCA_result.png'), dpi=300)
    plt.show()

    # 计算并绘制波数与主成分贡献率的关系图
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
    plt.plot(np.cumsum(explained), marker='o', linestyle='-')
    plt.xticks(ticks=np.arange(0, 20, 2), labels=np.arange(1, 21, 2))
    plt.yticks(np.arange(0, 1.1, 0.1))  # 设置 y 轴刻度
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Scree_Plot.png'), dpi=300)
    plt.show()
