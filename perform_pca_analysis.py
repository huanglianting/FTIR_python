import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import os

def perform_pca_analysis(spectrum_1, spectrum_2, x_1, save_path, confidence_level=0.95):
    # 主成分分析 PCA
    combined_spectrum = np.hstack((spectrum_1, spectrum_2))
    pca = PCA(n_components=20)  # 假设最多选择20个主成分
    score = pca.fit_transform(combined_spectrum.T)
    explained = pca.explained_variance_ratio_

    num_samples_1 = spectrum_1.shape[1]
    num_samples_2 = spectrum_2.shape[1]
    group = np.hstack((np.ones(num_samples_1), 2 * np.ones(num_samples_2)))

    colors = ['r', 'b']
    markers = ['o', 'x']
    labels = ['Cancer', 'Benign']
    fig, ax = plt.subplots()
    for i, (c, m, label) in enumerate(zip(colors, markers, labels)):
        # 获取每组数据
        group_data = combined_spectrum[:, group == (i + 1)]
        # 计算协方差矩阵和椭圆参数
        transformed_data = pca.transform(group_data.T)
        covariance_matrix = np.cov(transformed_data.T)
        center = np.mean(transformed_data, axis=0)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # 确保 eigenvectors 是浮点数数组
        eigenvectors = eigenvectors.astype(float)

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))  # 使用正确的参数传递方式

        width = 2 * np.sqrt(eigenvalues[0]) * np.sqrt(chi2.ppf(confidence_level, df=2))
        height = 2 * np.sqrt(eigenvalues[1]) * np.sqrt(chi2.ppf(confidence_level, df=2))

        # 绘制样本分布的散点图，并使用自定义的颜色和标签
        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=c, marker=m, label=label)

        # 绘制置信椭圆
        ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, edgecolor=c, facecolor='none')
        ax.add_patch(ellipse)

    # 获取主成分1和主成分2的方差贡献
    pc1_variance = explained[0] * 100
    pc2_variance = explained[1] * 100

    plt.xlabel(f'PC 1 ({pc1_variance:.2f}%)')
    plt.ylabel(f'PC 2 ({pc2_variance:.2f}%)')
    plt.title('PCA result with Confidence Ellipses')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'PCA_result.png'))
    plt.show()

    # 计算并绘制波数与主成分贡献率的关系图
    plt.figure()
    plt.plot(x_1, pca.components_[0], '-r', label='PC 1')
    plt.plot(x_1, pca.components_[1], '-b', label='PC 2')
    plt.gca().invert_xaxis()
    plt.xlabel('Wave-number(cm-1)')
    plt.ylabel('Coefficient')
    plt.title('Loading Plot')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, 'PCA_contribution_result.png'))
    plt.show()

    # 绘制解释方差比例图
    plt.figure()
    plt.plot(np.cumsum(explained), marker='o')
    plt.xticks(ticks=np.arange(1, 21, 2), labels=np.arange(2, 22, 2))
    plt.yticks(np.arange(0, 1.1, 0.1))  # 设置 y 轴刻度
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Scree_Plot.png'))
    plt.show()
