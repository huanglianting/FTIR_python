import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap
from split_dataset import split_dataset


def perform_pca_lda_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, save_path,
                             test_size, random_state, n_pca_components=20):
    """
    执行PCA-LDA
    参数:
    - spectrum_benign, spectrum_441, spectrum_520, spectrum_1299: 各类别的光谱数据
    - save_path: 结果保存路径
    - test_size: 测试集比例
    - random_state: 划分训练集、测试集时的随机种子
    - n_pca_components: PCA降维后的主成分数量
    """

    # 划分数据集
    X_train_scaled, X_test_scaled, y_train, y_test = split_dataset(spectrum_benign, spectrum_441,
                                                                   spectrum_520, spectrum_1299, test_size,
                                                                   random_state)

    # PCA降维
    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)  # 在训练集上拟合PCA
    X_test_pca = pca.transform(X_test_scaled)  # 使用相同的PCA模型转换测试集

    # 训练LDA分类器
    num_classes = len(np.unique(y_train))
    lda = LDA(n_components=min(num_classes - 1, 3))  # 最多 classes - 1 维
    # lda = LDA(n_components=3)
    lda.fit(X_train_pca, y_train)

    # 预测
    y_pred = lda.predict(X_test_pca)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵作图出来，更直观，在图片上标注出数值
    # 绘制混淆矩阵热力图，传递 method_name
    save_confusion_matrix_heatmap(cm, save_path, method_name='PCA-LDA', show_plot=False)
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
