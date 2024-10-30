import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap


def train_pca_lda_model(X_train_scaled, y_train, save_path, n_pca_components=20):
    """
    训练 PCA-LDA 模型并保存模型
    参数:
    - X_train_scaled: 训练特征数据
    - y_train: 训练标签
    - n_pca_components: PCA降维后的主成分数量
    储存:
    - lda: 训练好的LDA模型
    - pca: PCA对象
    """
    # PCA降维
    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)  # 在训练集上拟合PCA
    # 训练LDA分类器
    num_classes = len(np.unique(y_train))
    lda = LDA(n_components=min(num_classes - 1, 3))  # 最多 classes - 1 维
    lda.fit(X_train_pca, y_train)
    # 保存模型
    model_path = os.path.join(save_path, 'pca_lda_model.pkl')
    joblib.dump((lda, pca), model_path)
    print("LDA model and PCA saved successfully.")


def test_pca_lda_model(X_test_scaled, y_test, save_path, show_plot=True):
    """
    测试 PCA-LDA 模型并计算混淆矩阵和指标
    参数:
    - X_test_scaled: 测试特征数据
    - y_test: 测试标签
    - save_path: 结果保存路径
    """
    # 读取保存的模型
    model_path = os.path.join(save_path, 'pca_lda_model.pkl')
    lda, pca = joblib.load(model_path)
    # PCA转换测试集
    X_test_pca = pca.transform(X_test_scaled)  # 使用相同的PCA模型转换测试集
    # 预测
    y_pred = lda.predict(X_test_pca)
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵作图出来，更直观，在图片上标注出数值
    # 绘制混淆矩阵热力图，传递 method_name
    save_confusion_matrix_heatmap(cm, save_path, method_name='PCA-LDA', show_plot=show_plot)
    # 计算评价指标，保存到excel中
    classification_metrics(cm, y_test, y_pred, save_path, excel_filename='PCA-LDA_metrics.xlsx')
    print("Testing completed.")


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
