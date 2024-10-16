from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap
from split_dataset import split_dataset


def perform_pca_rf_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, save_path,
                            test_size, random_state, n_pca_components=20, n_estimators=100, max_depth=None):
    """
    使用PCA和随机森林进行四分类分析
    参数:
    - spectrum_benign, spectrum_441, spectrum_520, spectrum_1299: 各类光谱数据
    - save_path: 结果保存路径
    - n_pca_components: PCA降维后的主成分数量
    - test_size: 测试集比例
    - random_state: 随机种子
    - n_estimators: 随机森林中树的数量
    - max_depth: 随机森林中树的最大深度
    """

    # 划分数据集
    X_train_scaled, X_test_scaled, y_train, y_test = split_dataset(spectrum_benign, spectrum_441,
                                                                   spectrum_520, spectrum_1299, test_size,
                                                                   random_state)

    # PCA降维
    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)  # 在训练集上拟合PCA
    X_test_pca = pca.transform(X_test_scaled)  # 使用相同的PCA模型转换测试集

    # 训练随机森林分类器
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train_pca, y_train)

    # 预测
    y_pred = rf.predict(X_test_pca)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵热力图
    save_confusion_matrix_heatmap(cm, save_path, method_name='PCA-RF', show_plot=False)
    # 计算评价指标，保存到excel中
    classification_metrics(cm, y_test, y_pred, save_path, excel_filename='PCA-RF_metrics.xlsx')
