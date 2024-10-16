from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap
from split_dataset import split_dataset


def perform_svm_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, save_path,
                         test_size, random_state, kernel='rbf', C=1.0, gamma='scale'):
    """
    执行SVM
    参数:
    - spectrum_benign, spectrum_441, spectrum_520, spectrum_1299: 各类别的光谱数据
    - save_path: 结果保存路径
    - test_size: 测试集比例
    - random_state: 划分训练集、测试集时的随机种子
    - kernel: SVM核函数类型（例如 'linear', 'rbf', 'poly'）
    - C: 正则化参数
    - gamma: 核函数系数
    """

    # 划分数据集
    X_train_scaled, X_test_scaled, y_train, y_test = split_dataset(spectrum_benign, spectrum_441,
                                                                   spectrum_520, spectrum_1299, test_size,
                                                                   random_state)

    # 训练SVM分类器
    svm = SVC(kernel=kernel, C=C, gamma=gamma)
    svm.fit(X_train_scaled, y_train)

    # 预测
    y_pred = svm.predict(X_test_scaled)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵热力图，传递 method_name
    save_confusion_matrix_heatmap(cm, save_path, method_name='SVM', show_plot=True)
    # 计算评价指标，保存到excel中
    classification_metrics(cm, y_test, y_pred, save_path, excel_filename='SVM_metrics.xlsx')
