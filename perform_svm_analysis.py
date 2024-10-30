import os
import joblib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap


def train_svm_model(X_train_scaled, y_train, save_path, kernel='rbf', C=1.0, gamma='scale'):
    """
    训练 SVM 分类器
    参数:
    - X_train_scaled: 训练特征数据
    - y_train: 训练标签
    - kernel: SVM核函数类型（例如 'linear', 'rbf', 'poly'）
    - C: 正则化参数
    - gamma: 核函数系数
    返回:
    - svm: 训练好的 SVM 分类器
    """
    # 初始化 SVM 分类器
    svm = SVC(kernel=kernel, C=C, gamma=gamma)
    # 训练 SVM 分类器
    svm.fit(X_train_scaled, y_train)
    # 保存训练好的模型
    model_path = os.path.join(save_path, 'svm_model.pkl')
    joblib.dump(svm, model_path)
    print("SVM model saved successfully.")


def test_svm_model(X_test_scaled, y_test, save_path, show_plot=True):
    """
    测试 SVM 分类器并计算混淆矩阵和指标
    参数:
    - svm: 训练好的 SVM 分类器
    - X_test_scaled: 测试特征数据
    - y_test: 测试标签
    - save_path: 结果保存路径
    """
    # 读取保存的 SVM 模型
    model_path = os.path.join(save_path, 'svm_model.pkl')
    svm = joblib.load(model_path)
    # 预测
    y_pred = svm.predict(X_test_scaled)
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵热力图，传递 method_name
    save_confusion_matrix_heatmap(cm, save_path, method_name='SVM', show_plot=show_plot)
    # 计算评价指标，保存到excel中
    classification_metrics(cm, y_test, y_pred, save_path, excel_filename='SVM_metrics.xlsx')
    print("Testing completed.")