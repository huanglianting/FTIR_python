import joblib
import os
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap


def train_pca_rf_model(X_train_scaled, y_train, save_path, random_state, n_pca_components=20, n_estimators=100, max_depth=10):
    """
    训练随机森林分类器并保存模型
    参数:
    - X_train_scaled: 训练特征数据
    - y_train: 训练标签
    - n_pca_components: PCA降维后的主成分数量
    - n_estimators: 随机森林中树的数量
    - max_depth: 随机森林中树的最大深度
    储存:
    - rf: 训练好的随机森林分类器
    - pca: PCA对象
    """
    # PCA降维
    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)  # 在训练集上拟合PCA
    # 训练随机森林分类器
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train_pca, y_train)
    # 保存模型
    model_path = os.path.join(save_path, 'pca_rf_model.pkl')
    joblib.dump((rf, pca), model_path)
    print("Random Forest model and PCA saved successfully.")


def test_pca_rf_model(X_test_scaled, y_test, save_path, show_plot=True):
    """
    测试随机森林分类器并计算混淆矩阵和指标
    参数:
    - X_test_scaled: 测试特征数据
    - y_test: 测试标签
    - save_path: 结果保存路径
    """
    # 读取保存的模型
    model_path = os.path.join(save_path, 'pca_rf_model.pkl')
    rf, pca = joblib.load(model_path)
    # PCA转换测试集
    X_test_pca = pca.transform(X_test_scaled)
    # 预测
    y_pred = rf.predict(X_test_pca)
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵热力图
    save_confusion_matrix_heatmap(cm, save_path, method_name='PCA-RF', show_plot=show_plot)
    # 计算评价指标，保存到excel中
    classification_metrics(cm, y_test, y_pred, save_path, excel_filename='PCA-RF_metrics.xlsx')
    print("Testing completed.")
