import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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


def train_pca_rf_model(X_train_scaled, y_train, save_path, random_state, n_pca_components=20, n_estimators=100,
                       max_depth=10):
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


# CNN模型类
class CNNClassifier(nn.Module):
    def __init__(self, input_length, num_classes=4):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_length // 4), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_cnn_model(X_train_scaled, y_train, save_path, epochs=100, batch_size=32, lr=0.001):
    # 获取输入长度
    input_length = X_train_scaled.shape[1]
    # 将数据转换为 torch 张量并调整形状
    X_train_scaled = torch.tensor(X_train_scaled[..., np.newaxis].transpose((0, 2, 1)), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    # 构建数据加载器
    train_dataset = TensorDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 定义模型、损失函数和优化器
    model = CNNClassifier(input_length=input_length, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练模型
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    # 保存训练好的模型
    torch.save(model.state_dict(), os.path.join(save_path, "cnn_model.pth"))
    print("CNN Model saved successfully.")


def test_cnn_model(X_test_scaled, y_test, save_path, batch_size=32, show_plot=True):
    # 获取输入长度
    input_length = X_test_scaled.shape[1]
    # 将数据转换为 torch 张量并调整形状
    X_test_scaled = torch.tensor(X_test_scaled[..., np.newaxis].transpose((0, 2, 1)), dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    # 加载模型
    model = CNNClassifier(input_length=input_length, num_classes=4)
    model.load_state_dict(torch.load(os.path.join(save_path, "cnn_model.pth")))
    model.eval()
    # 测试模型
    y_pred = []
    test_loader = DataLoader(TensorDataset(X_test_scaled, y_test), batch_size=batch_size)
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())  # 转换为numpy数组
    # 计算评价指标并保存
    cm = confusion_matrix(y_test.numpy(), y_pred)
    save_confusion_matrix_heatmap(cm, save_path, method_name='CNN', show_plot=show_plot)
    classification_metrics(cm, y_test.numpy(), y_pred, save_path, excel_filename='CNN_metrics.xlsx')
    print("Testing completed.")