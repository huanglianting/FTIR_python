import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap
from split_dataset import split_dataset


class CNNClassifier(nn.Module):
    def __init__(self, input_length, num_classes=4):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        # 动态计算全连接层输入大小
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


def perform_cnn_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, save_path,
                         test_size=0.3, random_state=42, epochs=100, batch_size=32, lr=0.001):
    """
    执行CNN分类分析
    参数:
    - spectrum_benign, spectrum_441, spectrum_520, spectrum_1299: 各类别的光谱数据
    - save_path: 结果保存路径
    - test_size: 测试集比例
    - random_state: 划分训练集、测试集时的随机种子
    - epochs: 训练轮数
    - batch_size: 批量大小
    - lr: 学习率
    """

    # 划分数据集
    X_train_scaled, X_test_scaled, y_train, y_test = split_dataset(spectrum_benign, spectrum_441,
                                                                   spectrum_520, spectrum_1299, test_size,
                                                                   random_state)

    # 获取输入长度
    input_length = X_train_scaled.shape[1]

    # 将数据转换为 torch 张量并调整形状
    X_train_scaled = torch.tensor(X_train_scaled[..., np.newaxis].transpose((0, 2, 1)), dtype=torch.float32)
    X_test_scaled = torch.tensor(X_test_scaled[..., np.newaxis].transpose((0, 2, 1)), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 构建数据加载器
    train_dataset = TensorDataset(X_train_scaled, y_train)
    test_dataset = TensorDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

    # 测试模型
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())  # 转换为numpy数组

    # 计算混淆矩阵
    cm = confusion_matrix(y_test.numpy(), y_pred)
    save_confusion_matrix_heatmap(cm, save_path, method_name='CNN', show_plot=True)
    # 计算评价指标，保存到excel中
    classification_metrics(cm, y_test.numpy(), y_pred, save_path, excel_filename='CNN_metrics.xlsx')
