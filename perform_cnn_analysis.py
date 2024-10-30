import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap
import os


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
