import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import preprocess_data

# 定义基础路径
ftir_file_path = './data/'
mz_file_path = r'./data/compound_measurements.xlsx'
save_path = './result'  # 保存图片的路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 调用预处理函数（文件夹在预处理中创建，此处无需重复）
train_folder = os.path.join(save_path, 'train')
test_folder = os.path.join(save_path, 'test')
ftir_train, mz_train, y_train, ftir_test, mz_test, y_test = preprocess_data(ftir_file_path, mz_file_path, train_folder,
                                                                            test_folder, save_path)
"""
# ====================读取训练集和测试集=====================================
def read_data():
    train_folder = os.path.join(save_path, 'train')
    test_folder = os.path.join(save_path, 'test')
    ftir_train = np.load(os.path.join(train_folder, 'ftir_train.npy'))
    mz_train = np.load(os.path.join(train_folder, 'mz_train.npy'))
    y_train = np.load(os.path.join(train_folder, 'y_train.npy'))
    ftir_test = np.load(os.path.join(test_folder, 'ftir_test.npy'))
    mz_test = np.load(os.path.join(test_folder, 'mz_test.npy'))
    y_test = np.load(os.path.join(test_folder, 'y_test.npy'))
    return ftir_train, mz_train, y_train, ftir_test, mz_test, y_test


ftir_train, mz_train, y_train, ftir_test, mz_test, y_test = read_data()
"""
# 数据标准化
scaler_ftir = StandardScaler()
ftir_train = scaler_ftir.fit_transform(ftir_train)
ftir_test = scaler_ftir.transform(ftir_test)
scaler_mz = StandardScaler()
mz_train = scaler_mz.fit_transform(mz_train)
mz_test = scaler_mz.transform(mz_test)

# 转换为PyTorch张量
ftir_train = torch.tensor(ftir_train, dtype=torch.float32)
mz_train = torch.tensor(mz_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
ftir_test = torch.tensor(ftir_test, dtype=torch.float32)
mz_test = torch.tensor(mz_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# ==================主模型定义====================================
# 定义FTIR模态特异性特征提取的MLP分支网络
class FTIRMLP(nn.Module):
    def __init__(self, input_dim):
        super(FTIRMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


# 定义MZ模态特异性特征提取的MLP分支网络
class MZMLP(nn.Module):
    def __init__(self, input_dim):
        super(MZMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


# 定义Co - Attention net
class CoAttentionNet(nn.Module):
    def __init__(self, input_dim):
        super(CoAttentionNet, self).__init__()
        self.W_qA = nn.Linear(input_dim, input_dim)
        self.W_kA = nn.Linear(input_dim, input_dim)
        self.W_vA = nn.Linear(input_dim, input_dim)
        self.W_qB = nn.Linear(input_dim, input_dim)
        self.W_kB = nn.Linear(input_dim, input_dim)
        self.W_vB = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X_A, X_B):
        Q_A = self.W_qA(X_A)
        K_A = self.W_kA(X_A)
        V_A = self.W_vA(X_A)
        Q_B = self.W_qB(X_B)
        K_B = self.W_kB(X_B)
        V_B = self.W_vB(X_B)
        S_AB = torch.matmul(Q_A, K_B.transpose(-2, -1))
        S_BA = torch.matmul(Q_B, K_A.transpose(-2, -1))
        A_AB = self.softmax(S_AB)
        A_BA = self.softmax(S_BA)
        F_A = torch.matmul(A_AB, V_B)
        F_B = torch.matmul(A_BA, V_A)
        # 这里简单将两个融合特征拼接作为输出，可根据需求调整
        output = torch.cat([F_A, F_B], dim=-1)
        return output


# 定义完整的模型
class MultiModalModel(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim):
        super(MultiModalModel, self).__init__()
        self.ftir_mlp = FTIRMLP(ftir_input_dim)
        self.mz_mlp = MZMLP(mz_input_dim)
        self.co_attention = CoAttentionNet(32)
        self.fc = nn.Linear(32 * 2, 2)  # 直接输出类别概率
        self.softmax = nn.Softmax(dim=1)
        # 添加 L2 正则化
        self.fc.weight.data = torch.nn.Parameter(torch.nn.init.xavier_uniform_(self.fc.weight.data))
        self.fc.bias.data = torch.nn.Parameter(torch.nn.init.zeros_(self.fc.bias.data))

    def forward(self, ftir, mz):
        # 先通过 MLP 提取特征
        ftir_features = self.ftir_mlp(ftir)
        mz_features = self.mz_mlp(mz)
        # 放入Co-Attention Net
        combined_features = self.co_attention(ftir_features, mz_features)
        output = self.fc(combined_features)
        output = self.softmax(output)
        return output


# ==================主模型训练====================================
def train_main_model(model, ftir_data, mz_data, labels, epochs, batch_size, writer, noise_std):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    dataset = TensorDataset(ftir_data, mz_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for ftir_batch, mz_batch, label_batch in dataloader:
            # 确保每次迭代开始时梯度清零
            optimizer.zero_grad()
            #  ================== 添加高斯噪声  ==================
            # 生成与输入数据同形状的高斯噪声
            ftir_noise = torch.randn_like(ftir_batch) * noise_std  # 与FTIR数据同分布的噪声
            mz_noise = torch.randn_like(mz_batch) * noise_std  # 与MZ数据同分布的噪声
            # 将噪声添加到输入数据中（训练时添加，测试时不添加）
            ftir_noisy = ftir_batch + ftir_noise
            mz_noisy = mz_batch + mz_noise
            # ==================  输入带噪声的数据到模型 ==================
            outputs = model(ftir_noisy, mz_noisy)  # 注意：输入改为带噪声的数据
            loss = criterion(outputs, label_batch)
            # 执行反向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()

        epoch_loss /= len(dataloader)
        accuracy = correct / total
        losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

        # 添加训练损失和准确率到 TensorBoard
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', accuracy, epoch)

    return losses


# ==================主程序====================================
# 初始化 TensorBoard 的 SummaryWriter
writer = SummaryWriter('./runs')
# 检查输入维度和条件维度的值
input_dim_ftir = ftir_train.shape[1]
condition_dim = 1
print(f"FTIR input_dim: {input_dim_ftir}, condition_dim: {condition_dim}")
input_dim_mz = mz_train.shape[1]
print(f"MZ input_dim: {input_dim_mz}, condition_dim: {condition_dim}")

# 初始化主模型
model = MultiModalModel(ftir_train.shape[1], mz_train.shape[1])

# 训练主模型, y_train 是指 label_train
train_losses = train_main_model(model, ftir_train, mz_train, y_train, epochs=100, batch_size=32, writer=writer, noise_std=0.3)

# 保存模型 checkpoint
torch.save(model.state_dict(), './checkpoints/multimodal_model_checkpoint.pth')

# 绘制训练损失曲线
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('training_loss.png')
plt.close()

# 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(ftir_test, mz_test)
    _, predicted = torch.max(test_outputs, 1)

# 打印标签分布和预测结果
print("训练集标签分布:", np.bincount(y_train.numpy()))
print("测试集标签分布:", np.bincount(y_test.numpy()))
print("预测结果:", predicted[:10].numpy())
print("真实标签:", y_test[:10].numpy())
# 计算性能指标
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)

# 添加测试集性能指标到 TensorBoard
writer.add_scalar('Test Accuracy', accuracy, 0)
writer.add_scalar('Test Precision', precision, 0)
writer.add_scalar('Test Recall', recall, 0)
writer.add_scalar('Test F1 Score', f1, 0)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# 计算每个类别的准确率
class_0_indices = (y_test == 0).nonzero(as_tuple=True)[0]
class_1_indices = (y_test == 1).nonzero(as_tuple=True)[0]

accuracy_class_0 = (predicted[class_0_indices] == y_test[class_0_indices]).float().mean()
accuracy_class_1 = (predicted[class_1_indices] == y_test[class_1_indices]).float().mean()

print(f'类别0准确率: {accuracy_class_0:.4f}')
print(f'类别1准确率: {accuracy_class_1:.4f}')

# 绘制混淆矩阵
cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# 关闭 TensorBoard 的 SummaryWriter
writer.close()