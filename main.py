import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from cgan_utils import Generator, Discriminator, train_cgan, generate_enhanced_data  # 导入封装好的CGAN模块
from data_preprocessing import preprocess_data


# 定义基础路径
ftir_file_path = 'N:\\hlt\\FTIR\\FNA预实验\\code_test\\'
mz_file_path = r'N:\\hlt\\FTIR\\FNA预实验\\code_test\\compound measurements.xlsx'
save_path = 'N:\\hlt\\FTIR\\FNA预实验\\code_test\\result'  # 保存图片的路径
if not os.path.exists(save_path):
    os.makedirs(save_path)


# 调用预处理函数（文件夹在预处理中创建，此处无需重复）
# ftir_train, mz_train, y_train, ftir_test, mz_test, y_test = preprocess_data(ftir_file_path, mz_file_path, train_folder, test_folder, save_path)

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
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


# 定义MZ模态特异性特征提取的MLP分支网络
class MZMLP(nn.Module):
    def __init__(self, input_dim):
        super(MZMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
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
        self.co_attention = CoAttentionNet(64)
        self.fc = nn.Linear(64 * 2, 1)
        self.bilstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, bidirectional=True, batch_first=True)
        # 将输出维度 (hidden_size * 2) 映射到2（因为是二分类问题）
        self.fc_after_bilstm = nn.Linear(16 * 2, 2)
        # 将输出转换为概率分布，例如输出：0.8（80%的概率）为是癌症，0.2为不是癌症
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ftir, mz):
        # 先通过 MLP 提取特征
        ftir_features = self.ftir_mlp(ftir)
        mz_features = self.mz_mlp(mz)
        # 放入Co-Attention Net
        combined_features = self.co_attention(ftir_features, mz_features)
        output = self.fc(combined_features)
        # 调整维度以适应 BiLSTM 的输入要求 [batch_size, seq_len, input_size]
        output = output.unsqueeze(2)
        lstm_output, _ = self.bilstm(output)
        lstm_output = lstm_output[:, -1, :]
        output = self.fc_after_bilstm(lstm_output)
        output = self.softmax(output)
        return output


# ==================主模型训练====================================
def train_main_model(model, ftir_data, mz_data, labels, epochs, batch_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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
            outputs = model(ftir_batch, mz_batch)
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
    return losses


# ==================主程序====================================
# 初始化CGAN
latent_dim = 100
epochs_cgan = 20
batch_size_cgan = 32

# 检查输入维度和条件维度的值
input_dim_ftir = ftir_train.shape[1]
condition_dim = 1
print(f"FTIR input_dim: {input_dim_ftir}, condition_dim: {condition_dim}")
generator_ftir = Generator(latent_dim, condition_dim, ftir_train.shape[1])
discriminator_ftir = Discriminator(input_dim_ftir, condition_dim)
input_dim_mz = mz_train.shape[1]
print(f"MZ input_dim: {input_dim_mz}, condition_dim: {condition_dim}")
generator_mz = Generator(latent_dim, condition_dim, mz_train.shape[1])
discriminator_mz = Discriminator(input_dim_mz, condition_dim)

# 检查是否已经保存了增强数据
ftir_enhanced_path = os.path.join(save_path, 'ftir_enhanced.pt')
mz_enhanced_path = os.path.join(save_path, 'mz_enhanced.pt')

if os.path.exists(ftir_enhanced_path) and os.path.exists(mz_enhanced_path):
    print("Loading enhanced data from saved files...")
    ftir_combined, mz_combined, labels_combined = torch.load(ftir_enhanced_path), torch.load(
        mz_enhanced_path), torch.load(os.path.join(save_path, 'labels_combined.pt'))
else:
    # 训练CGAN
    # 主程序中训练CGAN部分保持不变
    generator_ftir, discriminator_ftir = train_cgan(
        generator_ftir, discriminator_ftir, ftir_train, y_train, latent_dim, epochs_cgan, batch_size_cgan
    )
    generator_mz, discriminator_mz = train_cgan(
        generator_mz, discriminator_mz, mz_train, y_train, latent_dim, epochs_cgan, batch_size_cgan
    )
    # 生成增强数据
    num_enhanced_samples = 200
    enhanced_ftir, enhanced_ftir_labels = generate_enhanced_data(generator_ftir, num_enhanced_samples, latent_dim)
    enhanced_mz, enhanced_mz_labels = generate_enhanced_data(generator_mz, num_enhanced_samples, latent_dim)
    # 打印 y_train 和 enhanced_ftir_labels 的维度
    print(f"y_train shape: {y_train.shape}")
    print(f"enhanced_ftir_labels shape: {enhanced_ftir_labels.shape}")
    # 合并增强数据和原始数据
    ftir_combined = torch.cat([ftir_train, enhanced_ftir], dim=0)
    mz_combined = torch.cat([mz_train, enhanced_mz], dim=0)
    labels_combined = torch.cat([y_train, enhanced_ftir_labels], dim=0)
    # 保存增强数据
    torch.save(ftir_combined, ftir_enhanced_path)
    torch.save(mz_combined, mz_enhanced_path)
    torch.save(labels_combined, os.path.join(save_path, 'labels_combined.pt'))
    print("Enhanced data saved to files.")

# 初始化主模型
model = MultiModalModel(ftir_train.shape[1], mz_train.shape[1])

# 训练主模型
train_losses = train_main_model(model, ftir_combined, mz_combined, labels_combined, epochs=200, batch_size=32)

# 绘制训练损失曲线
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(ftir_test, mz_test)
    _, predicted = torch.max(test_outputs, 1)

# 计算性能指标
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# 绘制混淆矩阵
cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
