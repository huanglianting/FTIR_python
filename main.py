import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from load_and_preprocess import load_and_preprocess


# 定义Generator和Discriminator
class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z, condition):
        gen_input = torch.cat((z, condition), dim=1)
        return self.model(gen_input)


class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        disc_input = torch.cat((x, condition), dim=1)
        return self.model(disc_input)


def train_cgan(generator, discriminator, real_data, labels, latent_dim, epochs, batch_size):
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))

    dataset = TensorDataset(real_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for real_batch, label_batch in dataloader:
            batch_size = real_batch.size(0)

            # 训练判别器
            optimizer_D.zero_grad()

            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # 真实数据
            real_output = discriminator(real_batch, label_batch.unsqueeze(1))
            d_real_loss = criterion(real_output, real_labels)

            # 生成假数据
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z, label_batch.unsqueeze(1))
            fake_output = discriminator(fake_data.detach(), label_batch.unsqueeze(1))
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_data, label_batch.unsqueeze(1))
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
            optimizer_G.step()

        print(f'Epoch [{epoch + 1}/{epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

    return generator, discriminator


def generate_enhanced_data(generator, num_samples, latent_dim):
    noise = torch.randn(num_samples, latent_dim)
    # 随机生成标签
    labels = torch.randint(0, 2, (num_samples, 1)).float()
    enhanced_data = generator(noise, labels)
    return enhanced_data, labels.long().squeeze()


# 定义基础路径
ftir_file_path = 'N:\\hlt\\FTIR\\FNA预实验\\code_test\\'
mz_file_path = r'N:\\hlt\\FTIR\\FNA预实验\\code_test\\compound measurements.xlsx'
save_path = 'N:\\hlt\\FTIR\\FNA预实验\\code_test\\result'  # 保存图片的路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 创建训练集和测试集保存文件夹
train_folder = os.path.join(save_path, 'train')
test_folder = os.path.join(save_path, 'test')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

'''
 # ===============================处理FTIR=================================================
 # 生成文件列表的通用函数
 def generate_file_lists(prefixes, num_files):
     all_file_lists = {}
     for prefix in prefixes:
         file_list = [f'{ftir_file_path}{prefix}_{i}.0.mat' for i in range(1, num_files + 1)]
         all_file_lists[prefix] = file_list
     return all_file_lists


 # 生成cancer和control的文件列表
 cancer_prefixes = [f'cancer{i}' for i in range(1, 11)]  # 一共有cancer1到cancer10，总计10个样品
 control_prefixes = [f'control{i}' for i in range(1, 11)]
 cancer_file_lists = generate_file_lists(cancer_prefixes, 3)  # 对于FTIR，每个样品重复三次滴加到基底
 control_file_lists = generate_file_lists(control_prefixes, 3)
 # 加载数据
 cancer_ftir_data = {k: [sio.loadmat(f) for f in v] for k, v in cancer_file_lists.items()}
 control_ftir_data = {k: [sio.loadmat(f) for f in v] for k, v in control_file_lists.items()}

 # FTIR光谱预处理相关参数
 threshold1 = 900  # 过滤掉小于threshold1的噪声
 threshold2 = 1800  # 过滤掉大于threshold2的噪声
 order = 2  # 多项式阶数
 frame_len = 13  # 窗口长度（帧长度）
 # 进行预处理
 control_ftir, cancer_ftir = {}, {}
 x_ftir, control_ftir['control1'] = load_and_preprocess(control_ftir_data['control1'], threshold1, threshold2,
                                                        order, frame_len, save_path)
 for key in list(control_ftir_data.keys())[1:]:
     _, control_ftir[key] = load_and_preprocess(control_ftir_data[key], threshold1, threshold2, order, frame_len,
                                                save_path)
 for key in cancer_ftir_data.keys():
     _, cancer_ftir[key] = load_and_preprocess(cancer_ftir_data[key], threshold1, threshold2, order, frame_len,
                                               save_path)

 # 打印每个样品的形状
 print("x_ftir shape:", x_ftir.shape)
 for key in control_ftir.keys():
     print(f"spectrum_control{key[len('control'):]} shape:",
           control_ftir[key].shape)  # 形状均为(467, xxxx)。例如：(467, 1517) (467, 1716)
 for key in cancer_ftir.keys():
     print(f"spectrum_cancer{key[len('cancer'):]} shape:",
           cancer_ftir[key].shape)  # 形状均为(467, xxxx)。例如：(467, 1165) (467, 1260)


 # ===================================处理mz===========================================================
 df = pd.read_excel(mz_file_path, header=1)  # 从第二行读取数据
 cancer_columns = [col for col in df.columns if 'cancer' in col.lower()]  # 分别提取每个癌症和正常样本
 control_columns = [col for col in df.columns if 'normal' in col.lower()]
 cancer_mz = {col: df[col].values for col in cancer_columns}
 control_mz = {col: df[col].values for col in control_columns}
 mz = df['m/z'].values  # 提取 m/z 列，形状为 (12572,)

 for col, values in cancer_mz.items():  # 癌症样本数据，每个样本形状为(12572,)
     print(f"{col} shape:", values.shape)
 for col, values in control_mz.items():  # 正常样本数据，每个样本形状为(12572,)
     print(f"{col} shape:", values.shape)
 print("mz shape:", mz.shape)


 # =============================分别对每个样本的FTIR和mz数据进行处理======================================
 # 合并所有样本的特征和标签
 ftir_features_list = []
 mz_features_list = []
 labels_list = []

 # 处理癌症样本
 for i in range(1, 11):
     cancer_ftir_key = f'cancer{i}'
     cancer_mz_key = f'cancer_{i} [1]'
     ftir_samples = cancer_ftir[cancer_ftir_key].T  # shape：(xxxx, 467)，如 (1421, 467)
     mz_sample = cancer_mz[cancer_mz_key].reshape(1, -1)  # shape：(1, 12572)
     # 复制代谢组学数据，使其样本数量和 FTIR 数据的样本数量相同
     mz_sample = np.repeat(mz_sample, ftir_samples.shape[0], axis=0)
     ftir_features_list.append(ftir_samples)
     mz_features_list.append(mz_sample)
     labels_list.extend([1] * ftir_samples.shape[0])  # 癌症的标签标记为1

 # 处理正常样本
 for i in range(1, 11):
     control_ftir_key = f'control{i}'
     control_mz_key = f'normal_{i} [1]'
     ftir_samples = control_ftir[control_ftir_key].T
     mz_sample = control_mz[control_mz_key].reshape(1, -1)
     # 复制代谢组学数据，使其样本数量和 FTIR 数据的样本数量相同
     mz_sample = np.repeat(mz_sample, ftir_samples.shape[0], axis=0)
     ftir_features_list.append(ftir_samples)
     mz_features_list.append(mz_sample)
     labels_list.extend([0] * ftir_samples.shape[0])  # 对照组（正常）的标签标记为0

 ftir_features = np.vstack(ftir_features_list)
 mz_features = np.vstack(mz_features_list)
 labels = np.array(labels_list)

 # ==========================================划分训练、测试集==================================================
 # 划分训练集和测试集7:3，记得打乱一下random，不要 1111100000，要0110010001这样的
 ftir_train, ftir_test, mz_train, mz_test, y_train, y_test = train_test_split(
     ftir_features, mz_features, labels, test_size=0.3, random_state=41
 )

 print("ftir_train shape:", ftir_train.shape)    # (20447, 467)
 print("mz_train shape:", mz_train.shape)    # (20447, 12572)
 print("y_train shape:", y_train.shape)  # (20447,)
 print("ftir_test shape:", ftir_test.shape)  # (8763, 467)
 print("mz_test shape:", mz_test.shape)  # (8763, 12572)
 print("y_test shape:", y_test.shape)    # (8763,)

 # 数据增强：对训练集添加高斯噪声。因为只有10个样本，很容易过拟合。
 noise_std = 0.1  # 噪声的标准差，可以根据需要调整
 ftir_noise = np.random.normal(0, noise_std, ftir_train.shape)
 mz_noise = np.random.normal(0, noise_std, mz_train.shape)
 ftir_train = ftir_train + ftir_noise
 mz_train = mz_train + mz_noise

 # 保存训练集和测试集
 np.save(os.path.join(train_folder, 'ftir_train.npy'), ftir_train)
 np.save(os.path.join(train_folder, 'mz_train.npy'), mz_train)
 np.save(os.path.join(train_folder, 'y_train.npy'), y_train)
 np.save(os.path.join(test_folder, 'ftir_test.npy'), ftir_test)
 np.save(os.path.join(test_folder, 'mz_test.npy'), mz_test)
 np.save(os.path.join(test_folder, 'y_test.npy'), y_test)
 '''


# ====================读取训练集和测试集=====================================
def read_data():
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
# 定义模态特异性特征提取的MLP分支网络
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
        self.co_attention = CoAttentionNet(64)
        self.fc = nn.Linear(64 * 2, 1)
        self.bilstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, bidirectional=True, batch_first=True)
        # 将输出维度 (hidden_size * 2) 映射到2（因为是二分类问题）
        self.fc_after_bilstm = nn.Linear(16 * 2, 2)
        # 将输出转换为概率分布，例如输出：0.8（80%的概率）为是癌症，0.2为不是癌症
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ftir, mz):
        # 放入Co-Attention Net
        combined_features = self.co_attention(ftir, mz)
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
