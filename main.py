import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from load_and_preprocess import load_and_preprocess

# 定义基础路径
ftir_file_path = 'N:\\hlt\\FTIR\\FNA预实验\\code_test\\'
mz_file_path = r'N:\\hlt\\FTIR\\FNA预实验\\code_test\\compound measurements.xlsx'
save_path = 'N:\\hlt\\FTIR\\FNA预实验\\code_test\\resul'  # 保存图片的路径
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


# ==================接下来把他们放进 MLP 里====================================
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
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        Q = self.W_q(x1)
        K = self.W_k(x2)
        V = self.W_v(x2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_probs = self.softmax(attn_scores)
        attn_output = torch.matmul(attn_probs, V)
        combined_features = attn_output + x1
        return combined_features


# 定义完整的模型
class MultiModalModel(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim):
        super(MultiModalModel, self).__init__()
        self.ftir_mlp = FTIRMLP(ftir_input_dim)
        self.mz_mlp = MZMLP(mz_input_dim)
        self.co_attention = CoAttentionNet(64)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ftir, mz):
        ftir_features = self.ftir_mlp(ftir)
        mz_features = self.mz_mlp(mz)
        combined_features = self.co_attention(ftir_features, mz_features)
        output = self.fc(combined_features)
        output = self.sigmoid(output)
        return output


# 初始化模型
model = MultiModalModel(ftir_train.shape[1], mz_train.shape[1])

# 转换为torch张量
ftir_train = torch.tensor(ftir_train, dtype=torch.float32)
mz_train = torch.tensor(mz_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# 前向传播
output = model(ftir_train, mz_train)
print("Output shape:", output.shape)

# softmax，例如输出：0.8（80%的概率）为是癌症，0.2为不是癌症
