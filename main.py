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
from sklearn.model_selection import KFold

# 定义基础路径
ftir_file_path = './data/'
mz_file_path1 = r'./data/compound_measurements.xlsx'
mz_file_path2 = r'./data/compound_measurements2.xlsx'
save_path = './result'  # 保存图片的路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 调用预处理函数
train_folder = os.path.join(save_path, 'train')
test_folder = os.path.join(save_path, 'test')
ftir_train, mz_train, y_train, patient_indices_train, ftir_test, mz_test, y_test, patient_indices_test = preprocess_data(
    ftir_file_path, mz_file_path1,
    mz_file_path2, train_folder,
    test_folder, save_path)
# 打印训练集和测试集的类别分布
print("训练集类别分布:", np.bincount(y_train))
print("测试集类别分布:", np.bincount(y_test))

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
class CoAttentionNet(nn.Module):
    def __init__(self, input_dim, d_k=None):
        super(CoAttentionNet, self).__init__()
        self.input_dim = input_dim
        self.d_k = d_k if d_k is not None else input_dim  # 缩放因子d_k，若未指定，默认=input_dim
        self.W_qI = nn.Linear(input_dim, self.d_k)  # Q的维度为d_k
        self.W_kI = nn.Linear(input_dim, self.d_k)  # K的维度为d_k
        self.W_vI = nn.Linear(input_dim, input_dim)  # V的维度保持input_dim
        self.W_qM = nn.Linear(input_dim, self.d_k)
        self.W_kM = nn.Linear(input_dim, self.d_k)
        self.W_vM = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X_I, X_M):
        # X_I (模态FTIR) 和 X_M (模态MZ) 形状均为 [batch_size, input_dim]
        # 计算模态FTIR的Q、K、V
        Q_I = self.W_qI(X_I)  # [batch_size, d_k]
        K_I = self.W_kI(X_I)  # [batch_size, d_k]
        V_I = self.W_vI(X_I)  # [batch_size, input_dim]
        # 计算模态MZ的Q、K、V
        Q_M = self.W_qM(X_M)  # [batch_size, d_k]
        K_M = self.W_kM(X_M)  # [batch_size, d_k]
        V_M = self.W_vM(X_M)  # [batch_size, input_dim]
        # 计算跨模态注意力分数（带缩放因子）
        S_IM = torch.matmul(Q_M, K_I.transpose(-2, -1)) / np.sqrt(self.d_k)  # [batch_size, batch_size]
        S_MI = torch.matmul(Q_I, K_M.transpose(-2, -1)) / np.sqrt(self.d_k)  # [batch_size, batch_size]
        # 软最大化归一化
        A_IM = self.softmax(S_IM)  # 模态I到M的注意力权重
        A_MI = self.softmax(S_MI)  # 模态M到I的注意力权重
        # 计算跨模态上下文向量
        F_IM = torch.matmul(A_IM, V_I)  # [batch_size, input_dim]
        F_MI = torch.matmul(A_MI, V_M)  # [batch_size, input_dim]
        # 模态内特征与注意力结果融合（逐元素相乘）
        Hat_A_IM = A_IM.unsqueeze(-1) * X_M.unsqueeze(1)  # [batch_size, batch_size, input_dim]
        Hat_A_MI = A_MI.unsqueeze(-1) * X_I.unsqueeze(1)  # [batch_size, batch_size, input_dim]
        # 模态间注意力矩阵乘法（特征维度交互）
        A_MI_final = torch.matmul(Hat_A_IM.permute(0, 2, 1), Hat_A_MI)  # [batch_size, input_dim, input_dim]
        A_MI_flatten = A_MI_final.flatten(start_dim=1)  # [batch_size, input_dim^2]
        # 融合所有特征：跨模态上下文 + 模态间注意力
        output = torch.cat([F_IM, F_MI, A_MI_flatten], dim=-1)  # [batch_size, input_dim*2 + input_dim^2]
        return output


# 完整的模型
class MultiModalModel(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim):
        super(MultiModalModel, self).__init__()
        self.input_dim = 128
        self.co_attention = CoAttentionNet(input_dim=self.input_dim, d_k=32)

        self.reduced_dim1 = 1024
        self.reduced_dim2 = 128
        self.flatten = nn.Flatten()
        # 计算全连接层输入维度：input_dim*2 + input_dim^2 = 128*2 + 128^2
        self.linear1 = nn.Linear(self.input_dim * 2 + self.input_dim ** 2, self.reduced_dim1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(self.reduced_dim1, self.reduced_dim2)
        self.dropout2 = nn.Dropout(0.3)

        self.fc = nn.Linear(self.reduced_dim2, 2)
        self.softmax = nn.Softmax(dim=1)

        # L2 正则化
        self.fc.weight.data = torch.nn.Parameter(torch.nn.init.xavier_uniform_(self.fc.weight.data))
        self.fc.bias.data = torch.nn.Parameter(torch.nn.init.zeros_(self.fc.bias.data))

    def forward(self, ftir, mz):
        combined_features = self.co_attention(ftir, mz)
        combined_features = self.flatten(combined_features)
        combined_features = self.relu(self.linear1(combined_features))
        combined_features = self.dropout1(combined_features)
        combined_features = self.relu(self.linear2(combined_features))
        combined_features = self.dropout2(combined_features)
        output = self.fc(combined_features)
        output = self.softmax(output)
        return output


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss  # 因为要最小化损失，所以取负号
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # 当验证损失降低时保存模型
        if self.verbose:
            print(f'Validation loss 下降 ({self.val_loss_min:.6f} --> {val_loss:.6f}).  保存模型 ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ==================数据增强====================================
def data_augmentation(x, noise_std=0.3, scale_range=(0.9, 1.1), shift_range=(-0.05, 0.05), keep_ratio=0.8):
    # 高斯噪声
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise
    # 随机缩放
    scale = torch.FloatTensor([np.random.uniform(*scale_range)]).to(x.device)
    x_aug = x_aug * scale
    # 随机偏移
    # shift = torch.FloatTensor([np.random.uniform(*shift_range)]).to(x.device)
    # x_aug = x_aug + shift
    # 随机保留峰（mask操作）
    # mask = torch.rand(x_aug.shape[1]) > (1 - keep_ratio)
    # x_aug = x_aug * mask.float().to(x.device)
    return x_aug


# ==================主模型训练====================================
def train_main_model(model, ftir_train, mz_train, y_train, ftir_val, mz_val, y_val, epochs, batch_size, writer,
                     noise_std):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # L2正则化
    early_stopping = EarlyStopping(patience=10, verbose=True, path='./checkpoints/best_model.pth')
    train_dataset = TensorDataset(ftir_train, mz_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(ftir_val, mz_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for ftir_batch, mz_batch, label_batch in train_dataloader:
            optimizer.zero_grad()
            ftir_noisy = data_augmentation(ftir_batch)
            mz_noisy = data_augmentation(mz_batch)
            outputs = model(ftir_noisy, mz_noisy)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()
        train_loss /= len(train_dataloader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for ftir_batch, mz_batch, label_batch in val_dataloader:
                outputs = model(ftir_batch, mz_batch)
                loss = criterion(outputs, label_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += label_batch.size(0)
                correct += (predicted == label_batch).sum().item()
        val_loss /= len(val_dataloader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        # 添加指标到TensorBoard
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Training Accuracy', train_accuracy, epoch)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 加载最佳模型
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    return model, train_losses, val_losses, train_accuracies, val_accuracies


# ==================主程序====================================
# 按患者级别实现五折交叉验证
n_splits = 5
fold_metrics = []
fold_results = []

# 获取训练集中的患者ID
unique_patients = np.unique(patient_indices_train)

# 患者级划分
patient_to_fold = {}
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(unique_patients)):
    for patient in unique_patients[val_idx]:
        patient_to_fold[patient] = fold

# 为每个样本分配折索引
fold_indices = np.array([patient_to_fold[pid] for pid in patient_indices_train])

# 进行交叉验证
for fold in range(n_splits):
    print(f"\n=========== 第 {fold + 1}/{n_splits} 折 ===========")
    val_mask = (fold_indices == fold)
    train_mask = ~val_mask
    # 提取数据
    ftir_train_fold = ftir_train[train_mask]
    mz_train_fold = mz_train[train_mask]
    y_train_fold = y_train[train_mask]
    ftir_val_fold = ftir_train[val_mask]
    mz_val_fold = mz_train[val_mask]
    y_val_fold = y_train[val_mask]
    print(f"Fold {fold} train classes: {np.bincount(y_train_fold)}")
    print(f"Fold {fold} val classes: {np.bincount(y_val_fold)}")
    print(f"训练集: {len(ftir_train_fold)}样本, {len(np.unique(patient_indices_train[train_mask]))}患者")
    print(f"验证集: {len(ftir_val_fold)}样本, {len(np.unique(patient_indices_train[val_mask]))}患者")

    # 训练模型
    writer = SummaryWriter(f'./runs/fold_{fold + 1}')
    model = MultiModalModel(ftir_train_fold.shape[1], mz_train_fold.shape[1])
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_main_model(
        model, ftir_train_fold, mz_train_fold, y_train_fold,
        ftir_val_fold, mz_val_fold, y_val_fold,
        epochs=200, batch_size=32, writer=writer, noise_std=0.3
    )
    writer.close()

    # 保存每个折的详细训练历史
    fold_results.append({
        'fold': fold + 1,
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    })

    # 获取最后一个epoch的准确率和loss
    final_accuracy = val_accuracies[-1]
    final_loss = val_losses[-1]
    fold_metrics.append({
        'accuracy': final_accuracy,
        'loss': final_loss
    })

# 计算平均指标
avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
avg_loss = np.mean([m['loss'] for m in fold_metrics])
print(f"\n五折交叉验证平均 - 准确率: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}")

# 找出表现最好的折
best_fold_idx = np.argmax([m['accuracy'] for m in fold_metrics])
best_fold = best_fold_idx + 1
print(f"\n表现最好的折: 第 {best_fold} 折")

# 使用最佳折的模型在测试集上评估
best_model = fold_results[best_fold_idx]['model']
best_model.eval()
with torch.no_grad():
    test_outputs = best_model(ftir_test, mz_test)
    _, test_pred = torch.max(test_outputs, 1)

test_accuracy = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
print(f"测试集结果 - 准确率: {test_accuracy:.4f}, 精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}, F1: {test_f1:.4f}")

# 计算每个类别的准确率
class_0_indices = (y_test == 0).nonzero(as_tuple=True)[0]
class_1_indices = (y_test == 1).nonzero(as_tuple=True)[0]
accuracy_class_0 = (test_pred[class_0_indices] == y_test[class_0_indices]).float().mean()
accuracy_class_1 = (test_pred[class_1_indices] == y_test[class_1_indices]).float().mean()
print(f'类别0准确率: {accuracy_class_0:.4f}')
print(f'类别1准确率: {accuracy_class_1:.4f}')

# 绘制混淆矩阵
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# 保存最佳模型
torch.save(best_model.state_dict(), os.path.join(save_path, 'best_model.pth'))

# 绘制所有折的训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i, fold_result in enumerate(fold_results):
    plt.plot(fold_result['train_losses'], label=f'Fold {i+1} Train Loss')
    plt.plot(fold_result['val_losses'], label=f'Fold {i+1} Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss of All Folds')
plt.legend()

plt.subplot(1, 2, 2)
for i, fold_result in enumerate(fold_results):
    plt.plot(fold_result['train_accuracies'], label=f'Fold {i+1} Train Accuracy')
    plt.plot(fold_result['val_accuracies'], label=f'Fold {i+1} Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy of All Folds')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'all_folds_training_metrics.png'))
plt.close()

print(f"\n所有结果已保存到 {save_path} 目录")
