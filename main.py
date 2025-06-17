import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import preprocess_data
from sklearn.model_selection import GroupKFold
import random
import os
import matplotlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.svm import SVC
from sklearn.metrics import classification_report

matplotlib.use('Agg')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)  # 在程序最开始调用

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
ftir_train, mz_train, y_train, patient_indices_train, ftir_test, mz_test, y_test, _ = preprocess_data(
    ftir_file_path, mz_file_path1,
    mz_file_path2, train_folder,
    test_folder, save_path)
# 打印训练集和测试集的类别分布
print("训练集类别分布:", np.bincount(y_train))
print("测试集类别分布:", np.bincount(y_test))

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
patient_indices_train = torch.tensor(patient_indices_train, dtype=torch.long)


# ==================主模型定义====================================
# 定义模态特征提取的分支
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(16 * 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class SimpleResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return x + self.net(x)


# 完整的模型
class MultiModalModel(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim):
        super(MultiModalModel, self).__init__()
        self.ftir_extractor = FeatureExtractor(input_dim=ftir_input_dim)
        self.mz_extractor = FeatureExtractor(input_dim=mz_input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SimpleResidualBlock(64),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz):
        ftir_feat = self.ftir_extractor(ftir)
        mz_feat = self.mz_extractor(mz)
        combined = torch.cat([ftir_feat, mz_feat], dim=-1)  # [B, ftir_dim + mz_dim]
        output = self.classifier(combined)  # [B, 2]
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
def data_augmentation(x, noise_std=0.1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    # 高斯噪声
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise
    return x_aug


# ==================主模型训练====================================
def train_main_model(model, ftir_train, mz_train, y_train, ftir_val, mz_val, y_val, epochs, batch_size, writer):
    class_counts = np.bincount(y_train_fold.numpy())
    weights = 1. / class_counts
    weights = torch.tensor([1.0, 2.0], dtype=torch.float)  # 给类别1更高的权重

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)  # 防止模型过于自信
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=10, verbose=True, path='./checkpoints/best_model.pth')

    train_dataset = TensorDataset(ftir_train, mz_train, y_train)
    g = torch.Generator()
    g.manual_seed(42)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
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
        scheduler.step(val_loss)  # 根据验证损失更新学习率

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

# 使用GroupKFold确保同一患者所有样本在同一折
gkf = GroupKFold(n_splits=n_splits)
for fold in range(n_splits):
    print(f"\n=========== 第 {fold + 1}/{n_splits} 折 ===========")
    # 获取当前折的训练/验证样本索引
    train_idx, val_idx = next(gkf.split(np.arange(len(y_train)), groups=patient_indices_train))
    # 提取对应的患者ID
    train_patients = patient_indices_train[train_idx]
    val_patients = patient_indices_train[val_idx]
    # 使用 np.unique 来去重并比较数量
    train_patients_unique = np.unique(train_patients)
    val_patients_unique = np.unique(val_patients)
    # 确保训练集与验证集无交集
    assert len(set(train_patients_unique) & set(val_patients_unique)) == 0, "患者跨折泄漏"
    print(f"Fold {fold} 训练患者ID: {train_patients_unique}")
    print(f"Fold {fold} 验证患者ID: {val_patients_unique}")
    # 提取数据
    train_mask = np.isin(patient_indices_train, train_patients)
    val_mask = np.isin(patient_indices_train, val_patients)
    # 提取数据
    ftir_train_fold = ftir_train[train_mask]
    mz_train_fold = mz_train[train_mask]
    y_train_fold = y_train[train_mask]
    patient_indices_train_fold = patient_indices_train[train_mask]
    ftir_val_fold = ftir_train[val_mask]
    mz_val_fold = mz_train[val_mask]
    y_val_fold = y_train[val_mask]

    # 训练模型
    writer = SummaryWriter(f'./runs/fold_{fold + 1}')
    model = MultiModalModel(ftir_train_fold.shape[1], mz_train_fold.shape[1])
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_main_model(
        model,
        ftir_train_fold, mz_train_fold, y_train_fold,
        ftir_val_fold, mz_val_fold, y_val_fold,
        epochs=200,
        batch_size=32,
        writer=writer
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


# 和SVM作对比，看看是当前神经网络效果如何
train_features = np.hstack([ftir_train.numpy(), mz_train.numpy()])
test_features = np.hstack([ftir_test.numpy(), mz_test.numpy()])
y_labels = y_train.numpy()
y_test_labels = y_test.numpy()

# 训练 SVM 分类器
clf = SVC(kernel='rbf', probability=True)
clf.fit(train_features, y_labels)

# 预测并输出报告
preds = clf.predict(test_features)
print("\nSVM 分类结果:")
print(classification_report(y_test_labels, preds))


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
    plt.plot(fold_result['train_losses'], label=f'Fold {i + 1} Train Loss')
    plt.plot(fold_result['val_losses'], label=f'Fold {i + 1} Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss of All Folds')
plt.legend()

plt.subplot(1, 2, 2)
for i, fold_result in enumerate(fold_results):
    plt.plot(fold_result['train_accuracies'], label=f'Fold {i + 1} Train Accuracy')
    plt.plot(fold_result['val_accuracies'], label=f'Fold {i + 1} Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy of All Folds')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'all_folds_training_metrics.png'))
plt.close()

print(f"\n所有结果已保存到 {save_path} 目录")
