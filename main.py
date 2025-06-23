import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import preprocess_data
from sklearn.model_selection import StratifiedGroupKFold
import random
import os
import matplotlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import itertools
import pandas as pd

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
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # [B, C, L]
        y = self.avg_pool(x).view(x.size(0), x.size(1))
        y = self.fc(y).view(x.size(0), x.size(1), 1)
        return x * y.expand_as(x)


# 定义模态特征提取的分支
class FTIREncoder(nn.Module):
    def __init__(self, input_dim):
        super(FTIREncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, 32, 7, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SEBlock(32),  # 添加 SE 注意力
            nn.MaxPool1d(3, 2),
            nn.Conv1d(32, 64, 5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(64 * 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.net(x)


class MZEncoder(nn.Module):
    def __init__(self, input_dim):
        super(MZEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, 32, 5, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SEBlock(32),
            nn.Conv1d(32, 64, 3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(64 * 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
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


class GatedFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.bias = nn.Parameter(torch.tensor([0.5, 0.5]))  # 初始为平均融合

    def forward(self, ftir_feat, mz_feat):
        combined = torch.cat([ftir_feat, mz_feat], dim=1)
        weights = self.gate(combined) * self.bias  # 加权融合
        weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化
        fused = weights[:, 0].unsqueeze(1) * ftir_feat + weights[:, 1].unsqueeze(1) * mz_feat
        return fused.squeeze(1)


class HybridFusion(nn.Module):
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        # Gate Fusion
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.gate_bias = nn.Parameter(torch.tensor([0.5, 0.5]))

        # Attention Fusion
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ftir_feat, mz_feat):
        # Gate Fusion Part
        combined_gate = torch.cat([ftir_feat, mz_feat], dim=1)
        weights = self.gate(combined_gate) * self.gate_bias
        weights = weights / weights.sum(dim=1, keepdim=True)
        gate_fused = weights[:, 0].unsqueeze(1) * ftir_feat + weights[:, 1].unsqueeze(1) * mz_feat

        # Attention Fusion Part
        ftir_seq = ftir_feat.unsqueeze(1)
        mz_seq = mz_feat.unsqueeze(1)
        cross_ftir, _ = self.attn(ftir_seq, mz_seq, mz_seq)
        cross_mz, _ = self.attn(mz_seq, ftir_seq, ftir_seq)
        attn_fused = (cross_ftir + cross_mz).squeeze(1)

        # 最终融合
        final_fused = torch.cat([gate_fused, self.proj(attn_fused)], dim=-1)  # [B, 256]
        # print("Gate weights:", weights.detach().cpu().numpy())
        return final_fused


# 多模态模型
class MultiModalModel(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim):
        super(MultiModalModel, self).__init__()
        self.ftir_extractor = FTIREncoder(input_dim=ftir_input_dim)
        self.mz_extractor = MZEncoder(input_dim=mz_input_dim)
        self.fuser = HybridFusion(dim=128, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SimpleResidualBlock(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz):
        ftir_feat = self.ftir_extractor(ftir)
        mz_feat = self.mz_extractor(mz)
        combined = self.fuser(ftir_feat, mz_feat)
        # combined = torch.cat([ftir_feat, mz_feat], dim=-1)
        output = self.classifier(combined)  # [B, 2]
        return output


# 单模态模型
class SingleFTIRModel(nn.Module):
    def __init__(self, input_dim):
        super(SingleFTIRModel, self).__init__()
        self.encoder = FTIREncoder(input_dim=input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SimpleResidualBlock(64),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return output


class SingleMZModel(nn.Module):
    def __init__(self, input_dim):
        super(SingleMZModel, self).__init__()
        self.encoder = MZEncoder(input_dim=input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SimpleResidualBlock(64),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return output


# 单模态模型评价指标函数
def evaluate_model(model, x_test, y_test, name="Model"):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(y_test, predicted)
        f1 = f1_score(y_test, predicted)
        prec = precision_score(y_test, predicted)
        rec = recall_score(y_test, predicted)
        print(f"{name} - 准确率: {acc:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}")
    return acc, prec, rec, f1


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
def data_augmentation(x, noise_std=0.1, scaling_factor=0.1, shift_range=0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    # 高斯噪声
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise
    # 随机缩放
    if abs(scaling_factor) > 1e-5:
        scale = 1 + torch.rand(1) * scaling_factor * torch.randint(-1, 2, (1,))
        x_aug = x_aug * scale.clamp(min=0.9, max=1.1)
    # 随机偏移
    if shift_range > 0:
        max_shift = int(x_aug.shape[1] * shift_range)
        if max_shift > 0:  # 关键修改点：确保 max_shift > 0
            shift = torch.randint(-max_shift, max_shift, (1,))
            x_aug = torch.roll(x_aug, shifts=shift.item(), dims=1)
    # 如果 shift_range == 0，则跳过偏移操作
    return x_aug


# ==================主模型训练====================================
# 多模态模型训练
def train_main_model(model, ftir_train, mz_train, y_train, ftir_val, mz_val, y_val, epochs, batch_size, writer):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 防止模型过于自信
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
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for ftir_batch, mz_batch, label_batch in val_dataloader:
                outputs = model(ftir_batch, mz_batch)
                loss = criterion(outputs, label_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += label_batch.size(0)
                correct += (predicted == label_batch).sum().item()
                probs = torch.softmax(outputs, dim=1)[:, 1]  # 取类别1的概率
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(label_batch.cpu().numpy())
        val_loss /= len(val_dataloader)
        val_accuracy = correct / total
        val_auc = roc_auc_score(all_targets, all_probs)
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
        writer.add_scalar('Validation AUC', val_auc, epoch)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 加载最佳模型
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    return model, train_losses, val_losses, train_accuracies, val_accuracies


# 单模态模型训练
def train_single_modal_model(model, x_train, y_train, x_val, y_val, epochs, batch_size, writer):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=10, verbose=True, path='./checkpoints/single_best_model.pth')

    train_dataset = TensorDataset(x_train, y_train)
    g = torch.Generator()
    g.manual_seed(42)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_dataset = TensorDataset(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            inputs_noisy = data_augmentation(inputs)
            outputs = model(inputs_noisy)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_dataloader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        # 写入 TensorBoard
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

    model.load_state_dict(torch.load('./checkpoints/single_best_model.pth'))
    return model, train_losses, val_losses, train_accuracies, val_accuracies


# ==================主程序====================================
def run_augmented_kfold(ftir_train, mz_train, y_train, patient_indices_train,
                        ftir_test, mz_test, y_test,
                        model_class, model_kwargs,
                        param_grid=None, n_splits=5, epochs=100, batch_size=32):
    """
    遍历 data_augmentation 参数组合，进行完整的五折交叉验证，
    并在测试集上评估最佳模型。
    """
    if param_grid is None:
        param_grid = {
            'noise_std': [0.0, 0.02, 0.05, 0.1, 0.15],
            'scale_range': [
                (0.85, 1.15),
                (0.9, 1.1),
                (0.93, 1.07),
                (0.95, 1.05)
            ],
            'shift_range': [0.0, 0.02, 0.03, 0.05, 0.07]
        }

    all_params = list(itertools.product(
        param_grid['noise_std'],
        param_grid['scale_range'],
        param_grid['shift_range']
    ))

    results = []

    for i, (noise_std, scale_range, shift_range) in enumerate(all_params):
        print(f"\n=== 数据增强实验 {i + 1}/{len(all_params)} ===")
        print(f"参数: noise_std={noise_std}, scale_range={scale_range}, shift_range={shift_range}")

        # 标准化（每个参数组合独立标准化）
        scaler_ftir = StandardScaler()
        ftir_scaled = scaler_ftir.fit_transform(ftir_train)
        ftir_test_scaled = scaler_ftir.transform(ftir_test)

        scaler_mz = StandardScaler()
        mz_scaled = scaler_mz.fit_transform(mz_train)
        mz_test_scaled = scaler_mz.transform(mz_test)

        # 转为 tensor
        full_dataset = TensorDataset(
            torch.tensor(ftir_scaled, dtype=torch.float32),
            torch.tensor(mz_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )

        test_data = TensorDataset(
            torch.tensor(ftir_test_scaled, dtype=torch.float32),
            torch.tensor(mz_test_scaled, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # 五折交叉验证
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(ftir_scaled, y_train, groups=patient_indices_train)):
            print(f"--- Fold {fold + 1} ---")

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_subsampler)
            val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_subsampler)

            model = model_class(**model_kwargs)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            early_stopping = EarlyStopping(patience=10)

            best_acc = 0.0
            best_model_state = None

            def should_apply_augmentation(noise_std, scale_range, shift_range):
                return noise_std > 0 or abs(scale_range[1] - scale_range[0]) > 1e-5 or shift_range > 0

            for epoch in range(epochs):
                model.train()
                for inputs_ftir, inputs_mz, labels in train_loader:
                    optimizer.zero_grad()
                    if should_apply_augmentation(noise_std, scale_range, shift_range):
                        ftir_noisy = data_augmentation(inputs_ftir, noise_std=noise_std,
                                                       scaling_factor=scale_range[1] - scale_range[0],
                                                       shift_range=shift_range, seed=42)
                        mz_noisy = data_augmentation(inputs_mz, noise_std=noise_std,
                                                     scaling_factor=scale_range[1] - scale_range[0],
                                                     shift_range=shift_range, seed=42)
                    else:
                        ftir_noisy = inputs_ftir
                        mz_noisy = inputs_mz
                    outputs = model(ftir_noisy, mz_noisy)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # 验证
                model.eval()
                all_preds = []
                with torch.no_grad():
                    for inputs_ftir, inputs_mz, labels in val_loader:
                        outputs = model(inputs_ftir, inputs_mz)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        all_preds.extend(preds)

                acc = accuracy_score(y_train[val_idx], all_preds)
                if acc > best_acc:
                    best_acc = acc
                    best_model_state = model.state_dict()

            print(f"Fold {fold+1} Best Acc: {best_acc:.4f}")
            fold_results.append(best_acc)

        avg_fold_acc = np.mean(fold_results)
        print(f"平均准确率: {avg_fold_acc:.4f}")

        # 测试集评估
        final_model = model_class(**model_kwargs)
        final_model.load_state_dict(best_model_state)
        final_model.eval()
        all_preds = []
        with torch.no_grad():
            for inputs_ftir, inputs_mz, _ in test_loader:
                outputs = final_model(inputs_ftir, inputs_mz)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

        test_acc = accuracy_score(y_test, all_preds)
        test_f1 = f1_score(y_test, all_preds)
        test_prec = precision_score(y_test, all_preds)
        test_rec = recall_score(y_test, all_preds)

        results.append({
            'noise_std': noise_std,
            'scale_range': str(scale_range),
            'shift_range': shift_range,
            'val_avg_acc': avg_fold_acc,
            'test_acc': test_acc,
            'test_prec': test_prec,
            'test_rec': test_rec,
            'test_f1': test_f1
        })

    # 保存结果
    df = pd.DataFrame(results)
    df.sort_values(by='test_acc', ascending=False, inplace=True)
    df.to_csv('augmentation_kfold_results.csv', index=False)
    print("✅ 完整五折实验完成，结果已保存至 augmentation_kfold_results.csv")

    return df


def run_experiments():
    # 定义你要搜索的参数空间
    param_grid = {
        'lr': [1e-4, 3e-4, 1e-3],
        'weight_decay': [0, 1e-4, 5e-4],
        'batch_size': [16, 32, 64],
        'label_smoothing': [0.0, 0.1, 0.2],
        'noise_std': [0.0, 0.05, 0.1],
        'scheduler_factor': [0.3, 0.5],
        'early_stop_patience': [10, 15]
    }

    all_params = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]

    results = []

    for i, params in enumerate(all_params):
        print(f"\n=== 实验 {i + 1}/{len(all_params)} ===")
        print("当前参数:", params)

        # 重新加载数据（确保每次实验独立）
        ftir_train, mz_train, y_train, patient_indices_train, ftir_test, mz_test, y_test, _ = preprocess_data(
            ftir_file_path, mz_file_path1, mz_file_path2, train_folder, test_folder, save_path)

        # 标准化
        scaler_ftir = StandardScaler()
        ftir_train = scaler_ftir.fit_transform(ftir_train)
        ftir_test = scaler_ftir.transform(ftir_test)
        scaler_mz = StandardScaler()
        mz_train = scaler_mz.fit_transform(mz_train)
        mz_test = scaler_mz.transform(mz_test)

        # 张量化
        ftir_train_tensor = torch.tensor(ftir_train, dtype=torch.float32)
        mz_train_tensor = torch.tensor(mz_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        ftir_test_tensor = torch.tensor(ftir_test, dtype=torch.float32)
        mz_test_tensor = torch.tensor(mz_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # 构建模型
        model = MultiModalModel(ftir_train.shape[1], mz_train.shape[1])

        # 定义损失函数、优化器等
        criterion = nn.CrossEntropyLoss(label_smoothing=params['label_smoothing'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params['scheduler_factor'], patience=3)
        early_stopping = EarlyStopping(patience=params['early_stop_patience'])

        # 构造 DataLoader
        train_dataset = TensorDataset(ftir_train_tensor, mz_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_dataset = TensorDataset(ftir_test_tensor, mz_test_tensor, y_test_tensor)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        # 简化训练过程（可替换为完整五折交叉验证）
        best_acc = 0.0
        for epoch in range(50):  # 快速训练，不跑满 200 epochs
            model.train()
            for ftir_batch, mz_batch, label_batch in train_loader:
                ftir_noisy = data_augmentation(ftir_batch, noise_std=params['noise_std'])
                mz_noisy = data_augmentation(mz_batch, noise_std=params['noise_std'])
                outputs = model(ftir_noisy, mz_noisy)
                loss = criterion(outputs, label_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 验证并记录指标
            model.eval()
            all_preds, all_probs = [], []
            with torch.no_grad():
                for ftir_batch, mz_batch, label_batch in val_loader:
                    outputs = model(ftir_batch, mz_batch)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_probs.extend(probs)

            acc = accuracy_score(y_test, all_preds)
            prec = precision_score(y_test, all_preds)
            rec = recall_score(y_test, all_preds)
            f1 = f1_score(y_test, all_preds)
            auc = roc_auc_score(y_test, all_probs)

            if acc > best_acc:
                best_acc = acc

        # 保存该组参数下的最好结果
        results.append({
            'params': str(params),
            'accuracy': best_acc,
            'precision': precision_score(y_test, all_preds),
            'recall': recall_score(y_test, all_preds),
            'f1': f1_score(y_test, all_preds),
            'auc': auc
        })
        print(f"结果: {results[-1]}")

    # 保存为 CSV
    df = pd.DataFrame(results)
    df.to_csv('hyperparameter_search_results.csv', index=False)
    print("\n✅ 所有实验完成，结果已保存至 hyperparameter_search_results.csv")

# ========================== 正式实验 ============================================
# 按患者级别实现五折交叉验证
n_splits = 5
fold_metrics = []
fold_results = []
test_metrics_list = []
# 使用GroupKFold确保同一患者所有样本在同一折
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(ftir_train, y_train, groups=patient_indices_train)):
    print(f"\n=========== 第 {fold + 1}/{n_splits} 折 ===========")
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
    # 提取训练集和验证集
    ftir_train_fold = ftir_train[train_idx]
    mz_train_fold = mz_train[train_idx]
    y_train_fold = y_train[train_idx]
    ftir_val_fold = ftir_train[val_idx]
    mz_val_fold = mz_train[val_idx]
    y_val_fold = y_train[val_idx]
    print(f"训练集标签分布: {np.bincount(y_train_fold)}")
    print(f"验证集标签分布: {np.bincount(y_val_fold)}")

    # 训练多模态模型
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

    final_accuracy = val_accuracies[-1]
    final_loss = val_losses[-1]
    fold_metrics.append({
        'accuracy': final_accuracy,
        'loss': final_loss
    })

    # 在测试集上评估该折模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(ftir_test, mz_test)
        probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()

    # 计算各项指标
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)

    # 打印并保存指标
    print(f"第 {fold + 1} 折模型在测试集上表现:")
    print(f"准确率: {acc:.4f}, F1: {f1:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, AUC: {auc:.4f}\n")

    test_metrics_list.append({
        'fold': fold + 1,
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc': auc
    })

    print("\n==== 训练 FTIR-only 模型 ====")
    ftir_model = SingleFTIRModel(ftir_train.shape[1])
    ftir_writer = SummaryWriter('./runs/ftir_only')
    ftir_model, ftir_train_losses, ftir_val_losses, ftir_train_accs, ftir_val_accs = train_single_modal_model(
        ftir_model,
        ftir_train,
        y_train,
        ftir_test,
        y_test,
        epochs=200,
        batch_size=32,
        writer=ftir_writer
    )
    ftir_writer.close()

    print("\n==== 训练 mz-only 模型 ====")
    mz_model = SingleMZModel(mz_train.shape[1])
    mz_writer = SummaryWriter('./runs/mz_only')
    mz_model, mz_train_losses, mz_val_losses, mz_train_accs, mz_val_accs = train_single_modal_model(
        mz_model,
        mz_train,
        y_train,
        mz_test,
        y_test,
        epochs=200,
        batch_size=32,
        writer=mz_writer
    )
    mz_writer.close()

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
    probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
    y_true = y_test.cpu().numpy()

test_accuracy = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
print(f"多模态网络测试集结果 - 准确率: {test_accuracy:.4f}, 精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}, F1: {test_f1:.4f}")

# 将所有折的测试集结果保存为CSV文件
df_test_metrics = pd.DataFrame(test_metrics_list)
df_test_metrics.to_csv(os.path.join(save_path, 'five_fold_test_metrics.csv'), index=False)
print("✅ 五折测试集指标已保存至 five_fold_test_metrics.csv")

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_true, probs)
roc_auc = roc_auc_score(y_true, probs)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(save_path, 'roc_curve.png'))
plt.close()

# 计算每个类别的准确率
class_0_indices = (y_test == 0).nonzero(as_tuple=True)[0]
class_1_indices = (y_test == 1).nonzero(as_tuple=True)[0]
accuracy_class_0 = (test_pred[class_0_indices] == y_test[class_0_indices]).float().mean()
accuracy_class_1 = (test_pred[class_1_indices] == y_test[class_1_indices]).float().mean()
print(f'类别0准确率: {accuracy_class_0:.4f}')
print(f'类别1准确率: {accuracy_class_1:.4f}')

# 评估 FTIR-only 模型
print("\n==== FTIR-only 模型测试结果 ====")
evaluate_model(ftir_model, ftir_test, y_test, "FTIR-only")

# 评估 mz-only 模型
print("\n==== mz-only 模型测试结果 ====")
evaluate_model(mz_model, mz_test, y_test, "mz-only")

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
'''
input_dim_ftir = ftir_train.shape[1]
input_dim_mz = mz_train.shape[1]

run_augmented_kfold(
    ftir_train=ftir_train.numpy(),
    mz_train=mz_train.numpy(),
    y_train=y_train.numpy(),
    patient_indices_train=patient_indices_train.numpy(),
    ftir_test=ftir_test.numpy(),
    mz_test=mz_test.numpy(),
    y_test=y_test.numpy(),
    model_class=MultiModalModel,
    model_kwargs={'ftir_input_dim': input_dim_ftir, 'mz_input_dim': input_dim_mz},
    epochs=100,
    batch_size=32
)
'''
"""
# 在这里加入调参逻辑
run_experiments()  # 添加这一行来启动调参实验
"""