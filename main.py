import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import preprocess_data
from sklearn.model_selection import StratifiedGroupKFold
import random
import os
import matplotlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
import pandas as pd
from evaluation import evaluate_model
from Multi_Single_modal import MultiModalModel, SingleFTIRModel, SingleMZModel, ConcatFusion, GateOnlyFusion, CoAttnOnlyFusion, SelfAttnOnlyFusion, SVMClassifier


matplotlib.use('Agg')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(4)  # 枚举到13。在程序最开始调用。best：4。

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
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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
def data_augmentation(x, noise_std=0.1, scaling_factor=0.1, shift_range=0.05, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    # 高斯噪声
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise
    # 随机缩放
    scale = 1 + torch.rand(1) * scaling_factor * torch.randint(-1, 2, (1,))
    x_aug= x_aug * scale.clamp(min=0.9, max=1.1)
    # 随机偏移
    shift = torch.randint(-int(x_aug.shape[1] * shift_range), int(x_aug.shape[1] * shift_range), (1,))
    x_aug = torch.roll(x_aug, shifts=shift.item(), dims=1)
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
        for epoch in range(50):  # 快速训练，不跑满 100 epochs
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


def train_and_evaluate_model(model_name, model_class, ftir_train, mz_train, y_train, ftir_test, mz_test, y_test, is_svm=False):
    print(f"\n=== 训练 {model_name} ===")
    if is_svm:
        # 特征拼接用于 SVM
        train_features = np.hstack([ftir_train.numpy(), mz_train.numpy()])
        test_features = np.hstack([ftir_test.numpy(), mz_test.numpy()])
        model = model_class(kernel='rbf')
        model.fit(train_features, y_train.numpy())
        preds = model.predict(test_features)
        probs = model.predict_proba(test_features)[:, 1]
    else:
        writer = SummaryWriter(f'./runs/ablation_{model_name}')
        model = model_class(ftir_input_dim=ftir_train.shape[1], mz_input_dim=mz_train.shape[1])
        trained_model, _, _, _, _ = train_main_model(
            model,
            ftir_train, mz_train, y_train,
            ftir_test, mz_test, y_test,
            epochs=100,
            batch_size=32,
            writer=writer
        )
        writer.close()
        # 测试集评估
        model.eval()
        with torch.no_grad():
            outputs = model(ftir_test, mz_test)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # 统一评估
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"{model_name} 测试集表现:")
    print(f"准确率: {acc:.4f}, F1: {f1:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, AUC: {auc:.4f}\n")

    return {
        'model': model_name,
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc': auc
    }


# 按患者级别实现五折交叉验证
n_splits = 5
fold_metrics = []
fold_results = []
ftir_only_results = []
mz_only_results = []
multi_modal_results = []
all_model_results = []
ablation_all_results = []
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
        epochs=100,
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
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    all_model_results.append({
        'fold': fold + 1,
        'model_type': 'MultiModal',
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc': auc
    })

    # 训练 FTIR-only 模型
    ftir_model = SingleFTIRModel(ftir_train.shape[1])
    ftir_writer = SummaryWriter('./runs/ftir_only')
    ftir_model, ftir_train_losses, ftir_val_losses, ftir_train_accs, ftir_val_accs = train_single_modal_model(
        ftir_model,
        ftir_train,
        y_train,
        ftir_test,
        y_test,
        epochs=100,
        batch_size=32,
        writer=ftir_writer
    )
    ftir_writer.close()
    ftir_model.eval()
    with torch.no_grad():
        test_outputs = ftir_model(ftir_test)
        probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()
    acc_ftir = accuracy_score(y_true, preds)
    f1_ftir = f1_score(y_true, preds)
    prec_ftir = precision_score(y_true, preds)
    rec_ftir = recall_score(y_true, preds)
    auc_ftir = roc_auc_score(y_true, probs)
    all_model_results.append({
        'fold': fold + 1,
        'model_type': 'FTIR-only',
        'accuracy': acc_ftir,
        'f1': f1_ftir,
        'precision': prec_ftir,
        'recall': rec_ftir,
        'auc': auc_ftir
    })

    # 训练 mz-only 模型
    mz_model = SingleMZModel(mz_train.shape[1])
    mz_writer = SummaryWriter('./runs/mz_only')
    mz_model, mz_train_losses, mz_val_losses, mz_train_accs, mz_val_accs = train_single_modal_model(
        mz_model,
        mz_train,
        y_train,
        mz_test,
        y_test,
        epochs=100,
        batch_size=32,
        writer=mz_writer
    )
    mz_writer.close()
    mz_model.eval()
    with torch.no_grad():
        test_outputs = mz_model(mz_test)
        probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()
    acc_mz = accuracy_score(y_true, preds)
    f1_mz = f1_score(y_true, preds)
    prec_mz = precision_score(y_true, preds)
    rec_mz = recall_score(y_true, preds)
    auc_mz = roc_auc_score(y_true, probs)
    all_model_results.append({
        'fold': fold + 1,
        'model_type': 'mz-only',
        'accuracy': acc_mz,
        'f1': f1_mz,
        'precision': prec_mz,
        'recall': rec_mz,
        'auc': auc_mz
    })

    # 训练其他消融试验模型
    ablation_models_for_cv = {
        "ConcatFusion": ConcatFusion,
        "GateOnlyFusion": GateOnlyFusion,
        "CoAttnOnlyFusion": CoAttnOnlyFusion,
        "SelfAttnOnlyFusion": SelfAttnOnlyFusion,
        "SVM": SVMClassifier
    }
    fold_ablation_results = []
    for name, model_class in ablation_models_for_cv.items():
        is_svm = name == "SVM"
        result = train_and_evaluate_model(
            model_name=name,
            model_class=model_class,
            ftir_train=ftir_train_fold,
            mz_train=mz_train_fold,
            y_train=y_train_fold,
            ftir_test=ftir_val_fold,
            mz_test=mz_val_fold,
            y_test=y_val_fold,
            is_svm=is_svm
        )
        result['fold'] = fold + 1
        fold_ablation_results.append(result)
    all_model_results.extend(fold_ablation_results)

# 导出所有模型结果到一张表中
df_all = pd.DataFrame(all_model_results)
df_all.to_csv(os.path.join(save_path, 'five_fold_all_models_comparison.csv'), index=False)
print("所有模型五折结果已合并保存至 five_fold_all_models_comparison.csv")


# 绘制-多模态网络模型-所有折的训练曲线
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


# 找出表现最好的折
best_fold_idx = np.argmax([m['accuracy'] for m in fold_metrics])
best_fold = best_fold_idx + 1
print(f"\n表现最好的折: 第 {best_fold} 折")

# 使用最佳折的模型在测试集上评估
best_model = fold_results[best_fold_idx]['model']
print(f"多模态网络测试集结果:")
multi_metrics = evaluate_model(best_model, (ftir_test, mz_test), y_test, name="MultiModal")
# 保存最佳模型
torch.save(best_model.state_dict(), os.path.join(save_path, 'best_model.pth'))

"""
# 在这里加入调参逻辑
run_experiments()  # 添加这一行来启动调参实验
"""