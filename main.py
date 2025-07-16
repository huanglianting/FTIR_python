import random
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from data_preprocessing import preprocess_data
from sklearn.model_selection import StratifiedGroupKFold
from evaluation import evaluate_model
from Multi_Single_modal import MultiModalModel, SingleFTIRModel, SingleMZModel, ConcatFusion, GateOnlyFusion, \
    CoAttnOnlyFusion, SelfAttnOnlyFusion, SelfAttnFusion, SVMClassifier

matplotlib.use('Agg')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(3)  # 枚举到13。在程序最开始调用。best：4。

# 定义基础路径
ftir_file_path = './data/'
mz_file_path1 = r'./data/compound_measurements.xlsx'
mz_file_path2 = r'./data/compound_measurements2.xlsx'
save_path = './result'  # 保存图片的路径
if not os.path.exists(save_path):
    os.makedirs(save_path)
os.makedirs('./checkpoints', exist_ok=True)

# 调用预处理函数
train_folder = os.path.join(save_path, 'train')
test_folder = os.path.join(save_path, 'test')
ftir_train, mz_train, y_train, patient_indices_train, ftir_test, mz_test, y_test, patient_indices_test, ftir_x, mz_x = preprocess_data(
    ftir_file_path, mz_file_path1,
    mz_file_path2, train_folder,
    test_folder, save_path)

print(ftir_train.shape)  # (768, 467)
print(mz_train.shape)  # (768, 2838)
print(y_train.shape)  # (768,)
print(ftir_x.shape)  # (467,)
print(mz_x.shape)  # (2838,)
# 打印训练集和测试集的类别分布
print("训练集类别分布:", np.bincount(y_train))
print("测试集类别分布:", np.bincount(y_test))

# 数据标准化
scaler_ftir = StandardScaler()
ftir_train = scaler_ftir.fit_transform(ftir_train)
ftir_test = scaler_ftir.transform(ftir_test)
ftir_x_scaled = scaler_ftir.transform(ftir_x.reshape(1, -1)).squeeze()
scaler_mz = StandardScaler()
mz_train = scaler_mz.fit_transform(mz_train)
mz_test = scaler_mz.transform(mz_test)
mz_x_scaled = scaler_mz.transform(mz_x.reshape(1, -1)).squeeze()

# 转换为PyTorch张量
ftir_train = torch.tensor(ftir_train, dtype=torch.float32)
mz_train = torch.tensor(mz_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
ftir_test = torch.tensor(ftir_test, dtype=torch.float32)
mz_test = torch.tensor(mz_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
ftir_x = torch.tensor(ftir_x_scaled, dtype=torch.float32)  # 形状: [467]
mz_x = torch.tensor(mz_x_scaled, dtype=torch.float32)      # 形状: [2838]
patient_indices_train = torch.tensor(patient_indices_train, dtype=torch.long)

# 验证标准化后的数据形状
print("标准化后 ftir_train 形状:", ftir_train.shape)
print("标准化后 mz_train 形状:", mz_train.shape)
print("标准化后 ftir_test 形状:", ftir_test.shape)
print("标准化后 mz_test 形状:", mz_test.shape)
print("标准化后 ftir_x 形状:", ftir_x.shape)
print("标准化后 mz_x 形状:", mz_x.shape)


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
            print(
                f'Validation loss 下降 ({self.val_loss_min:.6f} --> {val_loss:.6f}).  保存模型 ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ==================数据增强====================================
def data_augmentation(x, axis, noise_std=0.1, scaling_factor=0.05, shift_range=0.02, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    B, L = x.shape  # 批量大小和特征长度
    axis = axis.squeeze().expand(B, -1)
    # 高斯噪声
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise
    # 随机缩放
    scale = 1 + (torch.rand(B, 1, device=x.device) * 2 - 1) * scaling_factor
    x_aug = x_aug * scale
    # 随机偏移
    max_shift = int(L * shift_range)
    shifts = torch.randint(-max_shift, max_shift+1, (B,), device=x.device)
    x_aug = torch.stack([
        torch.roll(x_aug[i], shifts=shifts[i].item(), dims=-1)
        for i in range(B)
    ])
    axis = axis + (shifts.float() / L).unsqueeze(1)
    return x_aug, axis


# ==================主模型训练====================================
# 多模态模型训练
def train_main_model(model, ftir_train, mz_train, y_train, ftir_val, mz_val, y_val,
                     ftir_axis, mz_axis, epochs, batch_size, writer,
                     lr=3e-4, weight_decay=1e-4, label_smoothing=0.1,
                     scheduler_factor=0.5, early_stop_patience=10, model_type='undefined'):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=3)
    early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True,
                                   path=f'./checkpoints/{model_type}_best_model.pth')

    train_dataset = TensorDataset(ftir_train, mz_train, y_train)
    g = torch.Generator()
    g.manual_seed(42)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_dataset = TensorDataset(ftir_val, mz_val, y_val)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

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
            ftir_noisy, ftir_axis = data_augmentation(ftir_batch, ftir_axis)
            mz_noisy, mz_axis = data_augmentation(mz_batch, mz_axis)
            outputs = model(ftir_noisy, mz_noisy, ftir_axis, mz_axis)
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
                outputs = model(ftir_batch, mz_batch, ftir_axis, mz_axis)
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

    model.load_state_dict(torch.load(
        f'./checkpoints/{model_type}_best_model.pth'))
    return model, train_losses, val_losses, train_accuracies, val_accuracies


# 单模态模型训练
def train_single_modal_model(model, x_train, y_train, x_val, y_val, axis,
                             epochs, batch_size, writer,
                             lr=3e-4, weight_decay=1e-4, label_smoothing=0.1,
                             scheduler_factor=0.5, early_stop_patience=10, model_type='undefined'):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=3)
    early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True,
                                   path=f'./checkpoints/{model_type}_best_model.pth')

    train_dataset = TensorDataset(x_train, y_train)
    g = torch.Generator()
    g.manual_seed(42)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_dataset = TensorDataset(x_val, y_val)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

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
            inputs_noisy, axis = data_augmentation(inputs, axis)
            outputs = model(inputs_noisy, axis)
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
                outputs = model(inputs, axis)
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

    model.load_state_dict(torch.load(
        f'./checkpoints/{model_type}_best_model.pth'))
    return model, train_losses, val_losses, train_accuracies, val_accuracies


# ==================主程序====================================
# 按患者级别实现四折交叉验证
n_splits = 4
# 使用GroupKFold确保同一患者所有样本在同一折
sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)

param_grid = {
    'lr': [3e-4],
    'weight_decay': [1e-4],
    'batch_size': [32],
    'label_smoothing': [0.1],
    'scheduler_factor': [0.5],
    'early_stop_patience': [15]
}
all_params = [dict(zip(param_grid.keys(), values))
              for values in itertools.product(*param_grid.values())]

best_avg_auc = 0.0
best_params = None
grid_search_results = []  # 用于保存所有参数的四折结果


def run_grid_search_for_model(model_name, model_class, ftir_train, mz_train, y_train, ftir_axis, mz_axis,
                              patient_indices_train, param_grid):
    all_params = [dict(zip(param_grid.keys(), values))
                  for values in itertools.product(*param_grid.values())]
    results = []
    for params in all_params:
        print(f"\n=== [{model_name}] 测试参数组合: {params} ===")
        fold_accuracies = []
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(ftir_train, y_train, groups=patient_indices_train)):
            print(f"\n=========== 第 {fold + 1}/{n_splits} 折 ===========")
            # 提取对应的患者ID
            train_patients = patient_indices_train[train_idx]
            val_patients = patient_indices_train[val_idx]
            # 使用 np.unique 来去重并比较数量
            train_patients_unique = np.unique(train_patients)
            val_patients_unique = np.unique(val_patients)
            # 确保训练集与验证集无交集
            assert len(set(train_patients_unique) & set(
                val_patients_unique)) == 0, "患者跨折泄漏"
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

            if model_name == "MultiModal":
                model = MultiModalModel(
                    ftir_train_fold.shape[1], mz_train_fold.shape[1])
                writer = SummaryWriter(
                    f'./runs/gridsearch/{model_name}_fold{fold + 1}')
                trained_model, _, _, _, val_accs = train_main_model(
                    model,
                    ftir_train_fold, mz_train_fold, y_train_fold,
                    ftir_val_fold, mz_val_fold, y_val_fold,
                    ftir_axis, mz_axis,
                    epochs=100,
                    batch_size=params['batch_size'],
                    writer=writer,
                    lr=params['lr'],
                    weight_decay=params['weight_decay'],
                    label_smoothing=params['label_smoothing'],
                    scheduler_factor=params['scheduler_factor'],
                    early_stop_patience=params['early_stop_patience'],
                    model_type=model_name
                )
                writer.close()

            elif model_name == "FTIROnly":
                model = SingleFTIRModel(ftir_train.shape[1])
                writer = SummaryWriter(
                    f'./runs/gridsearch/{model_name}_fold{fold + 1}')
                trained_model, _, _, _, val_accs = train_single_modal_model(
                    model,
                    ftir_train_fold, y_train_fold,
                    ftir_val_fold, y_val_fold,
                    ftir_axis,
                    epochs=100,
                    batch_size=params['batch_size'],
                    writer=writer,
                    lr=params['lr'],
                    weight_decay=params['weight_decay'],
                    label_smoothing=params['label_smoothing'],
                    scheduler_factor=params['scheduler_factor'],
                    early_stop_patience=params['early_stop_patience'],
                    model_type=model_name
                )
                writer.close()

            elif model_name == "MZOnly":
                model = SingleMZModel(mz_train.shape[1])
                writer = SummaryWriter(
                    f'./runs/gridsearch/{model_name}_fold{fold + 1}')
                trained_model, _, _, _, val_accs = train_single_modal_model(
                    model,
                    mz_train_fold, y_train_fold,
                    mz_val_fold, y_val_fold,
                    mz_axis,
                    epochs=100,
                    batch_size=params['batch_size'],
                    writer=writer,
                    lr=params['lr'],
                    weight_decay=params['weight_decay'],
                    label_smoothing=params['label_smoothing'],
                    scheduler_factor=params['scheduler_factor'],
                    early_stop_patience=params['early_stop_patience'],
                    model_type=model_name
                )
                writer.close()

            elif "svm" in model_name.lower():
                train_features = np.hstack([ftir_train_fold.numpy(), mz_train_fold.numpy()]) \
                    if (isinstance(ftir_train_fold, torch.Tensor) and isinstance(mz_train_fold, torch.Tensor)) \
                    else np.hstack([ftir_train_fold, mz_train_fold])
                test_features = np.hstack([ftir_val_fold.numpy(), mz_val_fold.numpy()]) \
                    if (isinstance(ftir_val_fold, torch.Tensor) and isinstance(mz_val_fold, torch.Tensor)) \
                    else np.hstack([ftir_val_fold, mz_val_fold])
                model = SVMClassifier(kernel='rbf')
                model.fit(train_features, y_train_fold.numpy())
                preds = model.predict(test_features)
                probs = model.predict_proba(test_features)[:, 1]
                metrics = evaluate_model(model, ftir_test, mz_test, y_test, ftir_axis, mz_axis,
                                         name=model_name, model_type=model_name, is_svm=True)
                val_accs = [metrics['accuracy']]

            elif "fusion" in model_name.lower():
                # ConcatFusion、GateOnlyFusion、SelfAttnOnlyFusion 等
                model = model_class(
                    ftir_input_dim=ftir_train_fold.shape[1], mz_input_dim=mz_train_fold.shape[1])
                writer = SummaryWriter(
                    f'./runs/gridsearch/{model_name}_fold{fold + 1}')
                trained_model, _, _, _, val_accs = train_main_model(
                    model,
                    ftir_train_fold, mz_train_fold, y_train_fold,
                    ftir_val_fold, mz_val_fold, y_val_fold,
                    ftir_axis, mz_axis,
                    epochs=100,
                    batch_size=params['batch_size'],
                    writer=writer,
                    lr=params['lr'],
                    weight_decay=params['weight_decay'],
                    label_smoothing=params['label_smoothing'],
                    scheduler_factor=params['scheduler_factor'],
                    early_stop_patience=params['early_stop_patience'],
                    model_type=model_name
                )
                writer.close()
            best_acc = max(val_accs) if len(val_accs) > 0 else 0
            fold_accuracies.append(best_acc)
        avg_acc = np.mean(fold_accuracies)
        results.append({
            'model_type': model_name,
            'params': str(params),
            'avg_accuracy': avg_acc
        })
    return pd.DataFrame(results)


# 对所有模型，利用 k-fold 交叉验证调参，确定最优参数
models_to_evaluate = {
    "MultiModal": MultiModalModel,
    # "FTIROnly": SingleFTIRModel,
    # "MZOnly": SingleMZModel,
    # "ConcatFusion": ConcatFusion,
    # "GateOnlyFusion": GateOnlyFusion,
    # "CoAttnOnlyFusion": CoAttnOnlyFusion,
    # "SelfAttnOnlyFusion": SelfAttnOnlyFusion,
    # "SelfAttnFusion": SelfAttnFusion,
    # "SVM": SVMClassifier
}
all_model_dfs = []
for model_name, model_class in models_to_evaluate.items():
    print(f"\n\n 开始评估模型: {model_name}")
    if model_name == "SVM":
        pass
    else:
        df = run_grid_search_for_model(model_name, model_class, ftir_train, mz_train, y_train,
                                       ftir_x, mz_x, patient_indices_train, param_grid)
    all_model_dfs.append(df)
    # 合并所有模型结果
    all_results_df = pd.concat(all_model_dfs, ignore_index=True)
    all_results_df.to_csv(os.path.join(
        save_path, 'all_models_grid_search_results.csv'), index=False)
    print("所有模型 Grid Search 结果已保存至 all_models_grid_search_results.csv")

# 加载 Grid Search 结果
all_results_df = pd.read_csv(os.path.join(
    save_path, 'all_models_grid_search_results.csv'))
# 找出每个模型的最佳参数（按 avg_accuracy）
best_params_per_model = {}
for model_type in all_results_df['model_type'].unique():
    df_model = all_results_df[all_results_df['model_type'] == model_type]
    # 按 accuracy 找最优
    best_row = df_model.loc[df_model['avg_accuracy'].idxmax()]
    best_params = eval(best_row['params'])  # 字符串转字典
    best_params_per_model[model_type] = best_params
    print(f"[{model_type}] 最佳参数: {best_params}")

# 最后，使用最佳参数重新训练并在测试集上评估
final_test_results = []
training_history = {}
for model_name, params in best_params_per_model.items():
    print(f"\n=== 使用最优参数训练并评估模型: {model_name} ===")
    if model_name == "MultiModal":
        model = MultiModalModel(
            ftir_input_dim=ftir_train.shape[1], mz_input_dim=mz_train.shape[1])
        writer = SummaryWriter(f'./runs/final_{model_name}')
        trained_model, train_losses, test_losses, train_accuracies, test_accuracies = train_main_model(
            model,
            ftir_train, mz_train, y_train,
            ftir_test, mz_test, y_test,
            ftir_x, mz_x,
            epochs=100,
            batch_size=params['batch_size'],
            writer=writer,
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            label_smoothing=params['label_smoothing'],
            scheduler_factor=params['scheduler_factor'],
            early_stop_patience=params['early_stop_patience'],
            model_type=model_name
        )
        writer.close()
        metrics = evaluate_model(trained_model, ftir_test, mz_test, y_test, ftir_x, mz_x,
                                 name=model_name, model_type=model_name)

    elif model_name == "FTIROnly":
        model = SingleFTIRModel(input_dim=ftir_train.shape[1])
        writer = SummaryWriter(f'./runs/final_ftir_only')
        trained_model, train_losses, test_losses, train_accuracies, test_accuracies = train_single_modal_model(
            model,
            ftir_train, y_train,
            ftir_test, y_test,
            ftir_x,
            epochs=100,
            batch_size=params['batch_size'],
            writer=writer,
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            label_smoothing=params['label_smoothing'],
            scheduler_factor=params['scheduler_factor'],
            early_stop_patience=params['early_stop_patience'],
            model_type=model_name
        )
        writer.close()
        metrics = evaluate_model(trained_model, ftir_test, None, y_test, ftir_x, mz_x,
                                 name=model_name, model_type=model_name)

    elif model_name == "MZOnly":
        model = SingleMZModel(input_dim=mz_train.shape[1])
        writer = SummaryWriter(f'./runs/final_mz_only')
        trained_model, train_losses, test_losses, train_accuracies, test_accuracies = train_single_modal_model(
            model,
            mz_train, y_train,
            mz_test, y_test,
            mz_x,
            epochs=100,
            batch_size=params['batch_size'],
            writer=writer,
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            label_smoothing=params['label_smoothing'],
            scheduler_factor=params['scheduler_factor'],
            early_stop_patience=params['early_stop_patience'],
            model_type=model_name
        )
        writer.close()
        metrics = evaluate_model(trained_model, None, mz_test, y_test, ftir_x, mz_x,
                                 name=model_name, model_type=model_name)

    elif model_name == "SVM":
        # 将 ftir_axis 和 mz_axis 扩展为与当前数据相同的 batch 维度，并拼接至特征维度
        train_features_with_axis = np.hstack([
            ftir_train.numpy(), mz_train.numpy(),
            ftir_x.repeat(ftir_train.shape[0], 1),  # [batch_size, 467]
            mz_x.repeat(mz_train.shape[0], 1)  # [batch_size, 2838]
        ])
        test_features_with_axis = np.hstack([
            ftir_test.numpy(), mz_test.numpy(),
            ftir_x.repeat(ftir_test.shape[0], 1).numpy(),
            mz_x.repeat(mz_test.shape[0], 1).numpy()
        ])
        model = SVMClassifier(kernel='rbf')
        model.fit(train_features_with_axis, y_train.numpy())
        preds = model.predict(test_features_with_axis)
        probs = model.predict_proba(test_features_with_axis)[:, 1]
        metrics = evaluate_model(model, ftir_test, mz_test, y_test, ftir_x, mz_x,
                                 name=model_name, model_type=model_name, is_svm=True)
        continue

    else:
        model_class = eval(model_name)
        model = model_class(
            ftir_input_dim=ftir_train.shape[1], mz_input_dim=mz_train.shape[1])
        writer = SummaryWriter(f'./runs/final_{model_name}')
        trained_model, train_losses, test_losses, train_accuracies, test_accuracies = train_main_model(
            model,
            ftir_train, mz_train, y_train,
            ftir_test, mz_test, y_test,
            ftir_x, mz_x,
            epochs=100,
            batch_size=params['batch_size'],
            writer=writer,
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            label_smoothing=params['label_smoothing'],
            scheduler_factor=params['scheduler_factor'],
            early_stop_patience=params['early_stop_patience'],
            model_type='fusion'
        )
        writer.close()
        metrics = evaluate_model(trained_model, ftir_test, mz_test, y_test, ftir_x, mz_x,
                                 name=model_name, model_type=model_name)

    training_history[model_name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }
    final_test_results.append({
        'model_type': model_name,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'f1': metrics['f1'],
        'auc': metrics['auc']
    })

# 导出最终结果
df_final = pd.DataFrame(final_test_results)
df_final.to_csv(os.path.join(
    save_path, 'final_test_all_models_comparison.csv'), index=False)
print("所有模型最终测试结果已保存至 final_test_all_models_comparison.csv")

# 绘制每个模型 使用最优参数 在训练和测试时 的 loss 和 accuracy 曲线
plot_dir = os.path.join(save_path, 'training_plots')
os.makedirs(plot_dir, exist_ok=True)
soft_blue = '#6495ED'  # 柔和的蓝色
soft_red = '#CD5C5C'  # 柔和的红色
for model_name, data in training_history.items():
    # 绘制 Loss 曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(data['train_losses'], label='Train Loss',
             color=soft_blue, linestyle='-')
    plt.plot(data['test_losses'], label='Test Loss',
             color=soft_red, linestyle='-')
    plt.title(f'{model_name} - Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    plt.legend(loc='upper right')  # 设置固定位置的图例

    # 设置坐标轴为黑色实线
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)  # 加粗坐标轴
    # 设置刻度小短线
    ax.tick_params(axis='both', which='major',
                   length=5, width=1, direction='out')
    # 设置 x 轴刻度（每 5 个 epoch 显示一个）
    plt.xticks(np.arange(0, 30, 5))
    # 固定 y 轴范围并确保起始刻度可以标注
    plt.ylim(0.4, 0.7)  # 固定 y 轴范围
    plt.yticks(np.arange(0.4, 0.71, 0.05))
    plt.grid(False)
    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(data['train_accuracies'], label='Train Accuracy',
             color=soft_blue, linestyle='-')
    plt.plot(data['test_accuracies'], label='Test Accuracy',
             color=soft_red, linestyle='-')
    plt.title(f'{model_name} - Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')  # 设置固定位置的图例
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)  # 加粗坐标轴
    ax.tick_params(axis='both', which='major', length=5,
                   width=1, direction='out')  # 设置刻度小短线
    # 设置 x 轴刻度（每 5 个 epoch 显示一个）
    plt.xticks(np.arange(0, 30, 5))
    # 固定 y 轴范围并确保起始刻度可以标注
    plt.ylim(0.5, 1.0)
    plt.yticks(np.arange(0.6, 0.91, 0.05))
    plt.grid(False)
    """
    # 添加网格线
    ax.grid(True, linestyle='-', color='#EEEEEE', linewidth=0.5)  # 浅色实线网格
    ax.xaxis.set_minor_locator(MultipleLocator(1))  # 每 1 个 epoch 一个次刻度
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))  # 每 0.01 一个次刻度
    ax.grid(True, which='major', linestyle='-', color='#CCCCCC', linewidth=0.6)  # 主网格线
    ax.grid(True, which='minor', linestyle=':', color='#DDDDDD', linewidth=0.4)  # 次网格线
    """
    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(os.path.join(
        plot_dir, f'{model_name}_loss_accuracy_curve.png'))
    plt.close()

print(f"所有模型的 loss 和 accuracy 曲线已保存至 {plot_dir}")
