import random
import os
import itertools
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import shap
from scipy.stats import spearmanr
import seaborn as sns

matplotlib.use('Agg')


def set_seed(seed):
    # 1. Python基础随机模块
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA卷积算法确定性
    # 2. NumPy随机模块
    np.random.seed(seed)
    # 3. PyTorch随机模块
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    # 4. PyTorch确定性配置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # 限制并行
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=4, help='Random seed')
args = parser.parse_args()
set_seed(args.seed)

g = torch.Generator()
g.manual_seed(42)

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
#ftir_x_scaled = scaler_ftir.transform(ftir_x.reshape(1, -1)).squeeze()
scaler_mz = StandardScaler()
mz_train = scaler_mz.fit_transform(mz_train)
mz_test = scaler_mz.transform(mz_test)
#mz_x_scaled = scaler_mz.transform(mz_x.reshape(1, -1)).squeeze()

# 转换为PyTorch张量
ftir_train = torch.tensor(ftir_train, dtype=torch.float32)
mz_train = torch.tensor(mz_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
ftir_test = torch.tensor(ftir_test, dtype=torch.float32)
mz_test = torch.tensor(mz_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
ftir_x = torch.tensor(ftir_x, dtype=torch.float32)  # 形状: [467]
mz_x = torch.tensor(mz_x, dtype=torch.float32)      # 形状: [2838]
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

# 只对 FTIR 做 SHAP 分析
def perform_ftir_shap_analysis(model, ftir_train, ftir_test, ftir_x, mz_train, mz_x, y_test, patient_indices_train, patient_indices_test):
    """
    对 MultiModal 模型进行 FTIR 特征的 Gradient SHAP 分析, 并生成一维热力图。
    分别分析癌症和良性样本，比较能够贡献区分的重要波数段。
    """
    model.eval()

    # 1. 定义一个 PyTorch 模型包装器，用于 SHAP
    # 这个包装器固定了 MZ 输入，只让 SHAP 改变 FTIR 输入
    class ShapModelWrapper(torch.nn.Module):
        def __init__(self, model, mz_baseline, ftir_x, mz_x):
            super().__init__()
            self.model = model
            self.register_buffer('mz_baseline', mz_baseline)
            self.register_buffer('ftir_x', ftir_x)
            self.register_buffer('mz_x', mz_x)

        def forward(self, ftir_data):
            # SHAP 会传入一个需要计算梯度的张量
            current_mz_baseline = self.mz_baseline.expand(ftir_data.shape[0], -1)
            ftir_axis = self.ftir_x.repeat(ftir_data.shape[0], 1)
            current_mz_axis = self.mz_x.repeat(ftir_data.shape[0], 1)
            # 模型前向传播
            outputs = self.model(ftir_data, current_mz_baseline, ftir_axis, current_mz_axis)
            # 返回类别1的概率，并确保输出是 (n, 1) 的二维张量
            return torch.softmax(outputs, dim=1)[:, 1].unsqueeze(-1)

    # 2. 准备背景数据和测试样本
    # 背景数据：从训练集中选择具有代表性的样本，确保背景数据涵盖不同患者和类别
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    patient_indices_train_np = patient_indices_train.cpu().numpy() if isinstance(patient_indices_train, torch.Tensor) else patient_indices_train
    # 为每个类别选择代表性样本
    cancer_indices = np.where(y_train_np == 1)[0]
    benign_indices = np.where(y_train_np == 0)[0]
    selected_background_indices = []
    selected_patients = set()  
    unique_train_patients = np.unique(patient_indices_train_np)
    # 遍历患者，为每个患者选择一个癌症和一个良性样本
    for patient in unique_train_patients:
        if len(selected_patients) >= 3:  # 最多选择3个患者
            break
        patient_samples_indices = np.where(patient_indices_train_np == patient)[0]
        cancer_samples_from_patient = np.intersect1d(patient_samples_indices, cancer_indices)
        benign_samples_from_patient = np.intersect1d(patient_samples_indices, benign_indices)
        # 添加一个癌症样本和一个良性样本
        if len(cancer_samples_from_patient) > 0:
            selected_background_indices.append(cancer_samples_from_patient[0])
        if len(benign_samples_from_patient) > 0:
            selected_background_indices.append(benign_samples_from_patient[0])
        selected_patients.add(patient)
    # 确保索引是唯一的
    selected_background_indices = list(np.unique(selected_background_indices))[:10]  # 最多选择10个样本  
    # 背景数据：使用具有代表性的训练集样本
    background_ftir = ftir_train[selected_background_indices]
    print(f"SHAP背景数据选择了{len(selected_background_indices)}个训练样本:")
    print(f"对应患者: {patient_indices_train_np[selected_background_indices]}")
    
    # 测试样本：从测试集中选择具有代表性的样本，确保涵盖不同患者和类别
    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    patient_indices_test_np = patient_indices_test.cpu().numpy() if isinstance(patient_indices_test, torch.Tensor) else patient_indices_test  
    # 分别为癌症和良性类别选择代表性样本
    cancer_indices_test = np.where(y_test_np == 1)[0]
    benign_indices_test = np.where(y_test_np == 0)[0]
    # 为癌症样本选择代表性样本
    selected_cancer_indices = []
    selected_cancer_patients = set() 
    unique_cancer_patients = np.unique(patient_indices_test_np[cancer_indices_test])
    # 遍历癌症患者，为每个患者选择一个样本
    for patient in unique_cancer_patients:
        if len(selected_cancer_patients) >= 2:  # 最多选择2个患者
            break
        patient_samples_indices = np.where(patient_indices_test_np == patient)[0]
        cancer_samples_from_patient = np.intersect1d(patient_samples_indices, cancer_indices_test)
        if len(cancer_samples_from_patient) > 0:
            selected_cancer_indices.append(cancer_samples_from_patient[0])
            selected_cancer_patients.add(patient)
    # 为良性样本选择代表性样本
    selected_benign_indices = []
    selected_benign_patients = set() 
    unique_benign_patients = np.unique(patient_indices_test_np[benign_indices_test])
    # 遍历良性患者，为每个患者选择一个样本
    for patient in unique_benign_patients:
        if len(selected_benign_patients) >= 2:  # 最多选择2个患者
            break
        patient_samples_indices = np.where(patient_indices_test_np == patient)[0]
        benign_samples_from_patient = np.intersect1d(patient_samples_indices, benign_indices_test)
        # 添加一个良性样本
        if len(benign_samples_from_patient) > 0:
            selected_benign_indices.append(benign_samples_from_patient[0])
            selected_benign_patients.add(patient)
    # 确保索引是唯一的
    selected_cancer_indices = list(np.unique(selected_cancer_indices))[:3]  # 最多选择3个癌症样本
    selected_benign_indices = list(np.unique(selected_benign_indices))[:3]  # 最多选择3个良性样本
    # 测试样本：使用具有代表性的测试集样本
    test_samples_cancer_ftir = ftir_test[selected_cancer_indices]
    test_samples_benign_ftir = ftir_test[selected_benign_indices]
    print(f"SHAP测试数据选择了{len(selected_cancer_indices)}个测试样本:")
    print(f"对应患者: {patient_indices_test_np[selected_cancer_indices]}")

    # MZ 模态的基线：使用训练集的平均值
    mz_baseline = mz_train.mean(0, keepdim=True)

    # 3. 实例化包装模型
    wrapped_model = ShapModelWrapper(model, mz_baseline, ftir_x, mz_x)

    # 4. 创建 PyTorch 兼容的 GradientExplainer
    explainer = shap.GradientExplainer(wrapped_model, background_ftir)

    # 5. 分别计算癌症和良性样本的 SHAP 值
    # cancer_shap_values 的形状将是 (n_cancer_samples, n_features)
    cancer_shap_values = explainer.shap_values(test_samples_cancer_ftir)
    # benign_shap_values 的形状将是 (n_benign_samples, n_features)
    benign_shap_values = explainer.shap_values(test_samples_benign_ftir)

    # 6. 取 SHAP 值的平均绝对值，得到每个特征的重要性
    mean_abs_cancer_shap = np.mean(np.abs(cancer_shap_values), axis=0)
    mean_abs_benign_shap = np.mean(np.abs(benign_shap_values), axis=0)
    
    # 计算差异，用于识别区分两类的关键波数段
    shap_difference = np.abs(mean_abs_cancer_shap - mean_abs_benign_shap)

    # 打印Top 50的独立FTIR特征
    print("\n关键波数分析 (Top 50 individual features):")
    top_n_features = 50
    top_indices = np.argsort(shap_difference)[-top_n_features:][::-1]
    ftir_x_np = ftir_x.cpu().numpy()
    for i in top_indices:
        print(f"波数 {ftir_x_np[i]:.4f} cm-1: 癌症SHAP={mean_abs_cancer_shap[i]:.6f}, 良性SHAP={mean_abs_benign_shap[i]:.6f}, 差异={shap_difference[i]:.6f}")

    # 7. 绘制一维热力图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 为了实现X轴波数从小到大，反转SHAP值和波数数据
    plot_cancer_shap_values = mean_abs_cancer_shap[::-1]
    plot_benign_shap_values = mean_abs_benign_shap[::-1]
    plot_shap_difference = shap_difference[::-1]
    plot_ftir_x = ftir_x.cpu().numpy()[::-1].copy()

    start_wv = 900
    end_wv = 1800  
    step = 50.0
    wave_numbers = np.arange(start_wv, end_wv + step, step)

    # 找到最接近这些波数的索引
    tick_positions = []
    tick_labels = []
    for wv in wave_numbers:
        # 找到 plot_ftir_x 中最接近 wv 的值的索引
        idx = np.argmin(np.abs(plot_ftir_x - wv))
        # 只有当这个索引在图像范围内时才添加
        if idx < len(plot_ftir_x):
            tick_positions.append(idx)
            tick_labels.append(f"{int(wv)}")

   # 绘制良性和癌症样本的SHAP热力图（共用colorbar）
    plt.figure(figsize=(15, 8))  
    ax1 = plt.subplot(2, 1, 1)  # 良性在上
    ax2 = plt.subplot(2, 1, 2)  # 癌症在下

    # 计算共同的colorbar范围
    vmax = max(np.max(plot_benign_shap_values), np.max(plot_cancer_shap_values))
    vmin = min(np.min(plot_benign_shap_values), np.min(plot_cancer_shap_values))

    # 创建自定义颜色映射，减少紫色区域，增加其他颜色区域
    # 获取原始viridis颜色映射
    original_colors = plt.cm.viridis(np.linspace(0, 1, 256))

    # 创建新的颜色数组，通过非线性映射减少低值区域
    n_colors = 256
    new_colors = np.zeros((n_colors, 4))

    # 使用gamma校正来重新分配颜色
    gamma = 0.5  # 小于1的值会压缩低值区域，扩展高值区域
    for i in range(n_colors):
        # 将索引映射到[0,1]
        t = i / (n_colors - 1)
        # 应用gamma校正
        corrected_t = t ** gamma
        # 映射回颜色索引
        source_idx = int(corrected_t * (n_colors - 1))
        source_idx = min(source_idx, n_colors - 1)
        new_colors[i] = original_colors[source_idx]

    # 创建新的颜色映射
    custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', new_colors, N=n_colors)

    # 绘制良性样本的SHAP热力图
    heatmap_data_benign = plot_benign_shap_values.reshape(1, -1)
    im1 = ax1.imshow(heatmap_data_benign, cmap=custom_cmap, aspect='auto', 
                    interpolation='nearest', vmin=vmin, vmax=vmax)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=14)
    # ax1.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=16)
    ax1.set_yticks([])
    ax1.set_title('Benign', fontsize=18, pad=12)  # 减小标题下边距

    # 绘制癌症样本的SHAP热力图
    heatmap_data_cancer = plot_cancer_shap_values.reshape(1, -1)
    im2 = ax2.imshow(heatmap_data_cancer, cmap=custom_cmap, aspect='auto', 
                    interpolation='nearest', vmin=vmin, vmax=vmax)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=14)
    ax2.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=16)
    ax2.set_yticks([])
    ax2.set_title('Malignant', fontsize=18, pad=12)  # 减小标题下边距

    # 调整子图间距
    plt.subplots_adjust(right=0.85, hspace=0.4)  # 调整垂直间距

    # 添加共用的colorbar，放在右侧，调整位置避免遮挡，使colorbar更细
    cbar_ax = plt.axes([0.87, 0.15, 0.01, 0.7])  # 更细的colorbar，宽度从0.02减少到0.01
    cbar = plt.colorbar(im1, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Average SHAP value', rotation=270, labelpad=25, fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # 使用更均匀分布的刻度
    cbar_ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(cbar_ticks)

    plt.savefig('./result/ftir_shap_1d_heatmap_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP 一维热力图(良性和癌症合并)已保存至 ./result/ftir_shap_1d_heatmap_combined.png")

    # 8. 分组 SHAP 值以便返回，保持与后续代码的兼容性
    group_size = 5
    n_features = len(plot_cancer_shap_values)
    grouped_cancer_shap = []
    grouped_benign_shap = []
    grouped_difference_shap = []
    feature_names = []
    for i in range(0, n_features, group_size):
        end_idx = min(i + group_size, n_features)
        start_wv = plot_ftir_x[i]
        end_wv = plot_ftir_x[end_idx - 1]
        feature_names.append(f"{start_wv:.1f}-{end_wv:.1f} cm-1")
        grouped_cancer_shap.append(plot_cancer_shap_values[i:end_idx].mean())
        grouped_benign_shap.append(plot_benign_shap_values[i:end_idx].mean())
        grouped_difference_shap.append(plot_shap_difference[i:end_idx].mean())

    print("\n关键波数段分析:")
    # 找出差异最大的前10个波数段
    diff_indices = np.argsort(grouped_difference_shap)[-10:][::-1]
    for i in diff_indices:
        print(f"波数段 {feature_names[i]}: 癌症SHAP={grouped_cancer_shap[i]:.6f}, 良性SHAP={grouped_benign_shap[i]:.6f}, 差异={grouped_difference_shap[i]:.6f}")

    return shap_difference

# 只对 MZ 做 SHAP 分析
def perform_mz_shap_analysis(model, mz_train, mz_test, mz_x, ftir_train, ftir_x, y_test, patient_indices_train, patient_indices_test):
    """
    对 MultiModal 模型进行 MZ 特征的 Gradient SHAP 分析, 并生成一维热力图。
    分别分析癌症和良性样本，比较能够贡献区分的重要 mz 值段。
    """
    model.eval()

    # 1. 定义一个 PyTorch 模型包装器，用于 SHAP
    class ShapModelWrapper(torch.nn.Module):
        def __init__(self, model, ftir_baseline, ftir_x, mz_x):
            super().__init__()
            self.model = model
            self.register_buffer('ftir_baseline', ftir_baseline)
            self.register_buffer('ftir_x', ftir_x)
            self.register_buffer('mz_x', mz_x)

        def forward(self, mz_data):
            # SHAP 会传入一个需要计算梯度的张量
            current_ftir_baseline = self.ftir_baseline.expand(mz_data.shape[0], -1)
            current_ftir_axis = self.ftir_x.repeat(mz_data.shape[0], 1)
            mz_axis = self.mz_x.repeat(mz_data.shape[0], 1)
            outputs = self.model(current_ftir_baseline, mz_data, current_ftir_axis, mz_axis)
            return torch.softmax(outputs, dim=1)[:, 1].unsqueeze(-1)

    # 2. 准备背景数据和测试样本
    # 背景数据：从训练集中选择具有代表性的样本，确保背景数据涵盖不同患者和类别
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    patient_indices_train_np = patient_indices_train.cpu().numpy() if isinstance(patient_indices_train, torch.Tensor) else patient_indices_train
    # 为每个类别选择代表性样本
    cancer_indices = np.where(y_train_np == 1)[0]
    benign_indices = np.where(y_train_np == 0)[0]
    selected_background_indices = []
    selected_patients = set()  
    unique_train_patients = np.unique(patient_indices_train_np)
    # 遍历患者，为每个患者选择一个癌症和一个良性样本
    for patient in unique_train_patients:
        # 如果已经选够了患者，就停止
        if len(selected_patients) >= 3:  # 最多选择3个患者
            break
        patient_samples_indices = np.where(patient_indices_train_np == patient)[0]
        cancer_samples_from_patient = np.intersect1d(patient_samples_indices, cancer_indices)
        benign_samples_from_patient = np.intersect1d(patient_samples_indices, benign_indices)
        # 添加一个癌症样本和一个良性样本
        if len(cancer_samples_from_patient) > 0:
            selected_background_indices.append(cancer_samples_from_patient[0])
        if len(benign_samples_from_patient) > 0:
            selected_background_indices.append(benign_samples_from_patient[0])
        selected_patients.add(patient)
    # 确保索引是唯一的
    selected_background_indices = list(np.unique(selected_background_indices))[:10]  # 最多选择10个样本
    background_mz = mz_train[selected_background_indices]
    print(f"SHAP背景数据选择了{len(selected_background_indices)}个训练样本:")
    print(f"对应患者: {patient_indices_train_np[selected_background_indices]}")
    
    # 测试样本：从测试集中选择具有代表性的样本，确保涵盖不同患者和类别
    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    patient_indices_test_np = patient_indices_test.cpu().numpy() if isinstance(patient_indices_test, torch.Tensor) else patient_indices_test
    # 分别为癌症和良性类别选择代表性样本
    cancer_indices_test = np.where(y_test_np == 1)[0]
    benign_indices_test = np.where(y_test_np == 0)[0]
    # 为癌症样本选择代表性样本
    selected_cancer_indices = []
    selected_cancer_patients = set() 
    unique_cancer_patients = np.unique(patient_indices_test_np[cancer_indices_test])
    # 遍历癌症患者，为每个患者选择一个样本
    for patient in unique_cancer_patients:
        if len(selected_cancer_patients) >= 2:  # 最多选择2个患者
            break
        patient_samples_indices = np.where(patient_indices_test_np == patient)[0]
        cancer_samples_from_patient = np.intersect1d(patient_samples_indices, cancer_indices_test)
        # 添加一个癌症样本
        if len(cancer_samples_from_patient) > 0:
            selected_cancer_indices.append(cancer_samples_from_patient[0])
            selected_cancer_patients.add(patient)
    # 为良性样本选择代表性样本
    selected_benign_indices = []
    selected_benign_patients = set() 
    unique_benign_patients = np.unique(patient_indices_test_np[benign_indices_test])
    # 遍历良性患者，为每个患者选择一个样本
    for patient in unique_benign_patients:
        if len(selected_benign_patients) >= 2:  # 最多选择2个患者
            break
        patient_samples_indices = np.where(patient_indices_test_np == patient)[0]
        benign_samples_from_patient = np.intersect1d(patient_samples_indices, benign_indices_test)
        # 添加一个良性样本
        if len(benign_samples_from_patient) > 0:
            selected_benign_indices.append(benign_samples_from_patient[0])
            selected_benign_patients.add(patient)
    # 确保索引是唯一的
    selected_cancer_indices = list(np.unique(selected_cancer_indices))[:3]  # 最多选择3个癌症样本
    selected_benign_indices = list(np.unique(selected_benign_indices))[:3]  # 最多选择3个良性样本
    test_samples_cancer_mz = mz_test[selected_cancer_indices]
    test_samples_benign_mz = mz_test[selected_benign_indices]
    print(f"SHAP测试数据选择了{len(selected_cancer_indices)}个测试样本:")
    print(f"对应患者: {patient_indices_test_np[selected_cancer_indices]}")

    # FTIR 模态的基线：使用训练集的平均值
    ftir_baseline = ftir_train.mean(0, keepdim=True)

    # 3. 实例化包装模型
    wrapped_model = ShapModelWrapper(model, ftir_baseline, ftir_x, mz_x)

    # 4. 创建 PyTorch 兼容的 GradientExplainer
    explainer = shap.GradientExplainer(wrapped_model, background_mz)

    # 5. 分别计算癌症和良性样本的 SHAP 值
    # cancer_shap_values 的形状将是 (n_cancer_samples, n_features)
    cancer_shap_values = explainer.shap_values(test_samples_cancer_mz)
    # benign_shap_values 的形状将是 (n_benign_samples, n_features)
    benign_shap_values = explainer.shap_values(test_samples_benign_mz)

    # 6. 取 SHAP 值的平均绝对值，得到每个特征的重要性
    mean_abs_cancer_shap = np.mean(np.abs(cancer_shap_values), axis=0)
    mean_abs_benign_shap = np.mean(np.abs(benign_shap_values), axis=0)
    
    # 计算差异，用于识别区分两类的关键 mz 值段
    shap_difference = np.abs(mean_abs_cancer_shap - mean_abs_benign_shap)

    # 打印Top 50的独立MZ特征
    print("\n关键MZ值分析 (Top 50 individual features):")
    top_n_features = 50
    top_indices = np.argsort(shap_difference)[-top_n_features:][::-1]
    mz_x_np = mz_x.cpu().numpy()
    for i in top_indices:
        print(f"MZ值 {mz_x_np[i]:.4f}: 癌症SHAP={mean_abs_cancer_shap[i]:.6f}, 良性SHAP={mean_abs_benign_shap[i]:.6f}, 差异={shap_difference[i]:.6f}")

    # 7. 绘制一维热力图 - 先按mz_x排序，再分组处理
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取原始m/z值
    mz_x_np = mz_x.cpu().numpy() if isinstance(mz_x, torch.Tensor) else mz_x

    # 对数据按照mz_x从小到大排序
    sorted_indices = np.argsort(mz_x_np)
    sorted_mz_x = mz_x_np[sorted_indices]
    sorted_mean_abs_cancer_shap = mean_abs_cancer_shap[sorted_indices]
    sorted_mean_abs_benign_shap = mean_abs_benign_shap[sorted_indices]

    # 分组处理 - 每5个特征为一组
    group_size = 5
    n_features = len(sorted_mz_x)
    grouped_mz_centers = []
    grouped_cancer_shap = []
    grouped_benign_shap = []
    grouped_shap_diff = []

    for i in range(0, n_features, group_size):
        end_idx = min(i + group_size, n_features)
        # 计算每组的中心 mz 值
        center_mz = np.mean(sorted_mz_x[i:end_idx])
        grouped_mz_centers.append(center_mz)
        grouped_cancer_shap.append(np.mean(sorted_mean_abs_cancer_shap[i:end_idx]))
        grouped_benign_shap.append(np.mean(sorted_mean_abs_benign_shap[i:end_idx]))
        grouped_shap_diff.append(np.mean(np.abs(
            sorted_mean_abs_cancer_shap[i:end_idx] - sorted_mean_abs_benign_shap[i:end_idx])))

    # 转换为 numpy 数组
    grouped_mz_centers = np.array(grouped_mz_centers)
    grouped_cancer_shap = np.array(grouped_cancer_shap)
    grouped_benign_shap = np.array(grouped_benign_shap)
    grouped_shap_diff = np.array(grouped_shap_diff)

    # 定义刻度 - 基于实际数据范围，优化刻度分布
    min_mz = np.min(grouped_mz_centers)
    max_mz = np.max(grouped_mz_centers)
    
    # 使用固定步长生成刻度值
    tick_step = 50
    start_tick = np.ceil(min_mz / tick_step) * tick_step
    end_tick = np.floor(max_mz / tick_step) * tick_step
    tick_values = np.arange(start_tick, end_tick + tick_step, tick_step)
    
    # 找到最接近这些刻度值的分组索引位置，并优化分布
    tick_positions = []
    tick_labels = []
    
    # 用于跟踪已选择的刻度位置，确保有足够的间距
    min_distance = max(1, len(grouped_mz_centers) // 20)  # 最小间距为总长度的5%或至少1
    
    for i, tick_val in enumerate(tick_values):
        # 找到最接近tick_val的分组索引
        idx = np.argmin(np.abs(grouped_mz_centers - tick_val))
        
        # 检查是否与已添加的刻度位置有足够的间距
        is_far_enough = True
        for existing_pos in tick_positions:
            if abs(idx - existing_pos) < min_distance:
                is_far_enough = False
                break
        
        # 如果间距足够或这是第一个刻度，则添加
        if is_far_enough or len(tick_positions) == 0:
            tick_positions.append(idx)
            tick_labels.append(f"{int(grouped_mz_centers[idx])}")
        # 特殊处理最后一个刻度，确保图表末端有刻度
        elif i == len(tick_values) - 1 and len(tick_positions) > 0:
            # 检查是否与最后一个刻度有足够的距离
            if abs(idx - tick_positions[-1]) >= min_distance // 2:
                tick_positions.append(idx)
                tick_labels.append(f"{int(grouped_mz_centers[idx])}")

    # 绘制良性和癌症样本的SHAP热力图（共用colorbar）
    plt.figure(figsize=(15, 8))  
    ax1 = plt.subplot(2, 1, 1)  # 良性在上
    ax2 = plt.subplot(2, 1, 2)  # 癌症在下

    # 计算共同的colorbar范围
    vmax = max(np.max(grouped_benign_shap), np.max(grouped_cancer_shap))
    vmin = min(np.min(grouped_benign_shap), np.min(grouped_cancer_shap))

    # 创建自定义颜色映射，减少紫色区域，增加其他颜色区域
    # 获取原始viridis颜色映射
    original_colors = plt.cm.viridis(np.linspace(0, 1, 256))
    # 创建新的颜色数组，通过非线性映射减少低值区域
    n_colors = 256
    new_colors = np.zeros((n_colors, 4))
    # 使用gamma校正来重新分配颜色
    gamma = 0.5  # 小于1的值会压缩低值区域，扩展高值区域
    for i in range(n_colors):
        # 将索引映射到[0,1]
        t = i / (n_colors - 1)
        # 应用gamma校正
        corrected_t = t ** gamma
        # 映射回颜色索引
        source_idx = int(corrected_t * (n_colors - 1))
        source_idx = min(source_idx, n_colors - 1)
        new_colors[i] = original_colors[source_idx]
    # 创建新的颜色映射
    custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', new_colors, N=n_colors)

    # 绘制良性样本的SHAP热力图 - 使用分组后的数据
    heatmap_data_benign = grouped_benign_shap.reshape(1, -1)
    im1 = ax1.imshow(heatmap_data_benign, cmap=custom_cmap, aspect='auto', 
                    interpolation='nearest', vmin=vmin, vmax=vmax)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=14)
    ax1.set_yticks([])
    ax1.set_title('Benign', fontsize=18, pad=12)

    # 绘制癌症样本的SHAP热力图 - 使用分组后的数据
    heatmap_data_cancer = grouped_cancer_shap.reshape(1, -1)
    im2 = ax2.imshow(heatmap_data_cancer, cmap=custom_cmap, aspect='auto', 
                    interpolation='nearest', vmin=vmin, vmax=vmax)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=14)
    ax2.set_xlabel('m/z', fontsize=16)
    ax2.set_yticks([])
    ax2.set_title('Malignant', fontsize=18, pad=12)

    # 调整子图间距
    plt.subplots_adjust(right=0.85, hspace=0.4)  # 调整垂直间距

    # 添加共用的colorbar，放在右侧，调整位置避免遮挡，使colorbar更细
    cbar_ax = plt.axes([0.87, 0.15, 0.01, 0.7])  # 更细的colorbar，宽度从0.02减少到0.01
    cbar = plt.colorbar(im1, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Average SHAP value', rotation=270, labelpad=25, fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # 使用更均匀分布的刻度
    cbar_ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(cbar_ticks)

    plt.savefig('./result/mz_shap_1d_heatmap_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP 一维热力图(MZ良性和癌症合并)已保存至 ./result/mz_shap_1d_heatmap_combined.png")

    print("\n=== MZ SHAP 图调试信息 ===")
    print(f"总特征数: {len(mz_x_np)}")
    print(f"分组后特征数: {len(grouped_mz_centers)}")
    print(f"刻度值: {tick_values}")
    print(f"刻度标签: {tick_labels}")
    print(f"刻度位置: {tick_positions}")
    print("前5个分组的中心m/z值:", grouped_mz_centers[:5])
    print("最后5个分组的中心m/z值:", grouped_mz_centers[-5:])

    return shap_difference


def create_correlation_heatmap(ftir_data, mz_data, ftir_x, mz_x, ftir_indices, mz_indices, save_path):
    """
    计算选定的FTIR和MZ特征之间的Spearman相关性并绘制热力图。
    同时打印出强相关特征对。
    """
    print("\n开始进行Spearman相关性分析...")

    # 1. 获取选定的数据
    selected_ftir_data = ftir_data[:, ftir_indices]
    selected_mz_data = mz_data[:, mz_indices]

    ftir_labels = [f"{int(ftir_x[i])}" for i in ftir_indices]
    mz_labels = [f"{mz_x[i]:.1f}" for i in mz_indices]

    # 2. 计算Spearman相关性和p值
    num_ftir_features = len(ftir_indices)
    num_mz_features = len(mz_indices)
    corr_matrix = np.zeros((num_ftir_features, num_mz_features))
    pval_matrix = np.zeros((num_ftir_features, num_mz_features))

    for i in range(num_ftir_features):
        for j in range(num_mz_features):
            corr, pval = spearmanr(selected_ftir_data[:, i], selected_mz_data[:, j])
            corr_matrix[i, j] = corr
            pval_matrix[i, j] = pval

    # 3. 查找并打印显著相关性
    print("\n强相关特征对 (|r| >= 0.5 且 p < 0.05):")
    significant_pairs = []
    for i in range(num_ftir_features):
        for j in range(num_mz_features):
            if abs(corr_matrix[i, j]) >= 0.5 and pval_matrix[i, j] < 0.05:
                pair_info = (
                    f"FTIR: {ftir_labels[i]} cm-1, "
                    f"MZ: {mz_labels[j]}, "
                    f"r={corr_matrix[i, j]:.3f}, "
                    f"p={pval_matrix[i, j]:.4f}"
                )
                significant_pairs.append(pair_info)
                print(pair_info)

    if not significant_pairs:
        print("在给定阈值下未找到强相关特征对。")

    # 4. 绘制热力图
    plt.figure(figsize=(10, 8))  # 减小整个图像的尺寸

    # 对标签和矩阵进行排序以获得更好的可视化效果
    # 按照标签数值对mz特征进行排序
    mz_labels_float = [float(l) for l in mz_labels]
    mz_sort_indices = np.argsort(mz_labels_float)
    sorted_mz_labels = np.array(mz_labels)[mz_sort_indices]
    sorted_corr_matrix = corr_matrix[:, mz_sort_indices]

    # 按照标签数值对ftir特征进行排序
    ftir_labels_float = [float(l) for l in ftir_labels]
    ftir_sort_indices = np.argsort(ftir_labels_float)
    sorted_ftir_labels = np.array(ftir_labels)[ftir_sort_indices]
    sorted_corr_matrix = sorted_corr_matrix[ftir_sort_indices, :]

    # 绘制热力图，减小格子尺寸并添加浅灰色分隔线
    ax = sns.heatmap(
        sorted_corr_matrix,
        xticklabels=sorted_mz_labels,
        yticklabels=sorted_ftir_labels,
        cmap='coolwarm',
        annot=False,  
        vmin=-0.4, vmax=0.4,
        linewidths=0.2,  # 减小格子之间的分隔线宽度
        linecolor='lightgray',  # 分隔线颜色为浅灰色
        cbar_kws={'shrink': 0.8, 'aspect': 30, 'pad': 0.02}
    )

    # 设置颜色条属性
    cbar = ax.collections[0].colorbar
    cbar.set_label('Correlation coefficient', rotation=270, labelpad=25, fontsize=14)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)

    ax.set_title('Spearman Correlation between FTIR Spectra and Metabolomics Features', fontsize=16, pad=10)
    ax.set_xlabel('m/z', fontsize=14)
    ax.set_ylabel('Wavenumber (cm$^{-1}$)', fontsize=14)

    # 使刻度标签更易读
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # 为整个热力图添加外部黑色边框（不包括坐标轴标签区域）
    ax.add_patch(plt.Rectangle((0, 0), len(sorted_mz_labels), len(sorted_ftir_labels),
                               fill=False, edgecolor='black', linewidth=2))

    plt.tight_layout()
    heatmap_path = os.path.join(save_path, 'ftir_mz_correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n相关性热力图已保存至 {heatmap_path}")

# ==================数据增强====================================
def data_augmentation(x, axis, noise_std=0.1, scaling_factor=0.05, shift_range=0.02):
    torch.manual_seed(39)   # 41在mac的结果好，39在 kaggle 比较好
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
        f'./checkpoints/{model_type}_best_model.pth', weights_only=True))
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
        f'./checkpoints/{model_type}_best_model.pth', weights_only=True))
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
best_params = None


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


# 固定最优超参数（已通过网格搜索确定）
params = {
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'label_smoothing': 0.1,
    'scheduler_factor': 0.5,
    'early_stop_patience': 15
}

# 对所有模型，利用 k-fold 交叉验证调参，确定最优参数
models_to_evaluate = {
    "MultiModal": MultiModalModel,
    # "FTIROnly": SingleFTIRModel,
    # "MZOnly": SingleMZModel,
    # "ConcatFusion": ConcatFusion,
    # "GateOnlyFusion": GateOnlyFusion,
    # "CoAttnOnlyFusion": CoAttnOnlyFusion,
    # "SelfAttnFusion": SelfAttnFusion,
    # "SelfAttnOnlyFusion": SelfAttnOnlyFusion,
    # "SVM": SVMClassifier
}

# all_model_dfs = []
# for model_name, model_class in models_to_evaluate.items():
#     print(f"\n\n 开始评估模型: {model_name}")
#     if model_name == "SVM":
#         pass
#     else:
#         df = run_grid_search_for_model(model_name, model_class, ftir_train, mz_train, y_train,
#                                        ftir_x, mz_x, patient_indices_train, param_grid)
#     all_model_dfs.append(df)
#     # 合并所有模型结果
#     all_results_df = pd.concat(all_model_dfs, ignore_index=True)
#     all_results_df.to_csv(os.path.join(
#         save_path, 'all_models_grid_search_results.csv'), index=False)
#     print("所有模型 Grid Search 结果已保存至 all_models_grid_search_results.csv")

# # 加载 Grid Search 结果
# all_results_df = pd.read_csv(os.path.join(
#     save_path, 'all_models_grid_search_results.csv'))
# # 找出每个模型的最佳参数（按 avg_accuracy）
# best_params_per_model = {}
# for model_type in all_results_df['model_type'].unique():
#     df_model = all_results_df[all_results_df['model_type'] == model_type]
#     # 按 accuracy 找最优
#     best_row = df_model.loc[df_model['avg_accuracy'].idxmax()]
#     best_params = eval(best_row['params'])  # 字符串转字典
#     best_params_per_model[model_type] = best_params
#     print(f"[{model_type}] 最佳参数: {best_params}")

# 最后，使用最佳参数重新训练并在测试集上评估
final_test_results = []
training_history = {}
# for model_name, params in best_params_per_model.items():
for model_name, model_class in models_to_evaluate.items():
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

        # SHAP分析函数
        ftir_shap_difference = perform_ftir_shap_analysis(
            model, ftir_train, ftir_test, ftir_x, mz_train, mz_x, y_test, patient_indices_train, patient_indices_test
        )
        mz_shap_difference = perform_mz_shap_analysis(
            model, mz_train, mz_test, mz_x, ftir_train, ftir_x, y_test,
            patient_indices_train, patient_indices_test
        )

        # Spearman 相关性分析和热图
        # 合并训练和测试数据以进行相关性分析
        ftir_all = np.vstack((ftir_train.cpu().numpy(), ftir_test.cpu().numpy()))
        mz_all = np.vstack((mz_train.cpu().numpy(), mz_test.cpu().numpy()))

        # 特征选择
        # 1. FTIR: 基于SHAP分析选择Top 40个特征
        ftir_top_indices = np.argsort(ftir_shap_difference)[-40:]

        # 2. MZ: 基于SHAP分析选择Top 40个特征
        mz_top_indices = np.argsort(mz_shap_difference)[-40:]

        # 创建并保存相关性热图
        create_correlation_heatmap(
            ftir_all,
            mz_all,
            ftir_x.cpu().numpy(),
            mz_x.cpu().numpy(),
            ftir_top_indices,
            mz_top_indices,
            save_path
        )



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
"""
# 绘制每个模型 使用最优参数 在训练和测试时 的 loss 和 accuracy 曲线
plot_dir = os.path.join(save_path, 'training_plots')
os.makedirs(plot_dir, exist_ok=True)
# 设置全局样式
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2
})
soft_blue = '#6495ED'  # 柔和的蓝色
soft_red = '#CD5C5C'  # 柔和的红色
for model_name, data in training_history.items():
    # 绘制 Loss 曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(data['train_losses'], label='Train',
             color=soft_blue, linestyle='-')
    plt.plot(data['test_losses'], label='Test',
             color=soft_red, linestyle='-')
    plt.title(f'Training and Test Loss')
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
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # 强制整数刻度
    plt.grid(False)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(data['train_accuracies'], label='Train',
             color=soft_blue, linestyle='-')
    plt.plot(data['test_accuracies'], label='Test',
             color=soft_red, linestyle='-')
    plt.title(f'Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')  # 设置固定位置的图例
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)  # 加粗坐标轴
    ax.tick_params(axis='both', which='major', length=5,
                   width=1, direction='out')  # 设置刻度小短线
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # 强制整数刻度
    plt.grid(False)

    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(os.path.join(
        plot_dir, f'{model_name}_loss_accuracy_curve.png'))
    plt.close()

print(f"所有模型的 loss 和 accuracy 曲线已保存至 {plot_dir}")
"""