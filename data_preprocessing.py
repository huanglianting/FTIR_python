import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from ftir_process import load_and_preprocess
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from plot_spectrum_with_marked_peaks import plot_spectrum_with_marked_peaks


# sns.set_style("whitegrid")

# 统一图表样式配置
UNIFIED_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'font.size': 20,  # 全局字体大小
    'legend.fontsize': 16,  # 图例字体大小
    'lines.linewidth': 2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'font.family': 'Arial',
    'axes.unicode_minus': False
}
soft_blue = '#377EB8'
soft_red = '#E41A1C'
soft_green = '#4DAF4A'
soft_gray = '#b1b1b1'
TITLE_SIZE = 22
TITLE_PAD = 12
AXIS_LABEL_SIZE = 20
LABEL_PAD = 12
XTICK_SIZE = 16
YTICK_SIZE = 16
LEGEND_SIZE = 14
PLOT_LINE_WIDTH = 2
CBAR_LABEL_SIZE = 20
CBAR_TICK_SIZE = 16
CBAR_LABELPAD = 25
SUBPLOT_RIGHT = 0.85
SUBPLOT_HSPACE = 0.6
plt.rcParams.update(UNIFIED_STYLE)


# 从给定的样本组中选择一个与其他样本相似度最高的样本作为“原型”
def select_most_similar_sample(group_data):
    sample_names = list(group_data.keys())
    sample_values = np.column_stack(
        [group_data[name] for name in sample_names])
    similarities = cosine_similarity(sample_values.T)
    # 计算每个样本与其他样本的平均相似度
    avg_similarity = np.mean(
        similarities - np.eye(similarities.shape[0]), axis=1)
    # 找到平均相似度最高的样本索引
    most_similar_idx = np.argmax(avg_similarity)
    return sample_names[most_similar_idx], sample_values[:, most_similar_idx]


# 将归一化丰度转换为强度百分比
def normalize_to_intensity_percentage(abundance_values):
    max_abundance = np.max(abundance_values)
    intensity_percentage = (abundance_values / max_abundance) * 100
    return intensity_percentage


# 绘制良恶性的 mz 强度百分比
def plot_intensity_comparison(common_mz, cancer_abundance, normal_abundance, save_path=".", title="Intensity Comparison"):
    # 保存输入数据
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    intensity_data = pd.DataFrame({
        'mz': common_mz,
        'cancer_abundance': cancer_abundance,
        'normal_abundance': normal_abundance,
        'cancer_intensity': normalize_to_intensity_percentage(cancer_abundance),
        'normal_intensity': normalize_to_intensity_percentage(normal_abundance)
    })
    intensity_data.to_csv(os.path.join(
        save_path, 'intensity_comparison_input_data.csv'), index=False)

    # 转换为强度百分比
    cancer_intensity = normalize_to_intensity_percentage(cancer_abundance)
    normal_intensity = normalize_to_intensity_percentage(normal_abundance)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_width = 2.2
    # 微调柱状图的位置，避免重叠
    mz_offset = bar_width / 2
    ax.bar(common_mz - mz_offset, normal_intensity, color=soft_green,
           alpha=0.7, label='Benign', width=bar_width)
    ax.bar(common_mz + mz_offset, cancer_intensity, color=soft_red,
           alpha=0.7, label='Malignant', width=bar_width)
    ax.set_xlabel('m/z', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    ax.set_ylabel('Intensity (%)', fontsize=AXIS_LABEL_SIZE,
                  labelpad=LABEL_PAD)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    # # 绘制良性样本
    # ax1.bar(common_mz, normal_intensity, color=soft_green,
    #         alpha=0.7, label='Benign', width=2.2)
    # ax1.set_ylabel('Intensity (%)', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    # # 绘制恶性样本
    # ax2.bar(common_mz, cancer_intensity, color=soft_red,
    #         alpha=0.7, label='Malignant', width=2.2)
    # ax2.set_xlabel('m/z', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    # ax2.set_ylabel('Intensity (%)', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)

    # 设置统一样式
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=LEGEND_SIZE)
    ax.set_xlim(min(common_mz), max(common_mz))
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major',
                   length=5, width=1, direction='out', labelsize=XTICK_SIZE)
    # # 设置统一样式
    # for ax in [ax1, ax2]:
    #     ax.grid(False)
    #     ax.legend(loc='upper right', fontsize=LEGEND_SIZE)
    #     ax.set_xlim(min(common_mz), max(common_mz))
    #     for spine in ax.spines.values():
    #         spine.set_color('black')
    #         spine.set_linewidth(1.2)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.tick_params(axis='both', which='major',
    #                length=5, width=1, direction='out', labelsize=XTICK_SIZE)

    plt.tight_layout()
    plt.savefig(os.path.join(
        save_path, f'{title.replace(" ", "_")}.png'), dpi=300)
    plt.show()
    plt.close()


def preprocess_data(ftir_file_path, mz_file_path1, mz_file_path2, train_folder, test_folder, save_path):
    # ===============================处理FTIR=================================================
    # 生成文件列表的通用函数
    def generate_file_lists(prefixes, num_files, ftir_file_path):
        all_file_lists = {}
        for prefix in prefixes:
            file_list = [
                f'{ftir_file_path}{prefix}_{i}.mat' for i in range(1, num_files + 1)]
            all_file_lists[prefix] = file_list
        return all_file_lists

    # 生成cancer和normal的文件列表
    # 一共有cancer1到cancer11，总计11个样品
    cancer_prefixes = [f'cancer{i}' for i in range(1, 12)]
    normal_prefixes = [f'normal{i}' for i in range(1, 12)]
    cancer_file_lists = generate_file_lists(
        cancer_prefixes, 3, ftir_file_path)  # 对于FTIR，每个样品重复三次滴加到基底
    normal_file_lists = generate_file_lists(normal_prefixes, 3, ftir_file_path)
    # 加载数据
    cancer_ftir_data = {k: [sio.loadmat(f) for f in v]
                        for k, v in cancer_file_lists.items()}
    normal_ftir_data = {k: [sio.loadmat(f) for f in v]
                        for k, v in normal_file_lists.items()}

    # FTIR光谱预处理相关参数
    threshold1 = 900  # 过滤掉小于threshold1的噪声
    threshold2 = 1800  # 过滤掉大于threshold2的噪声
    order = 2  # 多项式阶数
    frame_len = 13  # 窗口长度（帧长度）
    # 进行预处理
    normal_ftir, cancer_ftir = {}, {}
    x_ftir, normal_ftir['normal1'] = load_and_preprocess(normal_ftir_data['normal1'], threshold1, threshold2,
                                                         order, frame_len, save_path)
    for key in list(normal_ftir_data.keys())[1:]:
        _, normal_ftir[key] = load_and_preprocess(normal_ftir_data[key], threshold1, threshold2, order, frame_len,
                                                  save_path)
    for key in cancer_ftir_data.keys():
        _, cancer_ftir[key] = load_and_preprocess(cancer_ftir_data[key], threshold1, threshold2, order, frame_len,
                                                  save_path)

    # 打印每个样品的形状
    print("x_ftir shape:", x_ftir.shape)  # (467,)
    # print(f"spectrum_normal1 shape: {normal_ftir['normal1'].shape}")  # 形状均为(467, xxxx)

    # 提取 normal 和 cancer 的所有光谱
    all_normal_spectra = np.hstack(
        list(normal_ftir.values()))  # (467, N_samples)
    all_cancer_spectra = np.hstack(
        list(cancer_ftir.values()))  # (467, N_samples)

    # ===================================处理mz===========================================================
    df1 = pd.read_excel(mz_file_path1, header=1)  # 从第二行读取数据
    df2 = pd.read_excel(mz_file_path2, header=1)

    # 两个文件的m/z列不一致
    def align_mz_values(mz1, mz2, tolerance=0.01):
        aligned_mz2 = np.full_like(mz1, np.nan)  # 初始化对齐数组
        common_indices = []
        for i, mz_val in enumerate(mz2):
            # 在mz1中寻找最接近的值
            closest_idx = np.argmin(np.abs(mz1 - mz_val))
            if np.abs(mz1[closest_idx] - mz_val) <= tolerance:
                aligned_mz2[closest_idx] = mz_val
                common_indices.append((closest_idx, i))
        return aligned_mz2, common_indices

    # 执行对齐
    aligned_mz2, common_indices = align_mz_values(
        df1['m/z'].values, df2['m/z'].values)
    # 提取共同m/z值 (取两者平均值)
    common_mz = []
    for mz1_idx, mz2_idx in common_indices:
        avg_mz = (df1['m/z'].values[mz1_idx] + df2['m/z'].values[mz2_idx]) / 2
        common_mz.append(avg_mz)
    common_mz = np.array(common_mz)
    print(f"找到的共同m/z特征数: {len(common_mz)}")
    print(f"共同m/z范围: {common_mz.min():.2f} - {common_mz.max():.2f}")

    # 重采样函数
    def resample_data(df, orig_mz, common_mz):
        resampled_df = pd.DataFrame()
        resampled_df['m/z'] = common_mz  # 新的m/z列
        for col in df.columns:
            if col != 'm/z':  # 跳过m/z列本身
                try:
                    interp_func = interp1d(
                        orig_mz,
                        df[col].values,
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate'  # 允许外推
                    )
                    resampled_values = interp_func(common_mz)
                    # 外推可能导致极端值，需限制在合理范围（例如非负）
                    resampled_values = np.clip(
                        resampled_values, a_min=0, a_max=None)
                except Exception as e:
                    print(f"列 {col} 外推失败，改用填充0: {str(e)}")
                    resampled_values = np.zeros_like(common_mz)
                resampled_df[col] = resampled_values
        return resampled_df

    # 对两个数据文件进行重采样
    df1_resampled = resample_data(df1, df1['m/z'].values, common_mz)
    df2_resampled = resample_data(df2, df2['m/z'].values, common_mz)
    print("重采样后统计（前5个m/z点）:\n", df1_resampled.iloc[:5].describe())
    print("NaN比例:", df1_resampled.isna().mean().mean())

    cancer_mz = {}
    normal_mz = {}
    # 合并df1_resampled的样本 (患者1-7)
    for col in df1_resampled.columns:
        if 'cancer' in col.lower():
            cancer_mz[col] = df1_resampled[col].values
        elif 'normal' in col.lower():
            normal_mz[col] = df1_resampled[col].values
    # 合并df2_resampled的样本 (患者8-11)
    for col in df2_resampled.columns:
        if 'cancer' in col.lower():
            cancer_mz[col] = df2_resampled[col].values
        elif 'normal' in col.lower():
            normal_mz[col] = df2_resampled[col].values
    print("common_mz shape:", common_mz.shape)  # (2838,)

    # 选择最相似的恶性样本、良性样本
    most_similar_cancer_name, most_similar_cancer_spectrum = select_most_similar_sample(
        cancer_mz)
    most_similar_normal_name, most_similar_normal_spectrum = select_most_similar_sample(
        normal_mz)
    plot_intensity_comparison(
        common_mz=common_mz,
        cancer_abundance=most_similar_cancer_spectrum,
        normal_abundance=most_similar_normal_spectrum,
        save_path=save_path,
        title="Intensity Comparison"
    )

    # =============================按患者i处理FTIR和mz数据并划分set======================================
    # 按患者初始化（ 训练(+验证) / 测试 ）列表
    train_ftir = []
    train_mz = []
    train_labels = []
    train_patient_ids = []
    test_ftir = []
    test_mz = []
    test_labels = []
    test_patient_ids = []
    # 随机打乱患者顺序（1-11）38、29、28、21、59
    np.random.seed(59)
    patients = np.arange(1, 12)  # 患者i=1到11
    np.random.shuffle(patients)
    # 划分比例：8训练(+验证)，3测试
    train_patients_list = patients[:8]
    test_patients_list = patients[8:]

    # 提取11个患者的平均FTIR光谱用于绘图
    all_patients_cancer = []
    all_patients_normal = []
    # 遍历每个患者，处理并分配到对应集合
    for i in patients:  # 按打乱后的顺序处理患者
        # 处理癌症样本
        cancer_ftir_key = f'cancer{i}'
        cancer_mz_key = f'cancer_{i} [1]'
        # (48, 467)(N_samples, N_features)
        ftir_cancer = cancer_ftir[cancer_ftir_key].T
        # 对每个患者的48张FTIR做平均，得到1张代表性谱图
        ftir_cancer_avg = np.mean(
            ftir_cancer, axis=0, keepdims=True)  # shape: (1, 467)
        # 存储平均后的癌症光谱用于绘图
        all_patients_cancer.append(ftir_cancer_avg)
        mz_cancer = cancer_mz[cancer_mz_key].reshape(1, -1)  # (1, 2838)
        print(f"ftir_cancer shape: {ftir_cancer_avg.shape}")
        print(f"mz_cancer shape: {mz_cancer.shape}")
        labels_cancer = np.ones(
            ftir_cancer_avg.shape[0], dtype=int)  # (1,), 癌症的标签标记为1
        print("labels_cancer shape:", labels_cancer.shape)

        # 处理正常样本
        normal_ftir_key = f'normal{i}'
        normal_mz_key = f'normal_{i} [1]'
        ftir_normal = normal_ftir[normal_ftir_key].T
        ftir_normal_avg = np.mean(
            ftir_normal, axis=0, keepdims=True)  # shape: (1, 467)
        # 存储平均后的正常光谱用于绘图
        all_patients_normal.append(ftir_normal_avg)
        mz_normal = normal_mz[normal_mz_key].reshape(1, -1)
        labels_normal = np.zeros(
            ftir_normal_avg.shape[0], dtype=int)  # 对照组（正常）的标签标记为0

        # 合并患者i的所有样本
        ftir_all = np.vstack([ftir_cancer_avg, ftir_normal_avg])
        mz_all = np.vstack([mz_cancer, mz_normal])
        labels_all = np.hstack([labels_cancer, labels_normal])
        patient_ids = np.full_like(labels_all, i)  # 为每个样本添加患者ID
        print(f"[患者 {i}] ftir_all shape:", ftir_all.shape)
        print(f"[患者 {i}] mz_all shape:", mz_all.shape)
        print(f"[患者 {i}] labels_all shape:", labels_all.shape)

        # 打乱单个患者内的癌症/正常顺序，不然都是010101
        # 生成打乱索引
        np.random.seed(i)  # 用i作种子，固定患者内的随机种子，确保特征和标签同步打乱
        indices = np.arange(ftir_all.shape[0])
        np.random.shuffle(indices)
        # 按索引打乱数据
        ftir_shuffled = ftir_all[indices]
        mz_shuffled = mz_all[indices]
        labels_shuffled = labels_all[indices]
        patient_ids_shuffled = np.full_like(labels_shuffled, i)
        print(f"[患者 {i}] ftir_shuffled shape:", ftir_shuffled.shape)
        print(f"[患者 {i}] mz_shuffled shape:", mz_shuffled.shape)
        print(f"[患者 {i}] labels_shuffled shape:", labels_shuffled.shape)
        print(f"[患者 {i}] patient_ids_shuffled shape:",
              patient_ids_shuffled.shape)

        # 根据患者i所在训练/测试集分配打乱后的数据
        if i in train_patients_list:
            train_ftir.append(ftir_shuffled)
            train_mz.append(mz_shuffled)
            train_labels.append(labels_shuffled)
            train_patient_ids.append(patient_ids_shuffled)
        else:
            test_ftir.append(ftir_shuffled)
            test_mz.append(mz_shuffled)
            test_labels.append(labels_shuffled)
            test_patient_ids.append(patient_ids_shuffled)

    # 将所有患者的癌症和正常光谱堆叠起来
    all_patients_cancer = np.array(all_patients_cancer)  # (11, 467)
    all_patients_normal = np.array(all_patients_normal)  # (11, 467)
    # 确保数组是正确的形状并转换为浮点类型
    all_patients_cancer_array = np.asarray(all_patients_cancer, dtype=np.float32)  # (11, 467)
    all_patients_normal_array = np.asarray(all_patients_normal, dtype=np.float32)  # (11, 467)

    # 检查数组形状
    print(f"all_patients_cancer_array shape: {all_patients_cancer_array.shape}")
    print(f"all_patients_normal_array shape: {all_patients_normal_array.shape}")

    # 调用FTIR绘图函数 - 确保输入的是 (467, n) 形状的数组
    plot_spectrum_with_marked_peaks(
        x=x_ftir,
        spectrum_1=all_patients_normal_array.T,  # (467, 11) - 良性样本
        spectrum_2=all_patients_cancer_array.T,  # (467, 11) - 恶性样本
        save_path=save_path,
        peak_wavenumbers=[990, 1030, 1075, 1100, 1150,
                        1200, 1230, 1313, 1360, 1415, 1455, 1585, 1640]
    )
    # 堆叠所有患者的数据
    train_ftir = np.vstack(train_ftir)  # (8*96, 467) = (768, 467)
    train_mz = np.vstack(train_mz)  # (768, 2838)
    train_labels = np.hstack(train_labels)  # (768,)
    train_patient_ids = np.hstack(train_patient_ids)  # (768,)
    test_ftir = np.vstack(test_ftir)  # (3*96, 467) = (288, 467)
    test_mz = np.vstack(test_mz)  # (288, 2838)
    test_labels = np.hstack(test_labels)  # (288,)
    test_patient_ids = np.hstack(test_patient_ids)  # (288,)

    """
    # FTIR train raw 的 PCA 图
    pca = PCA(n_components=2)
    ftir_pca = pca.fit_transform(ftir_train_raw)
    colors = ['#0072B2', '#D55E00']  # 蓝色和橙色，类似常见论文配色
    # 绘制散点图并设置透明度
    plt.scatter(ftir_pca[y_train == 0, 0], ftir_pca[y_train == 0, 1],
                label='Normal', c=colors[0], alpha=0.6)
    plt.scatter(ftir_pca[y_train == 1, 0], ftir_pca[y_train == 1, 1],
                label='Cancer', c=colors[1], alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title("FTIR Data Distribution")
    plt.savefig('FTIR_Data_Distribution.png')
    plt.close()
    """
    return train_ftir, train_mz, train_labels, train_patient_ids, \
        test_ftir, test_mz, test_labels, test_patient_ids, \
        x_ftir, common_mz
