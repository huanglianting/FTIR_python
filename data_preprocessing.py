import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
import seaborn as sns
from load_and_preprocess import load_and_preprocess
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from plot_spectrum_with_marked_peaks import plot_spectrum_with_marked_peaks

sns.set_style("whitegrid")


# 创建患者-代谢物矩阵
def prepare_heatmap_data(cancer_mz, normal_mz, common_mz):
    # 合并所有样本数据
    samples = []
    data = []
    groups = []
    # 添加癌症样本
    for col in cancer_mz:
        samples.append(col.replace('_', ' ').replace('[1]', '').strip())
        data.append(cancer_mz[col])
        groups.append(1)  # 癌症组标记为1
    # 添加正常样本
    for col in normal_mz:
        samples.append(col.replace('_', ' ').replace('[1]', '').strip())
        data.append(normal_mz[col])
        groups.append(0)  # 正常组标记为0
    # 创建数据框
    heatmap_df = pd.DataFrame(
        data,
        index=samples,
        columns=[f'm/z={mz:.2f}' for mz in common_mz]
    )
    return heatmap_df, np.array(groups)


# 筛选差异显著的代谢物（使用方差分析）
def select_significant_features(data, labels, n_features=50):
    from sklearn.feature_selection import f_classif
    f_values, p_values = f_classif(data, labels)
    sig_indices = np.argsort(p_values)[:n_features]
    return sig_indices


def plot_optimized_heatmap(cancer_mz, normal_mz, common_mz, save_path):
    heatmap_df, group_labels = prepare_heatmap_data(cancer_mz, normal_mz, common_mz)
    sig_indices = select_significant_features(heatmap_df.values, group_labels)
    filtered_df = heatmap_df.iloc[:, sig_indices]

    # 数据归一化到0-100% - 使用分位数缩放到增强对比度
    p5 = np.percentile(filtered_df.values, 5)  # 5%分位数
    p95 = np.percentile(filtered_df.values, 95)  # 95%分位数
    scaled_data = np.clip((filtered_df - p5) / (p95 - p5) * 100, 0, 100)

    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    # 优化热图参数：移除网格线，调整颜色中心点
    sns.heatmap(
        scaled_data.T,
        cmap='coolwarm',
        ax=ax,
        cbar=True,
        cbar_kws={'label': 'Intensity (%)', 'ticks': [0, 25, 50, 75, 100]},
        linewidths=0,  # 关键：移除网格线
        center=50,  # 增强颜色对比，使更多区域显示红色
        vmin=0,
        vmax=100
    )

    # 设置坐标轴标签
    ax.set_xlabel('Patients\n\nBenign            Malignant', fontsize=12, labelpad=10)
    ax.set_ylabel('Metabolomics Features', fontsize=12, labelpad=10)

    # 隐藏X轴具体患者编号
    ax.set_xticks([])
    ax.set_xticklabels([])

    # 设置标题
    plt.title('Clustering Heatmap of Significant m/z Features', fontsize=14)

    # 保存图像
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, 'metabolomics_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"热图已保存至: {output_path}")


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
    all_normal_spectra = np.hstack(list(normal_ftir.values()))  # (467, N_samples)
    all_cancer_spectra = np.hstack(list(cancer_ftir.values()))  # (467, N_samples)

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
    print("重采样df1...")
    df1_resampled = resample_data(df1, df1['m/z'].values, common_mz)
    print("重采样df2...")
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

    heatmap_df, group_labels = prepare_heatmap_data(
        cancer_mz, normal_mz, common_mz)
    sig_indices = select_significant_features(heatmap_df.values, group_labels)
    filtered_df = heatmap_df.iloc[:, sig_indices]
    plot_optimized_heatmap(cancer_mz, normal_mz, common_mz, save_path)

    """
    # 绘制热图
    plt.figure(figsize=(18, 12))
    sns.set(font_scale=0.8)
    # 创建颜色映射
    group_palette = sns.color_palette("Set1", 2)
    group_colors = [group_palette[x] for x in group_labels]
    # 绘制聚类热图
    g = sns.clustermap(
        filtered_df.T,  # 转置为 (代谢物 x 样本)
        method='average',
        metric='euclidean',
        z_score=0,  # 按行(代谢物)标准化
        cmap='coolwarm',  # 红蓝配色
        row_cluster=True,
        col_cluster=True,
        col_colors=group_colors,  # 样本分组颜色
        figsize=(18, 12),
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.85, 0.03, 0.1)
    )
    # 添加图例
    for i, label in enumerate(['Normal', 'Cancer']):
        g.ax_col_dendrogram.bar(
            0, 0, color=group_palette[i], label=label, linewidth=0)
    g.ax_col_dendrogram.legend(
        loc='upper left', ncol=2, bbox_to_anchor=(0.8, 0.9))
    # 优化布局
    plt.suptitle(
        'Clustering Heatmap of Significant m/z Features (Cancer vs Normal)', y=1.02, fontsize=14)
    g.ax_heatmap.set_xlabel('Patients')
    g.ax_heatmap.set_ylabel('m/z Features')
    # 保存图像
    output_path = os.path.join(save_path, 'metabolomics_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"热图已保存至: {output_path}")
    """

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

    # 遍历每个患者，处理并分配到对应集合
    for i in patients:  # 按打乱后的顺序处理患者
        # 处理癌症样本
        cancer_ftir_key = f'cancer{i}'
        cancer_mz_key = f'cancer_{i} [1]'
        # (48, 467)(N_samples, N_features)
        ftir_cancer = cancer_ftir[cancer_ftir_key].T
        mz_cancer = cancer_mz[cancer_mz_key].reshape(1, -1)  # (1, 3888/4780)
        print(f"ftir_cancer shape: {ftir_cancer.shape}")
        print(f"mz_cancer shape before repeat: {mz_cancer.shape}")
        # 复制代谢组学数据，使其样本数量和 FTIR 数据的样本数量相同，变成 (N_samples, N_features)
        mz_cancer_repeated = np.repeat(
            mz_cancer, ftir_cancer.shape[0], axis=0)  # shape: (48, 2838)
        print("mz_cancer_repeated shape:", mz_cancer_repeated.shape)
        labels_cancer = np.ones(
            ftir_cancer.shape[0], dtype=int)  # (48,), 癌症的标签标记为1
        print("labels_cancer shape:", labels_cancer.shape)

        # 处理正常样本
        normal_ftir_key = f'normal{i}'
        normal_mz_key = f'normal_{i} [1]'
        ftir_normal = normal_ftir[normal_ftir_key].T
        mz_normal = normal_mz[normal_mz_key].reshape(1, -1)
        # 复制代谢组学数据，使其样本数量和 FTIR 数据的样本数量相同
        mz_normal_repeated = np.repeat(mz_normal, ftir_normal.shape[0], axis=0)
        labels_normal = np.zeros(
            ftir_normal.shape[0], dtype=int)  # 对照组（正常）的标签标记为0

        # 合并患者i的所有样本
        ftir_all = np.vstack([ftir_cancer, ftir_normal])
        mz_all = np.vstack([mz_cancer_repeated, mz_normal_repeated])
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
