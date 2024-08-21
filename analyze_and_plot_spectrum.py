import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def analyze_and_plot_spectrum(x_1, spectrum_1, spectrum_2, save_path, significance_level=0.05):
    # 计算均值和标准差
    mean_cancer = np.mean(spectrum_1, axis=1)
    std_cancer = np.std(spectrum_1, axis=1)
    mean_benign = np.mean(spectrum_2, axis=1)
    std_benign = np.std(spectrum_2, axis=1)

    # 进行正态性检验和适当的统计检验
    p_values = []
    for i in range(spectrum_1.shape[0]):
        _, p_norm_cancer = stats.shapiro(spectrum_1[i, :])
        _, p_norm_benign = stats.shapiro(spectrum_2[i, :])
        if p_norm_cancer > 0.05 and p_norm_benign > 0.05:
            # 如果两组数据都服从正态分布，使用t检验
            _, p_val = stats.ttest_ind(spectrum_1[i, :], spectrum_2[i, :], equal_var=False)
        else:
            # 如果至少一组数据不服从正态分布，使用Mann-Whitney U检验
            _, p_val = stats.mannwhitneyu(spectrum_1[i, :], spectrum_2[i, :], alternative='two-sided')
        p_values.append(p_val)
    p_values = np.array(p_values)

    # 标记显著差异的特征峰所在波数段（p < significance_level）
    significant_peaks = x_1[p_values < significance_level]

    # 绘制光谱并标记显著差异的波数段
    plt.figure(figsize=(12, 6))
    plt.plot(x_1, mean_cancer, label='Cancer', color='red')
    plt.fill_between(x_1, mean_cancer - std_cancer, mean_cancer + std_cancer, color='red', alpha=0.2)
    plt.plot(x_1, mean_benign, label='Benign', color='green')
    plt.fill_between(x_1, mean_benign - std_benign, mean_benign + std_benign, color='green', alpha=0.2)

    # 标记显著波数段并绘制虚线
    for peak in significant_peaks:
        plt.axvline(x=peak, color='black', linestyle='--', linewidth=1)

    plt.xlabel('Wavenumber (cm^-1)')
    plt.ylabel('Absorbance')
    plt.title('Spectrum (mean ± SD) with Significant Peaks')
    plt.legend()
    plt.gca().invert_xaxis()    # 反转x轴方向
    plt.savefig(os.path.join(save_path, 'Spectrum_with_Significant_Peaks.png'))
    plt.show()