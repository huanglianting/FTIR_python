import numpy as np
import matplotlib.pyplot as plt
import os

# 设置统一风格参数
UNIFIED_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,  # 统一坐标轴线宽
    'font.size': 12,  # 统一字体大小
    'lines.linewidth': 2,  # 统一曲线线宽
    'xtick.major.width': 1.2,  # 统一x轴刻度线宽
    'ytick.major.width': 1.2,  # 统一y轴刻度线宽
    'axes.labelpad': 10  # 统一轴标签间距
}


def plot_individual_spectrum_with_marked_peaks(x, spectrum, label, color, save_path, peak_wavenumbers):
    # 计算均值和标准差
    mean = np.mean(spectrum, axis=1)
    std = np.std(spectrum, axis=1)

    # 创建绘图
    plt.figure(figsize=(12, 6))
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    # 在指定的波数点处标注短虚线
    for peak in peak_wavenumbers:
        # 获取该波数对应的y值
        idx = np.argmin(np.abs(x - peak))
        peak_height = mean[idx]  # 使用均值作为峰的高度
        # 绘制短虚线
        plt.plot([peak, peak], [peak_height - 0.02, peak_height +
                 0.02], color='black', linestyle='--', linewidth=1)
        # 在峰旁边标注波数值
        plt.text(peak, peak_height + 0.02, str(peak), fontsize=9, ha='center')

    # 设置图表标签和标题
    plt.xlabel('Wavenumber (cm^-1)')
    plt.ylabel('Absorbance')
    plt.title(f'{label} Spectrum (Mean ± SD) with Marked Peaks')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # 反转x轴方向
    plt.gca().invert_xaxis()

    # 调整布局
    plt.tight_layout()

    # 保存并显示图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(
        save_path, f'{label}_Spectrum_with_Peaks.png'), dpi=300)
    plt.show()


def plot_spectrum_with_marked_peaks(x, spectrum_1, spectrum_2, save_path, peak_wavenumbers):
    # peak_wavenumbers: 需要标注的波数点列表，例如：[1030, 1080, 1239, 1313, 1404, 1451, 1550, 1575]
    # 应用统一样式
    plt.style.use('default')
    plt.rcParams.update(UNIFIED_STYLE)
    # 计算均值和标准差
    mean_1 = np.mean(spectrum_1, axis=1)
    std_1 = np.std(spectrum_1, axis=1)
    mean_2 = np.mean(spectrum_2, axis=1)
    std_2 = np.std(spectrum_2, axis=1)

    # 设置子图（上下拼接）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # 绘制良性样本（Benign）
    ax1.plot(x, mean_1, label='Benign', color='green')
    # ax1.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='green', alpha=0.2)
    ax1.set_ylabel('Absorbance', fontsize=12)
    ax1.grid(False)
    ax1.invert_xaxis()

    # 绘制恶性样本（Malignant）
    ax2.plot(x, mean_2, label='Malignant', color='red')
    # ax2.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='red', alpha=0.2)
    ax2.set_xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=12)
    ax2.set_ylabel('Absorbance', fontsize=12)
    ax2.grid(False)
    ax2.invert_xaxis()

    # 设置统一样式
    for ax in [ax1, ax2]:
        ax.legend(loc='upper right', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)

    plt.tight_layout()

    """
    # 在指定的波数点处标注短虚线
    for peak in peak_wavenumbers:
        # 获取该波数对应的y值
        idx = np.argmin(np.abs(x - peak))
        peak_height = (mean_1[idx] + mean_2[idx]) / 2  # 使用四条曲线的平均值作为峰的高度
        # 绘制短虚线
        plt.plot([peak, peak], [peak_height - 0.02, peak_height + 0.02], color='black', linestyle='--', linewidth=1)
        # 在峰旁边标注波数值
        plt.text(peak, peak_height + 0.02, str(peak), fontsize=9, ha='center')
    """

    # plt.title('Spectrum with Marked Peaks', fontsize=14)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # 保存并显示图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'Spectrum_with_Peaks.png'), dpi=300)
    print(f"图像已保存至: {os.path.join(save_path, f'Spectrum_with_Peaks.png')}")
    plt.show()
    plt.close()
