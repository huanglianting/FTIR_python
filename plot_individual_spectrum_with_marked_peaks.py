import numpy as np
import matplotlib.pyplot as plt
import os


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
        plt.plot([peak, peak], [peak_height - 0.02, peak_height + 0.02], color='black', linestyle='--', linewidth=1)
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
    plt.savefig(os.path.join(save_path, f'{label}_Spectrum_with_Peaks.png'), dpi=300)
    plt.show()