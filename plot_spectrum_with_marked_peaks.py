import numpy as np
import matplotlib.pyplot as plt
import os


def plot_spectrum_with_marked_peaks(x, spectrum_1, spectrum_2, save_path, peak_wavenumbers):
    # peak_wavenumbers: 需要标注的波数点列表，例如：[1030, 1080, 1239, 1313, 1404, 1451, 1550, 1575]

    # 计算均值和标准差
    mean_cancer = np.mean(spectrum_1, axis=1)
    std_cancer = np.std(spectrum_1, axis=1)
    mean_benign = np.mean(spectrum_2, axis=1)
    std_benign = np.std(spectrum_2, axis=1)

    # 创建绘图
    plt.figure(figsize=(12, 6))
    plt.plot(x, mean_cancer, label='Cancer', color='red')
    plt.fill_between(x, mean_cancer - std_cancer, mean_cancer + std_cancer, color='red', alpha=0.2)
    plt.plot(x, mean_benign, label='Benign', color='green')
    plt.fill_between(x, mean_benign - std_benign, mean_benign + std_benign, color='green', alpha=0.2)

    # 在指定的波数点处标注短虚线
    for peak in peak_wavenumbers:
        # 获取该波数对应的y值
        idx = np.argmin(np.abs(x - peak))
        peak_height = (mean_cancer[idx] + mean_benign[idx]) / 2  # 使用两条曲线的平均值作为峰的高度
        # 绘制短虚线
        plt.plot([peak, peak], [peak_height - 0.02, peak_height + 0.02], color='black', linestyle='--', linewidth=1)
        # 在峰旁边标注波数值
        plt.text(peak, peak_height + 0.02, str(peak), fontsize=9, ha='center')

    # 设置图表标签和标题
    plt.xlabel('Wavenumber (cm^-1)')
    plt.ylabel('Absorbance')
    plt.title('Spectrum (Mean ± SD) with Marked Peaks')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # 反转x轴方向
    plt.gca().invert_xaxis()

    # 调整布局
    plt.tight_layout()

    # 保存并显示图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'Spectrum_with_Peaks.png'), dpi=300)
    plt.show()
