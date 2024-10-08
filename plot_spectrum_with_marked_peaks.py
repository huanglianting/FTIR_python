import numpy as np
import matplotlib.pyplot as plt
import os


def plot_spectrum_with_marked_peaks(x, spectrum_1, spectrum_2, spectrum_3, spectrum_4, save_path, peak_wavenumbers):
    # peak_wavenumbers: 需要标注的波数点列表，例如：[1030, 1080, 1239, 1313, 1404, 1451, 1550, 1575]

    # 计算均值和标准差
    mean_1 = np.mean(spectrum_1, axis=1)
    std_1 = np.std(spectrum_1, axis=1)
    mean_2 = np.mean(spectrum_2, axis=1)
    std_2 = np.std(spectrum_2, axis=1)
    mean_3 = np.mean(spectrum_3, axis=1)
    std_3 = np.std(spectrum_3, axis=1)
    mean_4 = np.mean(spectrum_4, axis=1)
    std_4 = np.std(spectrum_4, axis=1)

    # 创建包含四个子图的图形
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()  # 将2x2的子图数组展平，方便迭代

    # 定义每个子图的数据和标签
    spectra = [
        (mean_1, std_1, 'Benign', 'green'),
        (mean_2, std_2, '441', 'orange'),
        (mean_3, std_3, '520', 'red'),
        (mean_4, std_4, '1299', 'blue')
    ]

    for ax, (mean, std, label, color) in zip(axs, spectra):
        # 绘制均值曲线
        ax.plot(x, mean, label=label, color=color)
        # 填充均值±标准差区域
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

        # 在指定的波数点处标注短虚线
        for peak in peak_wavenumbers:
            # 获取该波数对应的y值
            idx = np.argmin(np.abs(x - peak))
            peak_height = mean[idx]
            # 绘制短虚线
            ax.plot([peak, peak], [peak_height - 0.02, peak_height + 0.02], color='black', linestyle='--', linewidth=1)
            # 在峰旁边标注波数值
            ax.text(peak, peak_height + 0.02, str(peak), fontsize=9, ha='center')

        # 设置子图标签和标题
        ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax.set_ylabel('Absorbance')
        ax.set_title(f'Spectrum: {label}')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.invert_xaxis()  # 反转x轴方向

    # 调整整体布局
    plt.tight_layout()

    # 保存并显示图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'Spectrum_with_Peaks_Subplots.png'), dpi=300)
    plt.show()

