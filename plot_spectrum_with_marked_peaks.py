import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


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


def plot_spectrum_with_marked_peaks(x, spectrum_1, spectrum_2, save_path, peak_wavenumbers):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 保存输入数据到CSV文件
    spectrum_data = pd.DataFrame({
        'wavenumber': x,
        'benign_mean': np.mean(spectrum_1, axis=1),
        'benign_std': np.std(spectrum_1, axis=1),
        'malignant_mean': np.mean(spectrum_2, axis=1),
        'malignant_std': np.std(spectrum_2, axis=1)
    })
    spectrum_data.to_csv(os.path.join(save_path, 'spectrum_input_data.csv'), index=False)
    # 保存峰位数据
    peak_data = pd.DataFrame({'peak_wavenumbers': peak_wavenumbers})
    peak_data.to_csv(os.path.join(save_path, 'peak_wavenumbers.csv'), index=False)
    
    # peak_wavenumbers: 需要标注的波数点列表，例如：[1030, 1080, 1239, 1313, 1404, 1451, 1550, 1575]
    mean_1 = np.mean(spectrum_1, axis=1)
    std_1 = np.std(spectrum_1, axis=1)
    mean_2 = np.mean(spectrum_2, axis=1)
    std_2 = np.std(spectrum_2, axis=1)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    # # 绘制良性样本
    # ax1.plot(x, mean_1, color=soft_green, 
    #          linewidth=PLOT_LINE_WIDTH, label='Benign')
    # # ax1.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='green', alpha=0.2)
    # ax1.set_ylabel('Absorbance', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    # ax1.invert_xaxis()

    # # 绘制恶性样本
    # ax2.plot(x, mean_2, color=soft_red, 
    #          linewidth=PLOT_LINE_WIDTH, label='Malignant')
    # # ax2.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='red', alpha=0.2)
    # ax2.set_xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    # ax2.set_ylabel('Absorbance', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    # ax2.invert_xaxis()

    # 在同一张图中绘制良性样本和恶性样本
    ax.plot(x, mean_1, color=soft_green, 
             linewidth=PLOT_LINE_WIDTH, label='Benign')
    ax.plot(x, mean_2, color=soft_red, 
             linewidth=PLOT_LINE_WIDTH, label='Malignant')
    
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    ax.set_ylabel('Absorbance', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    # ax.invert_xaxis()

    # # 设置统一样式
    # for ax in [ax1, ax2]:
    #     ax.grid(False)
    #     ax.legend(loc='upper right', fontsize=LEGEND_SIZE) 
    #     for spine in ax.spines.values():
    #         spine.set_color('black')
    #         spine.set_linewidth(1.2)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)    
    #     ax.tick_params(axis='both', which='major', 
    #                length=5, width=1, direction='out', labelsize=XTICK_SIZE)

    ax.grid(False)
    ax.legend(loc='upper right', fontsize=LEGEND_SIZE) 
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    ax.tick_params(axis='both', which='major', 
               length=5, width=1, direction='out', labelsize=XTICK_SIZE)

    plt.tight_layout()
    
    # 在指定的波数点处标注短虚线
    # for peak in peak_wavenumbers:
    #     idx = np.argmin(np.abs(x - peak))
    #     peak_height = (mean_1[idx] + mean_2[idx]) / 2  # 使用四条曲线的平均值作为峰的高度
    #     plt.plot([peak, peak], [peak_height - 0.02, peak_height + 0.02], color='black', linestyle='--', linewidth=1)
    #     # 在峰旁边标注波数值
    #     plt.text(peak, peak_height + 0.02, str(peak), fontsize=9, ha='center')

    # plt.title('Spectrum with Marked Peaks', fontsize=14)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'Spectrum_with_Peaks.png'), dpi=300)
    print(f"图像已保存至: {os.path.join(save_path, f'Spectrum_with_Peaks.png')}")
    plt.show()
    plt.close()
