import numpy as np
import matplotlib.pyplot as plt
import os


def plot_spectrum_with_mean_std(x_1, spectrum_1, spectrum_2, save_path):
    # 计算均值和标准差
    mean_cancer = np.mean(spectrum_1, axis=1)
    std_cancer = np.std(spectrum_1, axis=1)
    mean_benign = np.mean(spectrum_2, axis=1)
    std_benign = np.std(spectrum_2, axis=1)

    # 创建图形
    plt.figure()
    plt.plot(x_1, mean_cancer, label='Cancer', color='red')
    plt.fill_between(x_1, mean_cancer - std_cancer, mean_cancer + std_cancer, color='red', alpha=0.2)
    plt.plot(x_1, mean_benign, label='Benign', color='green')
    plt.fill_between(x_1, mean_benign - std_benign, mean_benign + std_benign, color='green', alpha=0.2)

    # 设置标签和标题
    plt.xlabel('Wavenumber (cm^-1)')
    plt.ylabel('Absorbance')
    plt.title('Spectrum (mean ± SD)')
    plt.legend()

    # 保存图像
    plt.savefig(os.path.join(save_path, 'Spectrum_result.png'))
    plt.show()
