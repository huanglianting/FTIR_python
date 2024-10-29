import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def save_confusion_matrix_heatmap(cm, save_path, method_name='method_name', show_plot=True):
    """
    保存混淆矩阵的热力图。
    参数:
    - cm_percent: 混淆矩阵的百分比形式 (numpy.ndarray)
    - save_path: 保存路径 (str)
    - method_name: 方法名称，用于生成标题和图像文件名 (str)
    """

    # 将混淆矩阵转换为百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 生成标题和图像文件名
    title = f'{method_name} Confusion Matrix Heatmap (%)'
    image_filename = f'{method_name}_confusion_matrix_heatmap.png'

    plt.figure(figsize=(8, 6))

    # 创建热力图
    ax = sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap='Blues',
        vmin=0,
        vmax=100,
        linewidths=0.5,  # 设置单元格之间的线宽
        linecolor='black',  # 设置线条颜色为黑色
        xticklabels=['Benign', '441', '520', '1299'],
        yticklabels=['Benign', '441', '520', '1299']
    )

    # 为热力图添加外部黑色边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')

    # 添加 colorbar 并设置边框
    colorbar = ax.collections[0].colorbar
    colorbar.outline.set_visible(True)
    colorbar.outline.set_linewidth(0.8)
    colorbar.outline.set_edgecolor('black')

    # 添加轴标签和标题
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)

    # 保存并显示热力图
    plt.tight_layout()
    image_path = os.path.join(save_path, image_filename)
    plt.savefig(image_path, dpi=300)

    if show_plot:
        plt.show()
    plt.close()

    print(f"Heatmap has been saved to {image_path}")
