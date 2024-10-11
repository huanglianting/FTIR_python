import seaborn as sns
import matplotlib.pyplot as plt
import os


def save_confusion_matrix_heatmap(cm_percent, save_path, image_filename='confusion_matrix_heatmap.png'):
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

    # 为颜色图例添加外部黑色边框
    current_axes = plt.gca()
    for ax_obj in current_axes.figure.axes:
        if ax_obj != ax:
            # 这是颜色图例的轴对象
            for _, spine in ax_obj.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color('black')

    # 添加轴标签和标题
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('PCA-LDA Confusion Matrix Heatmap (%)')

    # 保存并显示热力图
    plt.tight_layout()
    image_path = os.path.join(save_path, image_filename)
    plt.savefig(image_path, dpi=300)
    plt.show()

    print(f"Heatmap has been saved to {image_path}")
