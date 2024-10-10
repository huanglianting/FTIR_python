import numpy as np
import os
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from calculate_per_class_accuracy import calculate_per_class_accuracy
from calculate_specificity import calculate_specificity


def perform_pca_lda_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, x_1, save_path,
                             n_pca_components=20):
    # n_pca_components: PCA降维后的主成分数量
    # 合并所有光谱数据
    combined_spectrum = np.hstack((spectrum_benign, spectrum_441, spectrum_520, spectrum_1299))
    num_samples_benign = spectrum_benign.shape[1]
    num_samples_441 = spectrum_441.shape[1]
    num_samples_520 = spectrum_520.shape[1]
    num_samples_1299 = spectrum_1299.shape[1]

    # 创建标签
    y = np.hstack((
        np.zeros(num_samples_benign),  # 标签 0: Benign
        np.ones(num_samples_441),  # 标签 1: 441
        2 * np.ones(num_samples_520),  # 标签 2: 520
        3 * np.ones(num_samples_1299)  # 标签 3: 1299
    )).astype(int)

    # 划分训练集和测试集
    X_train_spectrum, X_test_spectrum, y_train, y_test = train_test_split(
        combined_spectrum.T, y, test_size=0.3, random_state=42, stratify=y
    )

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_spectrum)  # 仅在训练集上拟合
    X_test_scaled = scaler.transform(X_test_spectrum)  # 使用相同的标准化器转换测试集

    # PCA降维
    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)  # 在训练集上拟合PCA
    X_test_pca = pca.transform(X_test_scaled)  # 使用相同的PCA模型转换测试集

    # 训练LDA分类器
    lda = LDA(n_components=3)  # 最多 classes - 1 维
    lda.fit(X_train_pca, y_train)

    # 预测
    y_pred = lda.predict(X_test_pca)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵作图出来，更直观，在图片上标注出数值
    # 绘制混淆矩阵热力图
    plt.figure(figsize=(8, 6))
    # 将混淆矩阵转换为百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

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
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('PCA-LDA Confusion Matrix Heatmap (%)')
    # 保存并显示热力图
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix_heatmap.png'), dpi=300)
    plt.show()

    # 计算每类Accuracy, Precision, Recall, Specificity (Macro Averages)
    per_class_acc = calculate_per_class_accuracy(cm)
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    specificity = calculate_specificity(cm)
    # 计算Overall Accuracy, Precision, Recall, Specificity (Macro Averages)
    overall_accuracy = accuracy_score(y_test, y_pred)
    overall_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    overall_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    overall_specificity = np.mean(specificity)

    # 组织数据到DataFrame
    metrics = {
        'Accuracy': per_class_acc + [overall_accuracy],
        'Precision': list(precision) + [overall_precision],
        'Sensitivity': list(recall) + [overall_recall],
        'Specificity': list(specificity) + [overall_specificity]
    }
    columns = ['Benign', '441', '520', '1299', 'Overall']
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.columns = columns
    df_metrics = df_metrics.loc[['Accuracy', 'Precision', 'Sensitivity', 'Specificity']]

    # 导出到Excel
    excel_path = os.path.join(save_path, 'classification_metrics.xlsx')
    df_metrics.to_excel(excel_path, index=True)
    print(f"\nMetrics have been saved to {excel_path}")


'''
    # 可视化：使用前两个LDA组件
    plt.figure(figsize=(10, 8))
    colors = ['green', 'orange', 'red', 'blue']
    labels = ['Benign', '441', '520', '1299']
    for class_idx, color, label in zip(range(4), colors, labels):
        plt.scatter(
            X_test_pca[y_test == class_idx, 0],     # 待确定
            X_test_pca[y_test == class_idx, 1],
            label=label,
            color=color,
            alpha=0.7,
            edgecolors='w',
            s=50
        )
    plt.xlabel(f'LD 1')
    plt.ylabel(f'LD 2')
    plt.title('PCA-LDA Result')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # 保存并显示图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'PCA_LDA_result.png'), dpi=300)
    plt.show()
'''
