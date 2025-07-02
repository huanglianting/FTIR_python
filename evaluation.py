import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def evaluate_model(
        model=None,
        ftir_test=None,
        mz_test=None,
        y_test=None,
        preds=None,
        probs=None,
        name="Model",
        model_type="MultiModal",
        fold=1,
        save_path='./result',
        is_svm=False
):

    y_true = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    # 如果没有提供 preds 和 probs
    if preds is None or probs is None:
        if is_svm:
            test_features = np.hstack([ftir_test.numpy(), mz_test.numpy()]) \
                if (isinstance(ftir_test, torch.Tensor) and isinstance(mz_test, torch.Tensor)) \
                else np.hstack([ftir_test, mz_test])
            preds = model.predict(test_features)
            probs = model.predict_proba(test_features)[:, 1]
        else:
            model.eval()
            with torch.no_grad():
                if isinstance(ftir_test, torch.Tensor) and isinstance(mz_test, torch.Tensor):
                    outputs = model(ftir_test, mz_test)
                elif isinstance(ftir_test, torch.Tensor):  # FTIR-only
                    outputs = model(ftir_test)
                elif isinstance(mz_test, torch.Tensor):  # mz-only
                    outputs = model(mz_test)
                else:
                    raise ValueError("Invalid input type for model prediction.")
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # 计算性能指标
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    spec = tn / (tn + fp)
    print(
        f"{name} - 准确率: {acc:.4f}, 精确率: {prec:.4f}, "
        f"召回率(Sensitivity): {rec:.4f}, 特异性: {spec:.4f}, "
        f"F1: {f1:.4f}, AUC: {auc:.4f}"
    )
    # 每个类别的准确率
    class_0_mask = (y_true == 0)
    class_1_mask = (y_true == 1)
    class_0_acc = (preds[class_0_mask] == y_true[class_0_mask]).mean()
    class_1_acc = (preds[class_1_mask] == y_true[class_1_mask]).mean()
    print(
        f"{name} - 类别0准确率: {class_0_acc:.4f}, 类别1准确率: {class_1_acc:.4f}"
    )

    result_dict = {
        'model_type': model_type,
        'fold': fold,
        'accuracy': acc,
        'precision': prec,
        'sensitivity': rec,
        'specificity': spec,
        'f1': f1,
        'auc': auc,
        'class_0_accuracy': class_0_acc,
        'class_1_accuracy': class_1_acc
    }

    # 绘制并保存混淆矩阵热力图
    cm = confusion_matrix(y_true, preds)
    save_confusion_matrix_heatmap(cm, save_path=save_path, method_name=name, show_plot=False)

    # 绘制并保存 ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{name} ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, f'{name}_roc_curve.png'))
    plt.close()

    return result_dict


def save_confusion_matrix_heatmap(cm, save_path, method_name='Model', show_plot=True):
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
        xticklabels=['Benign', 'Cancer'],
        yticklabels=['Benign', 'Cancer']
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
