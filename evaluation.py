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


def evaluate_model(model, ftir_test, mz_test, y_test, ftir_axis, mz_axis,
                   preds=None, probs=None, name="Model", model_type="undefined",
                   fold=1, save_path='./result', is_svm=False):
    y_true = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    # 如果没有提供 preds 和 probs
    if preds is None or probs is None:
        if is_svm:
            ftir_test_np = ftir_test.numpy() if isinstance(ftir_test, torch.Tensor) else ftir_test
            mz_test_np = mz_test.numpy() if isinstance(mz_test, torch.Tensor) else mz_test
            ftir_axis_batch = np.tile(ftir_axis.numpy(), (ftir_test_np.shape[0], 1))  # [batch, 467]
            mz_axis_batch = np.tile(mz_axis.numpy(), (mz_test_np.shape[0], 1))  # [batch, 2838]
            test_features = np.hstack([ftir_test_np, mz_test_np, ftir_axis_batch, mz_axis_batch])
            preds = model.predict(test_features)
            probs = model.decision_function(test_features)  # 使用决策函数代替概率
            probs = (probs - probs.min()) / (probs.max() - probs.min())  # 可选归一化
        else:
            model.eval()
            with torch.no_grad():
                if isinstance(ftir_test, torch.Tensor) and isinstance(mz_test, torch.Tensor):
                    outputs = model(ftir_test, mz_test, ftir_axis, mz_axis)
                elif isinstance(ftir_test, torch.Tensor):  # FTIR-only
                    outputs = model(ftir_test, ftir_axis)
                elif isinstance(mz_test, torch.Tensor):  # mz-only
                    outputs = model(mz_test, mz_axis)
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

    save_roc_curve(y_true, probs, auc, name, save_path)

    # ========== t-SNE 可视化 ==========
    if name == "MultiModal":
        with torch.no_grad():
            # FTIR单模态特征
            ftir_feat = model.ftir_extractor(ftir_test, ftir_axis) if hasattr(model, 'ftir_extractor') else None
            # MZ单模态特征
            mz_feat = model.mz_extractor(mz_test, mz_axis) if hasattr(model, 'mz_extractor') else None
            # 融合特征
            fused_feat = model.fuser(ftir_feat, mz_feat) if hasattr(model, 'fuser') else None
        # 执行 t-SNE 降维
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        # 可视化各层次特征
        plot_tsne_features(
            tsne=tsne,
            ftir_feat=ftir_feat.cpu().numpy() if ftir_feat is not None else None,
            mz_feat=mz_feat.cpu().numpy() if mz_feat is not None else None,
            fused_feat=fused_feat.cpu().numpy() if fused_feat is not None else None,
            y_true=y_true,
            save_path=save_path,
            model_name=name
        )

    return result_dict


def plot_tsne_features(tsne, ftir_feat, mz_feat, fused_feat, y_true, save_path, model_name):
    plt.figure(figsize=(15, 5))
    feature_types = [
        ("FTIR Extractor Output", ftir_feat),
        ("Metabolomics Extractor Output", mz_feat),
        ("Fused Features", fused_feat)
    ]
    for idx, (title, feat) in enumerate(feature_types, start=1):
        if feat is None:
            continue
        # 降维并绘制
        reduced = tsne.fit_transform(feat)
        plt.subplot(1, 3, idx)
        scatter = sns.scatterplot(
            x=reduced[:, 0],
            y=reduced[:, 1],
            hue=y_true,
            palette={0: "#377EB8", 1: "#E41A1C"},  # 明确指定颜色
            style=y_true,  # 可选：用不同标记区分
            markers={0: "o", 1: "s"},  # 圆形和方形
            alpha=0.8,
            s=60,
            edgecolor='w',
            linewidth=0.5
        )
        plt.title(f"{title}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(labels=['Benign', 'Malignant'])
        # 优化图例
        handles, labels = scatter.get_legend_handles_labels()
        plt.legend(
            handles=handles,
            labels=['Benign', 'Malignant'],  # 明确标签
            frameon=True,
            edgecolor='black'
        )
    # 保存结果
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{model_name}_tsne_comparison.png"), dpi=300)
    plt.close()

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

def save_confusion_matrix_heatmap(cm, save_path, method_name='Model', show_plot=True):
    # 应用同样的统一样式
    plt.style.use('default')
    plt.rcParams.update(UNIFIED_STYLE)
    # 将混淆矩阵转换为百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap='Blues',
        vmin=0,
        vmax=100,
        linewidths=1.0,  # 单元格线宽加粗
        linecolor='black',
        annot_kws={'size': 13},  # 统一字体大小
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant']
    )

    # 设置坐标轴标签和标题
    plt.xlabel('Predicted Label', fontsize=13, labelpad=12)
    plt.ylabel('True Label', fontsize=13, labelpad=12)
    plt.title(f'Confusion Matrix Heatmap(%)', fontsize=14)
    # 设置刻度标签大小
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    # 加粗边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # 添加 colorbar 并设置边框
    colorbar = ax.collections[0].colorbar
    colorbar.outline.set_visible(True)
    colorbar.outline.set_linewidth(1.2)
    colorbar.outline.set_edgecolor('black')

    plt.tight_layout()
    save_path = os.path.join(save_path, f'{method_name}_confusion_matrix_heatmap.png')
    plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
    plt.close()
    return save_path


# 绘制并保存 ROC 曲线
def save_roc_curve(y_true, probs, auc, name, save_path):
    fpr, tpr, _ = roc_curve(y_true, probs, drop_intermediate=False)
    # 设置全局样式
    plt.style.use('default')
    plt.rcParams.update(UNIFIED_STYLE)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='#6495ED')
    plt.plot([0, 1], [0, 1], color='#b1b1b1', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13, labelpad=12)
    plt.ylabel('True Positive Rate', fontsize=13, labelpad=12)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.grid(False)

    # 设置坐标轴样式
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    # 统一设置刻度样式
    ax.tick_params(axis='both', which='major',
                   length=5, width=1, direction='out',
                   labelsize=12, pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{name}_roc_curve.png'), dpi=300)
    plt.close()
