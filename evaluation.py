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
    # cm = confusion_matrix(y_true, preds)
    # save_confusion_matrix_heatmap(cm, save_path=save_path, method_name=name, show_plot=False)
    # # 绘制并保存 ROC 曲线
    # save_roc_curve(y_true, probs, auc, name, save_path)

    plot_cm_roc(y_true, preds, probs, auc, save_path=save_path, method_name=name)

    # t-SNE 可视化
    if name == "MultiModal":
        with torch.no_grad():
            ftir_feat = model.ftir_extractor(ftir_test, ftir_axis) if hasattr(model, 'ftir_extractor') else None
            mz_feat = model.mz_extractor(mz_test, mz_axis) if hasattr(model, 'mz_extractor') else None
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
    # 保存输入数据
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tsne_input_data = {
        'y_true': y_true
    }
    if ftir_feat is not None:
        tsne_input_data['ftir_features'] = ftir_feat
    if mz_feat is not None:
        tsne_input_data['mz_features'] = mz_feat
    if fused_feat is not None:
        tsne_input_data['fused_features'] = fused_feat
    np.savez(os.path.join(save_path, f'{model_name}_tsne_input_data.npz'), **tsne_input_data)
    
    
    plt.figure(figsize=(15, 5))
    feature_types = [
        ("FTIR Spectra Extracted Feats", ftir_feat),
        ("Mass Spectra Extracted Feats", mz_feat),
        ("Hybrid Fused Feats", fused_feat)
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
            palette={0: soft_blue, 1: soft_red},  
            style=y_true,  
            markers={0: "o", 1: "s"},  # 圆形和方形
            alpha=0.8,
            s=60,
            edgecolor='w',
            linewidth=0.5
        )
        plt.title(f"{title}", fontsize=TITLE_SIZE, pad=TITLE_PAD)
        plt.xlabel("t-SNE 1", fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
        plt.ylabel("t-SNE 2", fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
        handles, labels = scatter.get_legend_handles_labels()
        plt.legend(
            handles=handles,
            labels=['Benign', 'Malignant'],  # 明确标签
            # frameon=True,
            # edgecolor='black',
            # fancybox=False,  # 禁用圆角
            # shadow=False,     # 禁用阴影
            loc='upper right', fontsize=LEGEND_SIZE
        )
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', 
                   length=5, width=1, direction='out',
                   labelsize=XTICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{model_name}_tsne_comparison.png"), dpi=300)
    plt.close()


def save_confusion_matrix_heatmap(cm, save_path, method_name='Model', show_plot=True):
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
        linewidths=1.0,  # 单元格线宽
        linecolor='black',
        annot_kws={'size': XTICK_SIZE}, 
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant']
    )

    plt.xlabel('Predicted Label', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.ylabel('True Label', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.title(f'Confusion Matrix Heatmap(%)', fontsize=TITLE_SIZE, pad=TITLE_PAD)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=XTICK_SIZE)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=XTICK_SIZE)
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

def save_roc_curve(y_true, probs, auc, name, save_path):
    fpr, tpr, _ = roc_curve(y_true, probs, drop_intermediate=False)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#6495ED')
    plt.plot([0, 1], [0, 1], color='#b1b1b1', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.ylabel('True Positive Rate', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve', fontsize=TITLE_SIZE, pad=TITLE_PAD)
    plt.grid(False)
    plt.legend(
        # frameon=True,
        # edgecolor='black',
        # fancybox=False,  
        # shadow=False,    
        loc='upper right', fontsize=LEGEND_SIZE
    )
    # 设置坐标轴样式
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', 
                   length=5, width=1, direction='out',
                   labelsize=XTICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{name}_roc_curve.png'), dpi=300)
    plt.close()


def plot_cm_roc(y_true, preds, probs, auc, save_path, method_name='Model'):
    # 保存输入数据
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cm_roc_data = pd.DataFrame({
        'y_true': y_true,
        'y_pred': preds,
        'y_prob': probs,
        'auc': auc
    })
    cm_roc_data.to_csv(os.path.join(save_path, f'{method_name}_cm_roc_input_data.csv'), index=False)
    
    plt.figure(figsize=(16, 7))
    
    # 混淆矩阵热力图
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    ax1 = sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap='Blues',
        vmin=0,
        vmax=100,
        linewidths=1.0,
        linecolor='black',
        annot_kws={'size': XTICK_SIZE}, 
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant'],
        cbar_kws={'aspect': 30, 'pad': 0.04}
    )
    plt.xlabel('Predicted Label', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.ylabel('True Label', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.title(f'Confusion Matrix Heatmap(%)', fontsize=TITLE_SIZE, pad=TITLE_PAD)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=XTICK_SIZE)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=XTICK_SIZE)
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)  
    ax1.tick_params(axis='both', which='major', 
                   length=5, width=1, direction='out',
                   labelsize=XTICK_SIZE)
    cbar = ax1.collections[0].colorbar
    # cbar.set_label('Percentage (%)', rotation=270, labelpad=CBAR_LABELPAD, fontsize=CBAR_LABEL_SIZE)
    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(1.2)
    cbar.outline.set_edgecolor('black')
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # ROC曲线 
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, probs, drop_intermediate=False)
    plt.plot(fpr, tpr, color=soft_blue, linestyle='-', 
             linewidth=PLOT_LINE_WIDTH, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color=soft_gray, linestyle='--', 
             linewidth=PLOT_LINE_WIDTH, label='Random Classifier')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.ylabel('True Positive Rate', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve', fontsize=TITLE_SIZE, pad=TITLE_PAD)
    plt.grid(False)
    # plt.legend(
    #     frameon=True,
    #     edgecolor='black',
    #     fancybox=False,  
    #     shadow=False,    
    #     loc='lower right', fontsize=LEGEND_SIZE
    # )
    ax2 = plt.gca()
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax2.tick_params(axis='both', which='major', 
                   length=5, width=1, direction='out',
                   labelsize=XTICK_SIZE)
    
    plt.tight_layout()
    save_file = os.path.join(save_path, f'{method_name}_cm_roc.png')
    plt.savefig(save_file, dpi=300)
    plt.close()
    
    return save_file