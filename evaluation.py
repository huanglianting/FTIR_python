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
            ftir_test_np = ftir_test.numpy() if isinstance(
                ftir_test, torch.Tensor) else ftir_test
            mz_test_np = mz_test.numpy() if isinstance(mz_test, torch.Tensor) else mz_test
            ftir_axis_batch = np.tile(
                ftir_axis.numpy(), (ftir_test_np.shape[0], 1))  # [batch, 467]
            mz_axis_batch = np.tile(
                mz_axis.numpy(), (mz_test_np.shape[0], 1))  # [batch, 2838]
            test_features = np.hstack(
                [ftir_test_np, mz_test_np, ftir_axis_batch, mz_axis_batch])
            preds = model.predict(test_features)
            # 使用 predict_proba 获取稳定概率，避免决策函数归一化造成 NaN
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(test_features)[:, 1]
            else:
                scores = model.decision_function(test_features)
                denom = (scores.max() - scores.min())
                probs = (scores - scores.min()) / denom if denom != 0 else np.zeros_like(scores)
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
                    raise ValueError(
                        "Invalid input type for model prediction.")
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # 计算性能指标
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
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

    plot_cm_roc(y_true, preds, probs, auc,
                save_path=save_path, method_name=name)

    # t-SNE 可视化
    if name == "MultiModal":
        with torch.no_grad():
            ftir_feat = model.ftir_extractor(ftir_test, ftir_axis) if hasattr(
                model, 'ftir_extractor') else None
            mz_feat = model.mz_extractor(mz_test, mz_axis) if hasattr(
                model, 'mz_extractor') else None
            fused_feat = model.fuser(ftir_feat, mz_feat) if hasattr(
                model, 'fuser') else None
        # 执行 t-SNE 降维
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(
            10, len(y_true)-1), random_state=42)
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


def calculate_fold_variability(all_fold_results):
    """
    计算四折交叉验证的折间变异指标
    Args:
        all_fold_results: 列表，每个元素是一个字典，包含每个折的测试结果
    Returns:
        dict: 包含均值、标准差、95%置信区间的统计结果
    """
    import numpy as np
    from scipy import stats
    import pandas as pd

    # 将结果转换为DataFrame以便处理
    df = pd.DataFrame(all_fold_results)

    # 需要统计的指标
    metrics = ['accuracy', 'precision',
               'sensitivity', 'specificity', 'f1', 'auc']

    stats_results = {}

    for metric in metrics:
        values = df[metric].values

        # 计算均值和标准差
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # 样本标准差

        # 使用Bootstrap方法计算95%置信区间
        n_bootstrap = 1000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            # 有放回抽样
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        # 计算百分位数置信区间
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        stats_results[metric] = {
            'mean': mean_val,
            'std': std_val,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'format_str': f"{mean_val*100:.2f}% ± {std_val*100:.2f}%",
            'ci_format_str': f"{ci_lower*100:.2f}% - {ci_upper*100:.2f}%"
        }

    return stats_results


def perform_nonparametric_tests(model_results_dict):
    """
    对多个模型的性能进行非参数检验
    Args:
        model_results_dict: 字典，键为模型名，值为该模型在四折上的测试结果列表
    Returns:
        dict: 包含Friedman检验和事后检验的结果
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scikit_posthocs import posthoc_nemenyi_friedman

    # 准备数据：每个模型在四折上的AUC值
    auc_data = []
    model_names = []

    for model_name, fold_results in model_results_dict.items():
        auc_values = [result['auc'] for result in fold_results]
        auc_data.append(auc_values)
        model_names.append(model_name)

    auc_data = np.array(auc_data).T  # 转置为 (n_folds, n_models)

    # Friedman检验（非参数版ANOVA）
    friedman_stat, friedman_p = stats.friedmanchisquare(*auc_data.T)

    results = {
        'friedman_test': {
            'statistic': friedman_stat,
            'p_value': friedman_p,
            'significant': friedman_p < 0.05
        }
    }

    # 如果Friedman检验显著，进行事后检验
    if friedman_p < 0.05:
        try:
            # Nemenyi事后检验
            posthoc_results = posthoc_nemenyi_friedman(auc_data)
            results['posthoc_nemenyi'] = posthoc_results

            # 也可以使用Wilcoxon符号秩检验进行两两比较
            pairwise_comparisons = {}
            n_models = len(model_names)

            for i in range(n_models):
                for j in range(i+1, n_models):
                    # Wilcoxon符号秩检验
                    stat, p = stats.wilcoxon(auc_data[:, i], auc_data[:, j])
                    pairwise_comparisons[f"{model_names[i]}_vs_{model_names[j]}"] = {
                        'statistic': stat,
                        'p_value': p,
                        'significant': p < 0.05
                    }

            results['pairwise_wilcoxon'] = pairwise_comparisons

        except ImportError:
            print("警告: scikit-posthocs 未安装，无法进行Nemenyi事后检验")
            print("请安装: pip install scikit-posthocs")

    return results


def generate_statistical_report(model_stats_dict, save_path='./result'):
    """
    生成统计报告
    Args:
        model_stats_dict: 字典，键为模型名，值为calculate_fold_variability返回的统计结果
        save_path: 保存路径
    """
    import os
    import pandas as pd

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建汇总表格
    summary_data = []

    for model_name, stats in model_stats_dict.items():
        row = {'Model': model_name}

        # 为每个指标添加均值和标准差
        for metric in ['auc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1']:
            if metric in stats:
                row[f'{metric}_mean'] = stats[metric]['mean'] * 100
                row[f'{metric}_std'] = stats[metric]['std'] * 100
                row[f'{metric}_format'] = stats[metric]['format_str']

        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)

    # 保存为CSV
    csv_path = os.path.join(save_path, 'model_performance_statistics.csv')
    df_summary.to_csv(csv_path, index=False, float_format='%.2f')

    # 生成文本报告
    report_path = os.path.join(save_path, 'statistical_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("模型性能统计报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("一、折间变异指标（均值 ± 标准差）\n")
        f.write("-" * 60 + "\n")

        for model_name, stats in model_stats_dict.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  AUC: {stats['auc']['format_str']}\n")
            f.write(f"  准确率: {stats['accuracy']['format_str']}\n")
            f.write(f"  灵敏度: {stats['sensitivity']['format_str']}\n")
            f.write(f"  特异性: {stats['specificity']['format_str']}\n")
            f.write(f"  精确率: {stats['precision']['format_str']}\n")
            f.write(f"  F1分数: {stats['f1']['format_str']}\n")

        f.write("\n\n二、95%置信区间（Bootstrap方法）\n")
        f.write("-" * 60 + "\n")

        for model_name, stats in model_stats_dict.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  AUC 95% CI: {stats['auc']['ci_format_str']}\n")
            f.write(f"  灵敏度 95% CI: {stats['sensitivity']['ci_format_str']}\n")

        f.write("\n\n三、统计说明\n")
        f.write("-" * 60 + "\n")
        f.write("1. 折间变异指标：标准差越小，说明模型在不同数据划分上的性能越稳定\n")
        f.write("2. 95%置信区间：使用Bootstrap自助法计算，重复抽样1000次\n")
        f.write("3. 置信区间表示：有95%的概率，模型的真实性能落在此范围内\n")
        f.write("4. 所有统计均为非参数方法，适用于小样本数据\n")

    print(f"统计报告已保存至: {report_path}")
    print(f"详细数据已保存至: {csv_path}")

    return df_summary


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
    np.savez(os.path.join(
        save_path, f'{model_name}_tsne_input_data.npz'), **tsne_input_data)

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
        # plt.legend(
        #     handles=handles,
        #     labels=['Benign', 'Malignant'],  # 明确标签
        #     # frameon=True,
        #     # edgecolor='black',
        #     # fancybox=False,  # 禁用圆角
        #     # shadow=False,     # 禁用阴影
        #     loc='best',
        #     fontsize=LEGEND_SIZE
        # )
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', which='major',
                       length=5, width=1, direction='out',
                       labelsize=XTICK_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(
        save_path, f"{model_name}_tsne_comparison.png"), dpi=300)
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
    plt.title(f'Confusion Matrix Heatmap(%)',
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
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
    save_path = os.path.join(
        save_path, f'{method_name}_confusion_matrix_heatmap.png')
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
    plt.xlabel('False Positive Rate',
               fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.ylabel('True Positive Rate',
               fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve',
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
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
    cm_roc_data.to_csv(os.path.join(
        save_path, f'{method_name}_cm_roc_input_data.csv'), index=False)

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
        annot_kws={'size': 20},
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant'],
        cbar_kws={'aspect': 30, 'pad': 0.04}
    )
    plt.xlabel('Predicted Label', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.ylabel('True Label', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.title(f'Confusion Matrix Heatmap(%)',
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=XTICK_SIZE+1)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=XTICK_SIZE+1)
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax1.tick_params(axis='both', which='major',
                    length=5, width=1, direction='out',
                    labelsize=XTICK_SIZE+1)
    cbar = ax1.collections[0].colorbar
    # cbar.set_label('Percentage (%)', rotation=270, labelpad=CBAR_LABELPAD, fontsize=CBAR_LABEL_SIZE)
    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(1.2)
    cbar.outline.set_edgecolor('black')
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE+1)

    # ROC曲线
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, probs, drop_intermediate=False)
    plt.plot(fpr, tpr, color=soft_blue, linestyle='-',
             linewidth=PLOT_LINE_WIDTH, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color=soft_gray, linestyle='--',
             linewidth=PLOT_LINE_WIDTH, label='Random Classifier')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',
               fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.ylabel('True Positive Rate',
               fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve',
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
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
                    labelsize=XTICK_SIZE+1)

    plt.tight_layout()
    save_file = os.path.join(save_path, f'{method_name}_cm_roc.png')
    plt.savefig(save_file, dpi=300)
    plt.close()

    return save_file
