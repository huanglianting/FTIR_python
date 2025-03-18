import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


def calculate_per_class_accuracy(conf_matrix):
    """
    计算每个类别的准确率
    :param conf_matrix: 混淆矩阵
    :return: 每个类别的准确率列表
    """
    per_class_accuracy = []
    num_classes = conf_matrix.shape[0]
    total = conf_matrix.sum()
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = total - (TP + FP + FN)
        acc = (TP + TN) / total
        per_class_accuracy.append(acc)
    return per_class_accuracy


def calculate_specificity(conf_matrix):
    """
    计算每个类别的特异性
    :param conf_matrix: 混淆矩阵
    :return: 每个类别的特异性列表
    """
    specificity = []
    num_classes = conf_matrix.shape[0]
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = conf_matrix.sum() - (TP + FP + FN)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity.append(spec)
    return specificity


def classification_metrics(cm, y_test, y_pred, save_path, excel_filename='classification_metrics.xlsx'):
    # 计算每类Accuracy, Precision, Recall, Specificity
    per_class_acc = calculate_per_class_accuracy(cm)
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    specificity = calculate_specificity(cm)

    # 计算总体的 Accuracy, Precision, Recall, Specificity
    overall_accuracy = accuracy_score(y_test, y_pred)
    overall_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    overall_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    overall_specificity = np.mean(specificity)

    # 组织数据到DataFrameA
    metrics = {
        'Accuracy': per_class_acc + [overall_accuracy],
        'Precision': list(precision) + [overall_precision],
        'Sensitivity': list(recall) + [overall_recall],
        'Specificity': list(specificity) + [overall_specificity]
    }

    # 创建DataFrame
    columns = ['Benign', 'Cancer', 'Overall']
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.columns = columns
    df_metrics = df_metrics.loc[['Accuracy', 'Precision', 'Sensitivity', 'Specificity']]

    # 导出到Excel
    excel_path = os.path.join(save_path, excel_filename)
    df_metrics.to_excel(excel_path, index=True)
    print(f"\nMetrics have been saved to {excel_path}")


def classification_metrics_4(cm, y_test, y_pred, save_path, excel_filename='classification_metrics.xlsx'):
    # 计算每类Accuracy, Precision, Recall, Specificity
    per_class_acc = calculate_per_class_accuracy(cm)
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    specificity = calculate_specificity(cm)

    # 计算总体的 Accuracy, Precision, Recall, Specificity
    overall_accuracy = accuracy_score(y_test, y_pred)
    overall_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    overall_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    overall_specificity = np.mean(specificity)

    # 组织数据到DataFrameA
    metrics = {
        'Accuracy': per_class_acc + [overall_accuracy],
        'Precision': list(precision) + [overall_precision],
        'Sensitivity': list(recall) + [overall_recall],
        'Specificity': list(specificity) + [overall_specificity]
    }

    # 创建DataFrame
    columns = ['Benign', '441', '520', '1299', 'Overall']
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.columns = columns
    df_metrics = df_metrics.loc[['Accuracy', 'Precision', 'Sensitivity', 'Specificity']]

    # 导出到Excel
    excel_path = os.path.join(save_path, excel_filename)
    df_metrics.to_excel(excel_path, index=True)
    print(f"\nMetrics have been saved to {excel_path}")
