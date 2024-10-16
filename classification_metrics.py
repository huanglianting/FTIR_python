import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from calculate_per_class_accuracy import calculate_per_class_accuracy
from calculate_specificity import calculate_specificity


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
    columns = ['Benign', '441', '520', '1299', 'Overall']
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.columns = columns
    df_metrics = df_metrics.loc[['Accuracy', 'Precision', 'Sensitivity', 'Specificity']]

    # 导出到Excel
    excel_path = os.path.join(save_path, excel_filename)
    df_metrics.to_excel(excel_path, index=True)
    print(f"\nMetrics have been saved to {excel_path}")
