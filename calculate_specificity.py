import numpy as np

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