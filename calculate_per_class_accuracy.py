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
