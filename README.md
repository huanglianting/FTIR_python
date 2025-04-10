# cgan_utils.py
CGAN网络相关的函数，用于生成增强数据
# load_and_preprocess.py
对FTIR数据的一个读取和预处理，ftir_process.py和多模态中的data_preprocessing.py都会用到
# ftir_process.py
单独处理分析FTIR，包含数据提取、预处理，吸收度关于FTIR光谱波长可视化作图，只对FTIR进行训练集测试集的划分，PCA-LDA、RF、SVM、CNN四种模型分析，计算评价指标，绘制混淆矩阵、作图
# mz_process.py
单独处理分析mz，包含数据提取、预处理，相对丰度关于MZ的可视化作图。
