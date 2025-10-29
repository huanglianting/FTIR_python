
# data_preprocessing.py
- **Fig.1(a)**: `良恶性的 FTIR 原始光谱`
  - 在def preprocess_data中调用，该函数用于预处理ftir和mz
  - 具体函数定义在 plot_spectrum_with_marked_peaks.py
  - fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

- **Fig.1(b)**: `良恶性的 mz 强度百分比`
  - def plot_intensity_comparison
  - fig, (ax1, ax2) = plt.subplots(2, 1, figsize==(7, 7), sharex=True)


# evaluation.py
- **Fig.4**: `混淆矩阵和ROC曲线`
  - def plot_cm_roc
  - plt.figure(figsize=(16, 7))

- **Fig.5**: `t-SNE`
  - def plot_tsne_features
  - plt.figure(figsize=(15, 5))


# main.py
- **Fig.3**: `loss 和 accuracy 曲线`
  - 1262-1325行
  - plt.figure(figsize=(12, 5))

- **Fig.6**: `FTIR SHAP热力图`
  - def perform_ftir_shap_analysis
  - plt.figure(figsize=(15, 8))  
  
- **Fig.7**: `MZ SHAP热力图`
  - def perform_mz_shap_analysis
  - plt.figure(figsize=(15, 8))  

- **Fig.8**: `特征相关性热力图`
  - def create_correlation_heatmap
  - plt.figure(figsize=(12, 10)) 


# 其他
- **Fig.2**: `网络架构图`
  - 见PPT


# Multi_Single_modal.py
- 单模态、多模态、各种消融实验的模块定义

# ftir_process.py
- 用于读取数据、TR转AB、过滤到指纹区、求二阶导等FTIR光谱预处理

# plot_spectrum_with_marked_peaks.py
- 绘制良恶性的 FTIR 原始光谱

