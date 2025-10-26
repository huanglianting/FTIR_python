
# data_preprocessing.py
- **Fig.1(a)**: `良恶性的 FTIR 原始光谱`
  - 125-130行，在def preprocess_data中调用，该函数用于预处理ftir和mz
  - 函数定义在 plot_spectrum_with_marked_peaks.py中 21-78 行

- **Fig.1(b)**: `良恶性的 mz 强度百分比`
  - 37-73行
  - def plot_intensity_comparison


# evaluation.py
- **Fig.4**: `混淆矩阵和ROC曲线`
  - 166-214行：def save_confusion_matrix_heatmap
  - 215-244行：def save_roc_curve

- **Fig.5**: `t-SNE`
  - 84-151行
  - def plot_tsne_features


# main.py
- **Fig.3**: `loss 和 accuracy 曲线`
  - 1262-1325行

- **Fig.6**: `FTIR SHAP热力图`
  - 150-360行
  - def perform_ftir_shap_analysis
  
- **Fig.7**: `MZ SHAP热力图`
  - 361-592行
  - def perform_mz_shap_analysis

- **Fig.8**: `特征相关性热力图`
  - 593-675行
  - def create_correlation_heatmap


# 其他
- **Fig.2**: `网络架构图`
  - 见PPT


# Multi_Single_modal.py
- 单模态、多模态、各种消融实验的模块定义

# ftir_process.py
- 用于读取数据、TR转AB、过滤到指纹区、求二阶导等FTIR光谱预处理

# plot_spectrum_with_marked_peaks.py
- 绘制良恶性的 FTIR 原始光谱

