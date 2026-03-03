# Multimodal Diagnostic Network
This repository implements a **multimodal diagnostic framework** for distinguishing between benign and malignant states by integrating **Fourier Transform Infrared (FTIR) spectroscopy** and **Mass Spectrometry (MS)** data. The project is designed for rigorous scientific evaluation, featuring a robust nested cross-validation pipeline, comprehensive statistical analysis, and advanced model interpretability.

## 📌 Key Features
- **Multimodal Fusion**: A novel hybrid fusion network combining gated fusion and cross-modal attention.
- **Rigorous Evaluation**: 5-times repeated 4-fold **Stratified Group Cross-Validation** to ensure patient-level independence and result stability.
- **Comprehensive Baselines**: Includes classical ML models (SVM, Random Forest) and deep learning baselines (MFCNN, CMSTF, CMACF).
- **Advanced Interpretability**: Uses **SHAP (SHapley Additive exPlanations)** to identify critical wavenumbers and m/z values that drive the model's decisions.
- **Full Reproducibility**: All intermediate results and plotting data are saved, enabling complete regeneration of all figures and tables without retraining.

## 📂 Project Structure
FTIR_python/
├── README.md                   # Project description document
├── main.py                     # Main training and evaluation pipeline (including nested cross-validation)
├── evaluation.py               # Model evaluation, metric calculation, statistical tests and visualization
├── Multi_Single_modal.py       # Multi-modal/single-modal model definitions (including ablation experiments and comparison models)
├── data_preprocessing.py       # Data loading, alignment, preprocessing and splitting
├── ftir_process.py             # FTIR spectrum-specific preprocessing (smoothing, derivatives, etc.)
├── plot_spectrum_with_marked_peaks.py  # Spectrum visualization tool with marked peaks
└── result/                     # Default result output directory (including model performance, SHAP, t-SNE, etc.)

## 🧪 Usage
### Environment Dependencies
Ensure the following Python libraries are installed:
```bash
numpy pandas scikit-learn torch matplotlib seaborn scipy shap
```
### Data Preparation
Organize FTIR `.mat` files and mass spectrometry (MS) `.xlsx` files according to the requirements in `data_preprocessing.py`.
### Run the Main Program
Execute `main.py` to start the complete pipeline of nested cross-validation, model training, evaluation, and statistical analysis:
```bash
python main.py
```
### Result Output
All results will be automatically saved to the `./result` directory, including:
- Model performance statistical report (`statistical_report.txt`)
- Visualization charts (ROC curves, confusion matrices, t-SNE plots, SHAP heatmaps, etc.)
- Intermediate data files (support result reproducibility)

## 📊 Output Examples
- **Performance Metrics**: Accuracy, AUC, Sensitivity, Specificity, F1-score, and their 95% confidence intervals
- **Statistical Tests**: Friedman test + Nemenyi post-hoc test / Wilcoxon signed-rank test
- **Interpretability Analysis**:
  - SHAP difference heatmaps
  - Spearman correlation heatmap
  - t-SNE feature space visualization