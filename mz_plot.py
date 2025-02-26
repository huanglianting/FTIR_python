import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perform_cnn_analysis import train_cnn_model, test_cnn_model
from perform_pca_lda_analysis import train_pca_lda_model, test_pca_lda_model
from perform_pca_rf_analysis import train_pca_rf_model, test_pca_rf_model
from perform_svm_analysis import train_svm_model, test_svm_model
from split_dataset import split_dataset


save_path = 'N:\\hlt\\FTIR\\result\\FNA_supernatant'  # 保存图片的路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 读取数据
file_path = r'N:\hlt\FTIR\result\FNA_supernatant\compound measurements.xlsx'
df = pd.read_excel(file_path, header=1)  # 从第二行读取数据

# 提取癌症样本和正常样本的数据
cancer_columns = [col for col in df.columns if 'cancer' in col.lower()]
normal_columns = [col for col in df.columns if 'normal' in col.lower()]
mz_cancer = df[cancer_columns].values  # 癌症样本数据
mz_benign = df[normal_columns].values  # 正常样本数据

# 计算每个 m/z 的癌症样本和正常样本的平均值
df['cancer_mean'] = df[cancer_columns].mean(axis=1)
df['normal_mean'] = df[normal_columns].mean(axis=1)
# 将 m/z 及其对应的 cancer_mean 和 normal_mean 列输出到 Excel 文件
output_df = df[['m/z', 'cancer_mean', 'normal_mean']]
output_file_path = os.path.join(save_path, 'mz_with_mean_values.xlsx')
output_df.to_excel(output_file_path, index=False)
print(f"m/z 及其对应的平均值已保存到: {output_file_path}")

# 计算差值绝对值并选取最大的几行（这里选取前 1200 行，可根据实际情况调整）
df['diff'] = abs(df['cancer_mean'] - df['normal_mean'])
df_mfs = df.sort_values('diff', ascending=False).head(1200)

# 条形图
plt.figure(figsize=(10, 6))
# 绘制癌症与正常的条形图
plt.bar(df_mfs['m/z'] - 0.2, df_mfs['cancer_mean'], width=0.4, label='Cancer', color='red', align='center')
plt.bar(df_mfs['m/z'] + 0.2, df_mfs['normal_mean'], width=0.4, label='Control', color='blue', align='center')
plt.xlabel('m/z')
plt.ylabel('Normalised Abundance')
plt.title('Metabolic Fingerprints of Cancer and Control')
plt.legend()
plt.grid(True)
plt.tight_layout()
# 保存图像
plt.savefig(os.path.join(save_path, 'cancer_normal_selected_mz_diff.png'), dpi=300)
plt.show()

# 划分训练集和测试集，只需run一次，就保存到save_path里储存为npy了
split_dataset(mz_benign, mz_cancer, save_path, test_size=0.3, random_state=42)
# 读取训练集和测试集
X_train_mz = np.load(f"{save_path}/X_train_mz.npy")
X_test_mz = np.load(f"{save_path}/X_test_mz.npy")
y_train_mz = np.load(f"{save_path}/y_train_mz.npy")
y_test_mz = np.load(f"{save_path}/y_test_mz.npy")


'''
# 训练 PCA-LDA 模型并保存模型参数，只需run一次，后续测试会自己读取train保存的参数
train_pca_lda_model(X_train_mz, y_train_mz, save_path, n_pca_components=20) #这里报错， The number of samples must be more than the number of classes.
test_pca_lda_model(X_test_mz, y_test_mz, save_path, show_plot=True)
train_pca_rf_model(X_train_mz, y_train_mz, save_path, random_state=42, n_pca_components=20, n_estimators=200, max_depth=10)
test_pca_rf_model(X_test_mz, y_test_mz, save_path, show_plot=True)
train_svm_model(X_train_mz, y_train_mz, save_path, kernel='rbf', C=1.0, gamma='scale')
test_svm_model(X_test_mz, y_test_mz, save_path, show_plot=False)
train_cnn_model(X_train_mz, y_train_mz, save_path, epochs=100, batch_size=32, lr=0.001)
test_cnn_model(X_test_mz, y_test_mz, save_path, batch_size=32, show_plot=False)
'''

'''
# 读取数据
file_path = r'N:\hlt\FTIR\result\FNA_supernatant\compound measurements.xlsx'
df = pd.read_excel(file_path, header=1)  # 从第二行读取数据
# 筛选 m/z 范围在 100-800 之间的数据
# df_filtered = df[(df['m/z'] >= 100) & (df['m/z'] <= 800)]
df_filtered = df

# 条形图
plt.figure(figsize=(10, 6))
# 绘制癌症与正常的条形图
plt.bar(df_filtered['m/z'] - 0.2, df_filtered['cancer'], width=0.4, label='Cancer', color='red', align='center')
plt.bar(df_filtered['m/z'] + 0.2, df_filtered['normal'], width=0.4, label='Control', color='blue', align='center')
plt.xlabel('m/z')
plt.ylabel('Normalised Abundance')
plt.title('Metabolic Fingerprints of Cancer and Control')
plt.legend()
plt.grid(True)
plt.tight_layout()
# 保存图像
plt.savefig(r'N:\hlt\FTIR\result\FNA_supernatant\cancer_normal_mz_0_4000.png')
plt.show()
'''