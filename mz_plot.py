import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
file_path = r'N:\hlt\FTIR\result\FNA_supernatant\compound measurements.xlsx'
df = pd.read_excel(file_path, header=1)  # 从第二行读取数据

# 筛选 m/z 范围在 100-800 之间的数据
df_filtered = df[(df['m/z'] >= 100) & (df['m/z'] <= 800)]

# 条形图
'''
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
plt.savefig(r'N:\hlt\FTIR\result\FNA_supernatant\cancer_normal_mz100_800.png')
plt.show()
'''


# 热力图
# 创建一个二维的 DataFrame，其中包含 m/z 和 cancer/normal 的归一化丰度
heatmap_data = pd.DataFrame({
    'm/z': df_filtered['m/z'],
    'Cancer': df_filtered['cancer'],
    'Normal': df_filtered['normal']
})

# 将数据设置为以 m/z 为行，以 "Cancer" 和 "Normal" 为列
heatmap_data = heatmap_data.set_index('m/z')

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data.T, cmap='coolwarm', annot=False, cbar_kws={'label': 'Normalized Abundance'})

plt.title('Metabolic Fingerprints of Cancer and Control')
plt.xlabel('m/z')
plt.ylabel('Condition')

# 保存热力图
plt.tight_layout()
plt.savefig(r'N:\hlt\FTIR\result\FNA_supernatant\cancer_normal_mz100_800_heatmap.png')
plt.show()

