import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from classification_metrics import classification_metrics
from save_confusion_matrix_heatmap import save_confusion_matrix_heatmap
from split_dataset import split_dataset


def perform_cnn_analysis(spectrum_benign, spectrum_441, spectrum_520, spectrum_1299, save_path,
                         test_size=0.3, random_state=42, epochs=100, batch_size=32):
    """
    执行CNN分类分析
    参数:
    - spectrum_benign, spectrum_441, spectrum_520, spectrum_1299: 各类别的光谱数据
    - save_path: 结果保存路径
    - test_size: 测试集比例
    - random_state: 划分训练集、测试集时的随机种子
    - epochs: 训练轮数
    - batch_size: 批量大小
    """

    # 划分数据集
    X_train_scaled, X_test_scaled, y_train, y_test = split_dataset(spectrum_benign, spectrum_441,
                                                                   spectrum_520, spectrum_1299, test_size,
                                                                   random_state)

    # 调整数据形状以适应CNN (samples, timesteps, features)
    X_train_scaled = X_train_scaled[..., np.newaxis]
    X_test_scaled = X_test_scaled[..., np.newaxis]

    # 将标签进行独热编码
    num_classes = 4
    y_train_categorical = to_categorical(y_train, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test, num_classes=num_classes)

    # 构建CNN模型
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 早停回调以防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    history = model.fit(
        X_train_scaled, y_train_categorical,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # 预测
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵热力图，传递 method_name
    save_confusion_matrix_heatmap(cm, save_path, method_name='CNN', show_plot=True)
    # 计算评价指标，保存到excel中
    classification_metrics(cm, y_test, y_pred, save_path, excel_filename='CNN_metrics.xlsx')