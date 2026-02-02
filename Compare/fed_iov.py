import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 配置部分
# ==========================================

CONFIG = {
    # 请修改为您实际的路径
    'dataset_dir': r'F:\Phd\Third\Work\IoV修改\Dataset\DDoS_Multi', 
    'tfrecord_dir': r'.\fed_iov_tfrecord_simple', # 修改文件夹名以防与旧数据混淆
    'train_tfrecord_name': 'train_data.tfrecords',
    'test_tfrecord_name': 'test_data.tfrecords',
    'batch_size': 256, # 将 batch_size 放入配置中
    'label_map': {"DrDos_NTP": 0, "MSSQL": 1, "NetBIOS": 2, "LDAP": 3, "Syn": 4, "UDP": 5},
    'sample_counts': {
        "DrDos_NTP": 7903, "MSSQL": 21987, "NetBIOS": 24760, 
        "LDAP": 25250, "Syn": 25190, "UDP": 24510
    }
}

# ==========================================
# 第一部分：数据处理 (CSV -> TFRecord)
# ==========================================

def process_csv_data():
    all_data = []
    all_labels = []
    
    print(f"开始从目录 {CONFIG['dataset_dir']} 处理 CSV 数据...")
    
    for name, label_id in CONFIG['label_map'].items():
        file_path = os.path.join(CONFIG['dataset_dir'], f"{name}.csv")
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 {file_path}")
            continue
            
        df = pd.read_csv(file_path, low_memory=False)

        # 数据清洗逻辑
        if df.shape[1] >= 86:
            df = df.drop(columns=[df.columns[85]])
        
        features = df.iloc[:, 8:-1]
        features = features.apply(pd.to_numeric, errors='coerce')
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        target_count = CONFIG['sample_counts'][name]
        current_count = len(features)

        if current_count > target_count:
            print(f"  - {name}: 总数据 {current_count} 条 -> 随机采样 {target_count} 条")
            # random_state=42 保证实验可复现，如果需要完全随机请去掉该参数
            features = features.sample(n=target_count, random_state=42)
        else:
            print(f"  - {name}: 总数据 {current_count} 条 (不足目标 {target_count}) -> 取全部并打乱")
            # frac=1 表示取全部，但会打乱顺序
            features = features.sample(frac=1, random_state=42)
        
        data_np = features.values.astype(np.float32)
        labels_np = np.full((len(data_np),), label_id, dtype=np.int32)
        
        all_data.append(data_np)
        all_labels.append(labels_np)

    if not all_data:
        raise ValueError("未读取到任何数据，请检查路径。")

    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print("正在进行归一化处理 (MinMax [0, 1])...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0.0)

    # 填充或截断特征到 81 (9x9)
    if X.shape[1] < 81:
        padding = np.zeros((X.shape[0], 81 - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, padding])
    elif X.shape[1] > 81:
        X = X[:, :81]
        
    return X, y

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecords(X, y, filepath):
    print(f"正在写入 {filepath} ...")
    with tf.io.TFRecordWriter(filepath) as writer:
        for i in range(len(X)):
            feature = X[i].tobytes()
            label = y[i]
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'feature': _bytes_feature(feature),
                'label': _int64_feature(label)
            }))
            writer.write(example.SerializeToString())
    print(f"{filepath} 写入完成。")

def prepare_data():
    if os.path.exists(CONFIG['tfrecord_dir']):
        import shutil
        print(f"清理旧数据目录: {CONFIG['tfrecord_dir']}")
        shutil.rmtree(CONFIG['tfrecord_dir'])
    
    os.makedirs(CONFIG['tfrecord_dir'])

    train_path = os.path.join(CONFIG['tfrecord_dir'], CONFIG['train_tfrecord_name'])
    test_path = os.path.join(CONFIG['tfrecord_dir'], CONFIG['test_tfrecord_name'])

    X, y = process_csv_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    write_tfrecords(X_train, y_train, train_path)
    write_tfrecords(X_test, y_test, test_path)
    
    return train_path, test_path

# ==========================================
# 第二部分：数据管道 (修正为不 Resize)
# ==========================================

def parse_tfrecord_fn(example_proto):
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    feature = tf.io.decode_raw(parsed_features['feature'], tf.float32)
    # 保持原始 9x9x1 形状
    feature = tf.reshape(feature, [9, 9, 1]) 
    label = parsed_features['label']
    return feature, label

def image_transform_fn(feature, label):
    """
    修正后的转换函数：
    1. 移除 RGB 转换
    2. 移除 Resize
    3. 保持数据为 float32，范围 [0, 1] (已经在 process_csv_data 中做过 MinMax)
    """
    # 确保数据类型正确
    feature = tf.cast(feature, tf.float32)
    return feature, label

def get_dataset(tfrecord_path, batch_size=32, is_training=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(image_transform_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def get_numpy_data_from_dataset(dataset):
    all_images = []
    all_labels = []
    print("正在提取测试集数据用于评估...")
    for images, labels in dataset:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_images, axis=0), np.concatenate(all_labels, axis=0)

# ==========================================
# 第三部分：简单 CNN 模型定义
# ==========================================

def build_simple_cnn(input_shape=(9, 9, 1), num_classes=6):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        
        # 第一层卷积：保持尺寸 (9, 9)
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        # 第二层卷积：保持尺寸 (9, 9)
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        # 池化层：尺寸变为 (4, 4)
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # 第三层卷积：(4, 4) -> (4, 4)
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        
        # 展平
        tf.keras.layers.Flatten(),
        
        # 全连接层
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        
        # 输出层
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name="Simple_CNN_9x9")
    return model

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    # 强制重新生成数据，以确保数据格式正确 (从之前的 Resize 版本切换过来必须重新生成)
    FORCE_REGENERATE = True 
    
    train_tfrecord_path = os.path.join(CONFIG['tfrecord_dir'], CONFIG['train_tfrecord_name'])
    test_tfrecord_path = os.path.join(CONFIG['tfrecord_dir'], CONFIG['test_tfrecord_name'])

    if FORCE_REGENERATE or not os.path.exists(train_tfrecord_path):
        print("准备生成数据...")
        try:
            train_tfrecord_path, test_tfrecord_path = prepare_data()
        except Exception as e:
            print(f"数据处理失败: {e}")
            exit()
    else:
        print(f"直接加载 TFRecord: {CONFIG['tfrecord_dir']}")

    print(f"加载训练集: {train_tfrecord_path}")
    print(f"加载测试集: {test_tfrecord_path}")
    
    # 获取 Dataset
    train_ds = get_dataset(train_tfrecord_path, batch_size=CONFIG['batch_size'], is_training=True)
    test_ds = get_dataset(test_tfrecord_path, batch_size=CONFIG['batch_size'], is_training=False)

    # 构建模型
    model = build_simple_cnn(input_shape=(9, 9, 1), num_classes=len(CONFIG['label_map']))
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 训练
    EPOCHS = 10 
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS
    )

    # 评估
    print("\n开始评估...")
    Y_pred , Y_true = [],[]
    s2 = time.time()
    for features,labels in test_ds:
        Y_ = model.predict(features)
        Y_ = np.argmax(Y_, axis=-1)
        Y_pred.append(Y_)
        Y_true.append(labels.numpy()) 
    test_time = (time.time() - s2) 
    print('test time:', test_time, 'second')
    Y_true = np.concatenate(Y_true)
    Y_pred = np.concatenate(Y_pred)

    target_names = list(CONFIG['label_map'].keys())
    
    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(Y_true, Y_pred, target_names=target_names, digits=4))

    # 绘制混淆矩阵
    try:
        cm = confusion_matrix(Y_true, Y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Simple CNN 9x9)')
        plt.show()
    except Exception as e:
        print(f"绘图错误: {e}")