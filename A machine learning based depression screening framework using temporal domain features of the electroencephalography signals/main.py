import numpy as np
import pandas as pd
import os
import scipy.io as sio
from tqdm import tqdm
from eeg_processor import EEGProcessor
from sklearn.preprocessing import StandardScaler
from classifier import DepressionClassifier

# --------------------------
# 1. 全局配置
# --------------------------
DATA_DIR = 'D:\山东第一医科大学\数据\兰州大学抑郁数据集\EEG_128channels_resting_lanzhou_2015'  # 请将您的 .mat 文件放入此目录
CH_NAMES = ['Fp1', 'Fp2', 'T7', 'T8', 'O1', 'O2']
CH_NUMBERS = ['22', '9', '45', '108', '70', '83']
SFREQ = 250  # 采样率
WINDOW_SEC = 10  # 论文建议的窗口大小
TRAIN_RATIO = 0.8

# --------------------------
# 2. 工具函数：从文件名生成 .mat 字段名
# --------------------------
def get_mat_field_name(filename):
    """
    从文件名自动生成对应的 .mat 数据字段名
    示例：
    输入 filename = "02010005rest 20150507 0907..mat"
    输出 field_name = "a02010005rest_20150507_0907mat"
    """
    # 步骤1：去掉文件后缀（.mat）
    name_without_ext = filename.replace('..mat', '').replace('.mat', '')

    # 步骤2：去掉多余的点
    name_without_dot = name_without_ext.replace('..', '')

    # 步骤3：将空格替换为下划线
    name_underscore = name_without_dot.replace(' ', '_')

    # 步骤4：开头添加字母 a
    field_name = 'a' + name_underscore + 'mat'
    return field_name


# --------------------------
# 3. 数据加载逻辑
# --------------------------
def load_and_process_mat_files(data_dir):
    """
    批量加载 .mat 文件并提取指定通道
    """
    subjects_data = []
    labels = []

    if not os.path.exists(data_dir):
        print(f"错误：目录 {data_dir} 不存在。")
        return [], []

    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    print(f"找到 {len(mat_files)} 个 .mat 文件，开始处理...")

    for filename in tqdm(mat_files, desc="加载数据"):
        file_path = os.path.join(data_dir, filename)
        try:
            mat_data = sio.loadmat(file_path)
            target_field = get_mat_field_name(filename)

            # 提取数据字段
            if target_field not in mat_data:
                valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if not valid_keys: continue
                target_field = valid_keys[-1]

            eeg_struct = mat_data[target_field]
            #print(eeg_struct.shape)

            # 解析结构化数组
            if isinstance(eeg_struct, np.ndarray) and eeg_struct.dtype == 'object':
                eeg_data = eeg_struct[0, 0]
                if 'data' in eeg_data.dtype.names:
                    eeg_data = eeg_data['data']
                else:
                    eeg_data = eeg_data.item()
            else:
                eeg_data = eeg_struct

            eeg_data = np.array(eeg_data)


            # 确保通道数在前 (n_channels, n_samples)
            if eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T

            # 提取指定通道 (128 导 -> 6 导)
            ch_indices = [int(num) - 1 for num in CH_NUMBERS]
            eeg_selected = eeg_data[ch_indices, 0:75000]

            subjects_data.append(eeg_selected)
            # 标签识别：0201 为抑郁 (1)，0202 为健康 (0)
            if filename.startswith('0201'):
                labels.append(1)
            elif filename.startswith('0202'):
                labels.append(0)
            else:
                labels.append(0)  # 默认健康

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    return subjects_data, labels


# --------------------------
# 4. 主程序
# --------------------------
def main():
    print("--- 正在初始化抑郁筛查框架复现程序 (适配 128 导数据) ---")
    processor = EEGProcessor(fs=SFREQ, ch_names=CH_NAMES)
    classifier = DepressionClassifier()

    # 1. 加载数据
    if os.path.exists(DATA_DIR) and any(f.endswith('.mat') for f in os.listdir(DATA_DIR)):
        subjects_data, subject_labels = load_and_process_mat_files(DATA_DIR)
        print(np.array(subjects_data).shape)
    else:
        print(f"未在 {DATA_DIR} 找到数据，请确保路径正确。")
        return

    if not subjects_data:
        print("未加载到有效数据。")
        return

    # 2.划分训练集和测试集(按受试者划分)
    n_subjects = len(subjects_data)
    n_train = int(n_subjects * TRAIN_RATIO)

    print(f"2. 正在按受试者划分数据集：训练集 {n_train} 人，测试集 {n_subjects - n_train} 人...")
    from sklearn.utils import shuffle

    subjects_data, subject_labels = shuffle(
        subjects_data,
        subject_labels,
        random_state=42
    )

    train_data = subjects_data[:n_train]
    train_labels = subject_labels[:n_train]

    test_data = subjects_data[n_train:]
    test_labels = subject_labels[n_train:]

    # 3. 特征提取 (训练集)
    print(f"3. 正在提取训练集特征 ({WINDOW_SEC}秒窗口)...")
    X_train_list = []
    y_train_list = []
    for data, label in zip(train_data, train_labels):
        subject_feats = processor.process_subject(data, window_sec=WINDOW_SEC)
        X_train_list.extend(subject_feats)
        y_train_list.extend([label] * len(subject_feats))

    df_train = pd.DataFrame(X_train_list)
    y_train = np.array(y_train_list)

    # 4. 特征提取 (测试集)
    print(f"4. 正在提取测试集特征 ({WINDOW_SEC}秒窗口)...")
    X_test_list = []
    y_test_list = []
    for data, label in zip(test_data, test_labels):
        subject_feats = processor.process_subject(data, window_sec=WINDOW_SEC)
        X_test_list.extend(subject_feats)
        y_test_list.extend([label] * len(subject_feats))

    df_test = pd.DataFrame(X_test_list)
    y_test = np.array(y_test_list)

    # 5. 特征选择
    print("5. 正在根据论文逻辑筛选关键特征...")
    X_train_selected = classifier.get_paper_selected_features(df_train, ch_names=CH_NAMES)
    X_test_selected = classifier.get_paper_selected_features(df_test, ch_names=CH_NAMES)

    print(f"   训练集特征形状: {X_train_selected.shape}, 测试集特征形状: {X_test_selected.shape}")

    # 6. 特征标准化（防止数据泄露）
    scaler = StandardScaler()

    X_train_selected = scaler.fit_transform(X_train_selected)
    X_test_selected = scaler.transform(X_test_selected)

    # 7. 模型训练与评估
    print("6. 正在评估模型性能 (KNN, AdaBoost, BF-Tree)...")
    results = classifier.evaluate_on_split(
        X_train_selected,
        y_train,
        X_test_selected,
        y_test
    )

    print("\n--- 复现结果评估报告 (基于受试者划分，无数据泄露) ---")
    print(results.to_markdown(index=False))

    print("\n注：由于按受试者划分，测试集包含的是模型从未见过的受试者，这比传统的交叉验证更具挑战性，也更符合临床实际。")


if __name__ == "__main__":
    main()