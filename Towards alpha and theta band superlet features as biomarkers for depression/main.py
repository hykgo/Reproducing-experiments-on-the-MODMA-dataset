"""
主程序：完整的 Superlet 变换抑郁症筛查框架复现
严格按照论文《Towards alpha and theta band superlet features as biomarkers for depression》

执行流程：
1. 加载 MODMA 数据集
2. 预处理 EEG 数据
3. 提取 Superlet 特征
4. 按被试进行训练/测试集划分
5. 训练 SVM 分类器并评估性能
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from eeg_processor import EEGProcessor, get_mat_field_name, load_mat_data, SELECTED_CHANNELS_19
from feature_extractor import FeatureExtractor, extract_features_for_subject
from classifier import SVMClassifier, cross_validate_svm, compute_metrics

# 全局配置
CONFIG = {
    'MODMA_DATA_DIR': 'D:\山东第一医科大学\数据\兰州大学抑郁数据集\EEG_128channels_resting_lanzhou_2015',  # MODMA 数据集目录（相对路径）
    'MODMA_SFREQ': 250,  # MODMA 采样率
    'WINDOW_SEC': 12,  # 窗口长度（秒）
    'STEP_SEC': 8,  # 步长（秒）
    'SVM_C': 50,  # SVM 参数 C
    'SVM_GAMMA': 'scale',  # SVM gamma 参数
    'N_SPLITS': 10,  # 交叉验证折数
    'TRAIN_RATIO': 0.8  # 训练集比例（按被试）
}


def get_subject_id_from_filename(filename):
    """
    从文件名提取受试者 ID

    例如：
    - 02010002rest_20150416_1017.mat → 02010002
    - 02010004rest_20150427_1335.mat → 02010004
    """
    # 移除扩展名
    if filename.endswith('.mat'):
        filename = filename[:-4]

    # 提取前 8 位作为受试者 ID
    subject_id = filename[:8]

    return subject_id


def get_label_from_subject_id(subject_id):
    """
    根据受试者 ID 推断标签

    - 0201 开头：MDD（抑郁症患者）= 1
    - 0202 开头：HC（健康对照）= 0
    """
    if subject_id.startswith('0201'):
        return 1  # MDD
    else:
        return 0  #HC


def load_and_process_dataset(data_dir, sfreq, max_files=None):
    """
    加载和处理数据集

    参数：
    - data_dir: 数据集目录
    - sfreq: 采样率
    - max_files: 最多加载的文件数（用于测试）

    返回：
    - X: 特征矩阵
    - y: 标签向量
    - subject_ids: 受试者 ID 向量
    - feature_names: 特征名称列表
    """
    # 获取所有 .mat 文件
    mat_files = list(Path(data_dir).glob('*.mat'))

    if max_files is not None:
        mat_files = mat_files[:max_files]

    if len(mat_files) == 0:
        raise ValueError(f"在 {data_dir} 中未找到 .mat 文件")

    print(f"找到 {len(mat_files)} 个 .mat 文件")

    # 初始化处理器和特征提取器
    processor = EEGProcessor(sfreq=sfreq, window_sec=CONFIG['WINDOW_SEC'], step_sec=CONFIG['STEP_SEC'])
    extractor = FeatureExtractor(sfreq=sfreq)

    # 存储所有特征和标签
    all_features = []
    all_labels = []
    all_subject_ids = []
    feature_names = None

    # 处理每个文件
    for mat_file in tqdm(mat_files, desc="处理数据"):
        try:
            # 加载数据
            print("加载数据")
            eeg_data = load_mat_data(str(mat_file))
            #print(eeg_data.shape)
            # 处理 EEG 数据
            segments = processor.process_full_signal(eeg_data[:,0:75000])

            if len(segments) == 0:
                print(f"警告：{mat_file.name} 未产生任何分段，跳过")
                continue
            print("提取特征")
            # 提取特征
            features_list = extractor.extract_features_all_segments(
                segments, channel_names=SELECTED_CHANNELS_19, verbose=False
            )
            print("转换为数组")
            # 转换为数组
            X_subject, feature_names = extractor.features_to_array(features_list)
            print("获取标签")
            # 获取标签
            subject_id = get_subject_id_from_filename(mat_file.name)
            label = get_label_from_subject_id(subject_id)

            # 添加到总体数据
            all_features.append(X_subject)
            all_labels.extend([label] * X_subject.shape[0])
            all_subject_ids.extend([subject_id] * X_subject.shape[0])

        except Exception as e:
            print(f"错误处理 {mat_file.name}: {e}")
            continue

    # 合并所有特征
    X = np.vstack(all_features)
    y = np.array(all_labels)
    subject_ids = np.array(all_subject_ids)

    return X, y, subject_ids, feature_names


def train_test_split_by_subject(X, y, subject_ids, train_ratio=0.8):
    """
    按受试者进行训练/测试集划分（防止数据泄露）

    参数：
    - X: 特征矩阵 (样本数 × 特征数)
    - y: 标签向量
    - subject_ids: 受试者 ID 向量
    - train_ratio: 训练集比例

    返回：
    - X_train, X_test, y_train, y_test: 划分后的数据
    """
    # 获取唯一的受试者 ID
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    # 计算训练集受试者数
    n_train_subjects = int(n_subjects * train_ratio)

    # 随机选择训练集受试者
    np.random.seed(42)
    train_subjects = np.random.choice(unique_subjects, n_train_subjects, replace=False)

    # 划分数据
    train_mask = np.isin(subject_ids, train_subjects)
    test_mask = ~train_mask

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test, train_subjects, unique_subjects[~np.isin(unique_subjects, train_subjects)]


def evaluate_cross_subject(X_train, y_train, X_test, y_test, train_subjects, test_subjects):
    """
    跨被试验证

    参数：
    - X_train: 训练特征
    - y_train: 训练标签
    - X_test: 测试特征
    - y_test: 测试标签
    - train_subjects: 训练集被试 ID
    - test_subjects: 测试集被试 ID

    返回：
    - metrics: 性能指标
    """
    print(f"\n{'=' * 60}")
    print(f"跨被试验证")
    print(f"{'=' * 60}")
    print(f"训练集被试数: {len(train_subjects)}")
    print(f"训练集样本数: {X_train.shape[0]}")
    print(f"测试集被试数: {len(test_subjects)}")
    print(f"测试集样本数: {X_test.shape[0]}")

    # 创建和训练分类器
    clf = SVMClassifier(kernel='rbf', C=CONFIG['SVM_C'], gamma=CONFIG['SVM_GAMMA'])
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)

    # 计算指标
    metrics = compute_metrics(y_test, y_pred)

    # 打印结果
    print(f"\n性能指标:")
    print(f"  准确率 (ACC): {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  特异性 (SPC): {metrics['specificity']:.4f} ({metrics['specificity'] * 100:.2f}%)")
    print(f"  敏感性 (SEN): {metrics['sensitivity']:.4f} ({metrics['sensitivity'] * 100:.2f}%)")
    print(f"  F1 分数: {metrics['f1_score']:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")

    return metrics


def main():
    """主程序"""
    print("=" * 60)
    print("Superlet 变换抑郁症筛查框架复现")
    print("论文：Towards alpha and theta band superlet features as biomarkers for depression")
    print("=" * 60)

    # 检查数据目录
    if not os.path.exists(CONFIG['MODMA_DATA_DIR']):
        print(f"\n错误：MODMA 数据目录不存在: {CONFIG['MODMA_DATA_DIR']}")
        print("请将 MODMA 数据集放在此目录中")
        return

    # 1. 加载和处理 MODMA 数据集
    print(f"\n1. 加载 MODMA 数据集...")
    try:
        X, y, subject_ids, feature_names = load_and_process_dataset(
            CONFIG['MODMA_DATA_DIR'],
            CONFIG['MODMA_SFREQ'],
            max_files=None  # 加载所有文件
        )
    except Exception as e:
        print(f"加载 MODMA 数据集失败: {e}")
        return

    print(f"\nMODMA 数据集统计:")
    print(f"  总样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  MDD 样本数: {np.sum(y == 1)}")
    print(f"  HC 样本数: {np.sum(y == 0)}")

    unique_subjects = np.unique(subject_ids)
    print(f"  总被试数: {len(unique_subjects)}")
    print(f"  MDD 被试数: {len(np.unique(subject_ids[y == 1]))}")
    print(f"  HC 被试数: {len(np.unique(subject_ids[y == 0]))}")

    # 2. 方法一：按被试的训练/测试集划分 + 单次评估
    print(f"\n2. 按被试进行训练/测试集划分...")
    X_train, X_test, y_train, y_test, train_subjects, test_subjects = train_test_split_by_subject(
        X, y, subject_ids, train_ratio=CONFIG['TRAIN_RATIO']
    )

    print(f"训练集: {len(train_subjects)} 个被试，{X_train.shape[0]} 个样本")
    print(f"测试集: {len(test_subjects)} 个被试，{X_test.shape[0]} 个样本")

    # 评估跨被试性能
    print(f"\n3. 评估跨被试性能...")
    cross_subject_metrics = evaluate_cross_subject(
        X_train, y_train, X_test, y_test, train_subjects, test_subjects
    )

    # 3. 方法二：10 折交叉验证（在所有被试上）
    print(f"\n4. 进行 10 折交叉验证...")
    print(f"{'=' * 60}")
    print(f"10 折交叉验证（所有被试）")
    print(f"{'=' * 60}")

    cv_results, cv_metrics = cross_validate_svm(X, y, n_splits=CONFIG['N_SPLITS'], C=CONFIG['SVM_C'])

    # 打印结果
    print(f"\n整体性能指标:")
    print(f"  准确率 (ACC): {cv_metrics['accuracy']:.4f} ({cv_metrics['accuracy'] * 100:.2f}%)")
    print(f"  特异性 (SPC): {cv_metrics['specificity']:.4f} ({cv_metrics['specificity'] * 100:.2f}%)")
    print(f"  敏感性 (SEN): {cv_metrics['sensitivity']:.4f} ({cv_metrics['sensitivity'] * 100:.2f}%)")
    print(f"  F1 分数: {cv_metrics['f1_score']:.4f}")

    # 打印每折结果
    print(f"\n每折结果:")
    for result in cv_results:
        print(f"  Fold {result['fold']:2d}: "
              f"ACC={result['accuracy']:.4f}, "
              f"SPC={result['specificity']:.4f}, "
              f"SEN={result['sensitivity']:.4f}")

    # 4. 总结
    print(f"\n{'=' * 60}")
    print("总结")
    print(f"{'=' * 60}")
    print(f"\n跨被试验证（训练/测试集划分）:")
    print(f"  准确率: {cross_subject_metrics['accuracy'] * 100:.2f}%")
    print(f"  特异性: {cross_subject_metrics['specificity'] * 100:.2f}%")
    print(f"  敏感性: {cross_subject_metrics['sensitivity'] * 100:.2f}%")

    print(f"\n10 折交叉验证（所有被试）:")
    print(f"  准确率: {cv_metrics['accuracy'] * 100:.2f}%")
    print(f"  特异性: {cv_metrics['specificity'] * 100:.2f}%")
    print(f"  敏感性: {cv_metrics['sensitivity'] * 100:.2f}%")

    print(f"\n论文报告的性能:")
    print(f"  MODMA（10 折交叉验证）: 97.57%")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()