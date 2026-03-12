"""
多频段特征提取模块
严格按照论文《Towards alpha and theta band superlet features as biomarkers for depression》

特征提取步骤：
1. 对每个分段应用 Superlet 变换
2. 从 Delta/Theta/Alpha/Beta 四个频段提取特征
3. 每个频段提取：平均值 + 能量值
4. 总共 19 通道 × 4 频段 × 2 特征 = 152 个特征
"""

import numpy as np
from superlet import superlet_transform, extract_band_features, compute_time_frequency_power
from tqdm import tqdm


class FeatureExtractor:
    """多频段特征提取器"""

    def __init__(self, sfreq=250):
        """
        初始化特征提取器

        参数：
        - sfreq: 采样率 (Hz)
        """
        self.sfreq = sfreq

        # 定义频段
        self.bands = {
            'Delta': (0.5, 3),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30)
        }

        # 频率数组（1-40 Hz）
        self.frequencies = np.linspace(1, 40, 40)

    def extract_features_single_segment(self, segment, channel_names=None):
        """
        从单个分段提取特征

        参数：
        - segment: 分段数据 (通道数 × 样本数)
        - channel_names: 通道名称列表

        返回：
        - features: 特征字典
        """
        n_channels = segment.shape[0]

        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(n_channels)]

        features = {}

        # 对每个通道提取特征
        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx]
            ch_data = segment[ch_idx, :]

            # 应用 Superlet 变换
            # 参数：c1=6, order=6
            slt_matrix = superlet_transform(ch_data, self.sfreq, self.frequencies, c1=6, order=6)

            # 从每个频段提取特征
            for band_name, band_range in self.bands.items():
                band_features = extract_band_features(slt_matrix, self.frequencies, band_range)

                # 保存特征
                features[f"{ch_name}_{band_name}_mean"] = band_features['mean']
                features[f"{ch_name}_{band_name}_energy"] = band_features['energy']

        return features

    def extract_features_all_segments(self, segments, channel_names=None, verbose=True):
        """
        从所有分段提取特征

        参数：
        - segments: 分段列表，每个元素为 (通道数 × 样本数)
        - channel_names: 通道名称列表
        - verbose: 是否显示进度条

        返回：
        - all_features: 特征列表，每个元素为特征字典
        """
        all_features = []

        iterator = tqdm(segments, desc="提取特征") if verbose else segments

        for segment in iterator:
            features = self.extract_features_single_segment(segment, channel_names)
            all_features.append(features)

        return all_features

    def features_to_array(self, features_list):
        """
        将特征列表转换为数组

        参数：
        - features_list: 特征字典列表

        返回：
        - X: 特征矩阵 (样本数 × 特征数)
        - feature_names: 特征名称列表
        """
        if len(features_list) == 0:
            raise ValueError("特征列表为空")

        # 获取特征名称
        feature_names = list(features_list[0].keys())

        # 构建特征矩阵
        n_samples = len(features_list)
        n_features = len(feature_names)
        X = np.zeros((n_samples, n_features))

        for i, features in enumerate(features_list):
            for j, feat_name in enumerate(feature_names):
                X[i, j] = features[feat_name]

        return X, feature_names


def extract_features_for_subject(segments, sfreq=250, channel_names=None):
    """
    为单个受试者的所有分段提取特征

    参数：
    - segments: 分段列表
    - sfreq: 采样率
    - channel_names: 通道名称列表

    返回：
    - X: 特征矩阵 (样本数 × 特征数)
    - feature_names: 特征名称列表
    """
    extractor = FeatureExtractor(sfreq=sfreq)
    features_list = extractor.extract_features_all_segments(segments, channel_names, verbose=False)
    X, feature_names = extractor.features_to_array(features_list)

    return X, feature_names


# 测试代码
if __name__ == "__main__":
    from eeg_processor import EEGProcessor

    # 生成测试数据
    sfreq = 250
    duration = 60  # 60 秒
    n_samples = int(sfreq * duration)
    n_channels = 128

    # 生成随机 EEG 数据
    eeg_test = np.random.randn(n_channels, n_samples) * 10

    # 处理 EEG 数据
    processor = EEGProcessor(sfreq=sfreq, window_sec=12.5, step_sec=0.5)
    segments = processor.process_full_signal(eeg_test)

    print(f"分段数: {len(segments)}")

    # 定义通道名称（19 个通道）
    channel_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Pz', 'Fz', 'Cz'
    ]

    # 提取特征
    print("\n提取特征...")
    X, feature_names = extract_features_for_subject(segments, sfreq=sfreq, channel_names=channel_names)

    print(f"特征矩阵形状: {X.shape}")
    print(f"特征数: {len(feature_names)}")
    print(f"样本数: {X.shape[0]}")

    # 显示前几个特征名称
    print("\n前 10 个特征名称:")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {i + 1}. {name}")

    # 显示特征统计信息
    print("\n特征统计信息:")
    print(f"  最小值: {X.min():.4f}")
    print(f"  最大值: {X.max():.4f}")
    print(f"  平均值: {X.mean():.4f}")
    print(f"  标准差: {X.std():.4f}")