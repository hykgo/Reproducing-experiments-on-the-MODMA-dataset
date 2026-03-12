import numpy as np
import pandas as pd
from scipy.signal import iirnotch, lfilter, savgol_filter
from scipy.stats import skew, kurtosis


class EEGProcessor:
    def __init__(self, fs=250, ch_names=['Fp1', 'Fp2', 'T7', 'T8', 'O1', 'O2']):
        """
        初始化 EEG 处理器
        :param fs: 采样频率
        :param ch_names: 通道名称列表
        """
        self.fs = fs
        self.ch_names = ch_names

    def apply_preprocessing(self, data):
        """
        应用预处理步骤：平均参考、50Hz 陷波滤波、平滑滤波
        :param data: 形状为 (n_channels, n_samples) 的 numpy 数组
        :return: 预处理后的数据
        """
        # 1. 平均参考 (Average Reference)
        avg_ref = np.mean(data, axis=0)
        data_ref = data - avg_ref

        # 2. 50Hz 陷波滤波 (Notch Filter)
        b, a = iirnotch(50.0, 30.0, self.fs)
        data_notch = lfilter(b, a, data_ref, axis=1)

        # 3. 三阶平滑滤波 (Third-order smoothing filter)
        # 使用 Savitzky-Golay 滤波器模拟论文中的平滑处理
        data_smooth = savgol_filter(data_notch, window_length=5, polyorder=3, axis=1)

        return data_smooth

    def extract_temporal_features(self, segment):
        """
        提取 12 个时间域特征
        :param segment: 形状为 (n_samples,) 的单通道数据段
        :return: 包含 12 个特征的字典
        """
        features = {}

        # 1. Maximum Amplitude
        features['max_amp'] = np.max(segment)

        # 2. Minimum Amplitude
        features['min_amp'] = np.min(segment)

        # 3. Mean Value
        features['mean_val'] = np.mean(segment)

        # 4. Standard Deviation
        features['std_dev'] = np.std(segment)

        # 5. Kurtosis
        features['kurtosis'] = kurtosis(segment)

        # 6. Skewness
        features['skewness'] = skew(segment)

        # 7. Peak to Peak Signal Value
        features['p2p_val'] = features['max_amp'] - features['min_amp']

        # 8. Peak to Peak Time
        pos_peak_idx = np.argmax(segment)
        neg_peak_idx = np.argmin(segment)
        features['p2p_time'] = abs(pos_peak_idx - neg_peak_idx) / self.fs

        # 9. Mean of absolute values of first difference
        features['mean_abs_1st_diff'] = np.mean(np.abs(np.diff(segment)))

        # 10. Mean of absolute values of second difference
        features['mean_abs_2nd_diff'] = np.mean(np.abs(np.diff(segment, n=2)))

        # 11. Energy
        features['energy'] = np.sum(np.square(segment)) / len(segment)

        # 12. Shannon Entropy
        # 使用直方图估计概率分布
        hist, _ = np.histogram(segment, bins=100, density=True)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        features['shannon_entropy'] = -np.sum(prob * np.log2(prob))

        return features

    def process_subject(self, data, window_sec=10):
        """
        处理单个受试者的数据，进行分窗并提取特征
        :param data: (n_channels, n_samples) 数组
        :param window_sec: 窗口大小（秒）
        :return: 特征列表
        """
        n_channels, n_samples = data.shape
        #print(n_samples)
        win_samples = int(window_sec * self.fs)
        n_windows = n_samples // win_samples
        #print(n_windows) 30个窗
        all_window_features = []

        processed_data = self.apply_preprocessing(data)

        for i in range(n_windows):
            start = i * win_samples
            end = start + win_samples
            win_data = processed_data[:, start:end]

            win_features = {}
            for ch_idx, ch_name in enumerate(self.ch_names):
                ch_feats = self.extract_temporal_features(win_data[ch_idx]) #得到每一个通道的每个窗口所有特征
                for feat_name, val in ch_feats.items():#把每个特征拿出来放到win_features
                    win_features[f"{ch_name}_{feat_name}"] = val

            all_window_features.append(win_features)#得到这一个被试的30个窗口
        #print(np.array(all_window_features))

        return all_window_features

