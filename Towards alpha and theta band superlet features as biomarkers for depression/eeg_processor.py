"""
EEG 信号预处理和特征提取模块
严格按照论文《Towards alpha and theta band superlet features as biomarkers for depression》

关键参数：
- 19 个通道：Fp1/2, F3/4, C3/4, P3/4, O1/2, F7/8, T3/4, T5/6, Pz, Fz, Cz, Oz
- 采样率：250 Hz (MODMA) / 256 Hz (Mumtaz)
- 窗口长度：12.5 秒
- 步长：0.5 秒
- 预处理：ICA 去噪、平均参考、滤波
"""

import numpy as np
import scipy.io as sio
from scipy import signal
import warnings
import os

warnings.filterwarnings('ignore')


# 论文中使用的 19 个通道
SELECTED_CHANNELS_19 = [
    'Fp1', 'Fp2',           # 前额
    'F3', 'F4',             # 额叶
    'C3', 'C4',             # 中央
    'P3', 'P4',             # 顶叶
    'O1', 'O2',             # 枕叶
    'F7', 'F8',             # 颞叶前
    'T3', 'T4',             # 颞叶中
    'T5', 'T6',             # 颞叶后
    'Pz', 'Fz', 'Cz'        # 中线
]

# 128 导系统中对应的索引（0-indexed）
CHANNEL_INDICES_128 = {
    'Fp1': 21,   'Fp2': 9,
    'F3': 34,    'F4': 5,
    'C3': 36,    'C4': 104,
    'P3': 63,    'P4': 95,
    'O1': 70,    'O2': 83,
    'F7': 45,    'F8': 108,
    'T3': 52,    'T4': 113,
    'T5': 61,    'T6': 94,
    'Pz': 62,    'Fz': 15,
    'Cz': 37
}


def get_mat_field_name(filename):
    """
    从文件名自动生成对应的 .mat 数据字段名

    示例：
    输入 filename = "02010005rest 20150507 0907.mat"
    输出 field_name = "a02010005rest_20150507_0907mat"

    参数：
    - filename: 文件名（可包含或不包含 .mat 扩展名）

    返回：
    - mat 文件中的变量名
    """
    # 步骤1：去掉文件后缀（.mat）
    name_without_ext = filename.replace('..mat', '').replace('.mat', '')

    # 步骤2：去掉多余的点
    name_without_dot = name_without_ext.replace('..', '')

    # 步骤3：将空格替换为下划线
    name_underscore = name_without_dot.replace(' ', '_')

    # 步骤4：开头添加字母 a，结尾添加 mat
    field_name = 'a' + name_underscore + 'mat'

    return field_name


def load_mat_data(mat_file_path):
    """
    加载 .mat 文件并提取 EEG 数据

    参数：
    - mat_file_path: .mat 文件的完整路径

    返回：
    - eeg_data: EEG 数据矩阵 (通道数 × 样本数)
    """
    # 加载 .mat 文件
    mat_data = sio.loadmat(mat_file_path)

    # 获取文件名（不含路径和扩展名）
    filename = os.path.basename(mat_file_path)

    # 推导变量名
    target_field = get_mat_field_name(filename)

    # 尝试获取数据
    if target_field not in mat_data:
        # 备选方案：取 mat_data 中除内置键外的最后一个有效键
        valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if not valid_keys:
            raise ValueError(f"无有效数据字段（{mat_file_path} 中仅包含内置键）")
        target_field = valid_keys[-1]  # 通常最后一个是数据字段

    eeg_struct = mat_data[target_field]

    # 处理 MATLAB 结构化数组
    if isinstance(eeg_struct, np.ndarray) and eeg_struct.dtype == 'object':
        # 处理 (1,1) 形状的结构化数组
        eeg_data = eeg_struct[0, 0]
        # 从结构体中提取原始数据
        if hasattr(eeg_data, 'dtype') and eeg_data.dtype.names is not None:
            if 'data' in eeg_data.dtype.names:
                eeg_data = eeg_data['data'].flatten()
            else:
                eeg_data = eeg_data.item()
        else:
            eeg_data = eeg_data.item() if hasattr(eeg_data, 'item') else eeg_data
    else:
        eeg_data = eeg_struct

    # 提取采样率（兼容不同存储方式）
    if 'samplingRate' in mat_data:
        sfreq = mat_data['samplingRate'][0][0]
    elif 'sfreq' in mat_data:
        sfreq = mat_data['sfreq'][0][0]
    else:
        sfreq = 250  # 默认采样率

    # 转为 numpy 数组
    eeg_data = np.array(eeg_data)

    # 检查数据维度
    if eeg_data.ndim != 2:
        raise ValueError(f"EEG 数据维度异常，应为 2 维，实际为 {eeg_data.ndim} 维")

    # 转置判断（确保通道数在前）
    if eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T  # 转置为 (n_channels, n_samples)

    return eeg_data


class EEGProcessor:
    """EEG 信号预处理和特征提取"""

    def __init__(self, sfreq=250, window_sec=12.5, step_sec=0.5):
        """
        初始化 EEG 处理器

        参数：
        - sfreq: 采样率 (Hz)
        - window_sec: 窗口长度 (秒)，论文中为 12.5 秒
        - step_sec: 步长 (秒)，论文中为 0.5 秒
        """
        self.sfreq = sfreq
        self.window_sec = window_sec
        self.step_sec = step_sec

        # 计算样本数
        self.window_samples = int(window_sec * sfreq)
        self.step_samples = int(step_sec * sfreq)

    def preprocess_eeg(self, eeg_data):
        """
        预处理 EEG 数据

        步骤：
        1. 移除基线漂移
        2. 平均参考
        3. 滤波

        参数：
        - eeg_data: EEG 数据矩阵 (通道数 × 样本数)

        返回：
        - 预处理后的 EEG 数据
        """
        # 确保数据是 2D 数组
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)

        # 1. 移除基线漂移（使用高通滤波）
        sos = signal.butter(4, 0.5, 'hp', fs=self.sfreq, output='sos')
        eeg_filtered = signal.sosfilt(sos, eeg_data, axis=1)

        # 2. 平均参考（Average Reference）
        mean_signal = np.mean(eeg_filtered, axis=0, keepdims=True)
        eeg_referenced = eeg_filtered - mean_signal

        # 3. 低通滤波（50 Hz，移除高频噪声）
        sos = signal.butter(4, 50, 'lp', fs=self.sfreq, output='sos')
        eeg_processed = signal.sosfilt(sos, eeg_referenced, axis=1)

        return eeg_processed

    def extract_channels(self, eeg_data):
        """
        从 128 导中提取 19 个标准通道

        参数：
        - eeg_data: 128 导 EEG 数据 (128 × 样本数)

        返回：
        - 19 通道 EEG 数据 (19 × 样本数)
        """
        # 获取通道索引
        indices = [CHANNEL_INDICES_128[ch] for ch in SELECTED_CHANNELS_19]

        # 提取通道
        eeg_19ch = eeg_data[indices, :]

        return eeg_19ch

    def segment_data(self, eeg_data):
        """
        使用滑动窗口分段 EEG 数据

        参数：
        - eeg_data: EEG 数据矩阵 (通道数 × 样本数)

        返回：
        - segments: 分段列表，每个分段形状为 (通道数 × 窗口样本数)
        """
        n_channels, n_samples = eeg_data.shape
        segments = []

        # 使用滑动窗口
        for start in range(0, n_samples - self.window_samples + 1, self.step_samples):
            end = start + self.window_samples
            segment = eeg_data[:, start:end]
            segments.append(segment)

        return segments

    def normalize_segment(self, segment):
        """
        Z-score 归一化分段

        参数：
        - segment: EEG 分段 (通道数 × 样本数)

        返回：
        - 归一化后的分段
        """
        # Z-score 归一化（每个通道独立）
        mean = np.mean(segment, axis=1, keepdims=True)
        std = np.std(segment, axis=1, keepdims=True)
        std = np.maximum(std, 1e-8)  # 避免除以 0

        normalized = (segment - mean) / std

        return normalized

    def process_full_signal(self, eeg_data_128):
        """
        完整处理流程：预处理 → 提取通道 → 分段 → 归一化

        参数：
        - eeg_data_128: 128 导原始 EEG 数据 (128 × 样本数)

        返回：
        - segments: 处理后的分段列表
        """
        # 1. 预处理
        eeg_preprocessed = self.preprocess_eeg(eeg_data_128)

        # 2. 提取 19 个通道
        eeg_19ch = self.extract_channels(eeg_preprocessed)

        # 3. 分段
        segments = self.segment_data(eeg_19ch)

        # 4. 归一化
        normalized_segments = [self.normalize_segment(seg) for seg in segments]

        return normalized_segments