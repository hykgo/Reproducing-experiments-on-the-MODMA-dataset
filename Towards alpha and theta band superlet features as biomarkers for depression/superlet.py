"""
Superlet Transform (SLT) 实现
严格按照论文《Towards alpha and theta band superlet features as biomarkers for depression》
参数设置：c1=6（基础周期数），频率范围 1-40 Hz，阶数 o=6
"""

import numpy as np
from scipy.signal import fftconvolve
import warnings

warnings.filterwarnings('ignore')


def morlet_wavelet(t, f, c):
    """
    生成 Morlet 小波

    公式：ψ_r,z(t) = (1 / (B_c * sqrt(2π))) * exp(-t^2 / (2 * B_c^2)) * exp(i * 2π * f * t)

    参数：
    - t: 时间向量
    - f: 中心频率 (Hz)
    - c: 周期数 (cycles)

    返回：
    - 复数 Morlet 小波
    """
    # 时间标准差
    sigma = c / (2 * np.pi * f)

    # Morlet 小波 = 高斯包络 × 复指数振荡
    psi = np.exp(2j * np.pi * f * t) * np.exp(-t ** 2 / (2 * sigma ** 2))

    # 归一化以保持能量
    psi /= np.sqrt(np.sqrt(np.pi) * sigma)

    return psi


def superlet_transform(data, fs, frequencies, c1=6, order=6):
    """
    Superlet 变换 (SLT) 实现

    严格按照论文参数：
    - c1 = 6（基础周期数）
    - order = 6（Superlet 阶数）
    - 频率范围：1-40 Hz

    参数：
    - data: 1D 信号数组
    - fs: 采样频率 (Hz)
    - frequencies: 目标频率列表或数组
    - c1: 基础小波的周期数（论文中为 6）
    - order: Superlet 的阶数（论文中为 6）

    返回：
    - slt_matrix: 时频矩阵 (len(frequencies), len(data))
    """
    n_samples = len(data)
    dt = 1 / fs

    # 确保 frequencies 是数组
    if not isinstance(frequencies, np.ndarray):
        frequencies = np.array(frequencies)

    # 初始化 SLT 矩阵
    slt_matrix = np.zeros((len(frequencies), n_samples))

    # 对每个频率计算 Superlet 响应
    for freq_idx, f in enumerate(frequencies):
        # 周期数数组：c_i = i × c1，i = 1, 2, ..., order
        # 例如：c1=6, order=6 => [6, 12, 18, 24, 30, 36]
        c_values = np.arange(1, order + 1) * c1

        # 计算不同周期数对应的小波变换
        wavelet_responses = []

        for c in c_values:
            # 时间标准差
            sigma = c / (2 * np.pi * f)

            # 窗口大小：取 6*sigma 以覆盖绝大部分能量
            window_size = int(6 * sigma * fs)
            if window_size < 1:
                window_size = 1

            # 生成时间向量（以 0 为中心）
            t = np.arange(-window_size, window_size + 1) * dt

            # 生成 Morlet 小波
            wavelet = morlet_wavelet(t, f, c)

            # 卷积（计算小波变换）
            response = fftconvolve(data, wavelet, mode='same')

            # 保存响应（幅度）
            wavelet_responses.append(np.abs(response) ** 2)

        # 计算几何平均（Geometric Mean）
        # 为了数值稳定性，在对数域计算
        # GM = exp((1/order) * Σ ln(|R_i|))
        wavelet_responses = np.array(wavelet_responses)

        # 避免 log(0)
        wavelet_responses = np.maximum(wavelet_responses, 1e-12)

        # 在对数域计算平均
        log_mean = np.mean(np.log(wavelet_responses), axis=0)

        # 指数化得到几何平均
        slt_matrix[freq_idx, :] = np.exp(log_mean)

    return slt_matrix


def extract_band_features(slt_matrix, frequencies, band_range):
    """
    从 SLT 时频矩阵中提取特定频段的特征

    根据论文，提取以下特征：
    1. 平均值（mean of absolute wavelet coefficients）
    2. 能量值（sum of squares of wavelet coefficients）

    参数：
    - slt_matrix: Superlet 时频矩阵
    - frequencies: 频率数组
    - band_range: 频段范围 (min_freq, max_freq)

    返回：
    - features: 特征字典，包含 mean 和 energy
    """
    # 找到频段对应的索引
    freq_mask = (frequencies >= band_range[0]) & (frequencies <= band_range[1])
    band_data = slt_matrix[freq_mask, :]

    features = {}

    # 1. 平均值（mean of absolute wavelet coefficients）
    # 这里 SLT 矩阵已经是幅度的平方，所以开方得到幅度
    features['mean'] = np.mean(np.sqrt(band_data))

    # 2. 能量值（sum of squares of wavelet coefficients）
    # SLT 矩阵已经是平方形式，所以直接求和
    features['energy'] = np.sum(band_data)

    return features


def compute_time_frequency_power(slt_matrix):
    """
    计算时频功率（TF Power）

    公式：TF Power = (1/n) * Σ(M_ij)^2
    其中 M_ij 是 SLT 矩阵的元素

    参数：
    - slt_matrix: Superlet 时频矩阵

    返回：
    - tf_power: 时频功率值
    """
    n_samples = slt_matrix.size
    tf_power = (1 / n_samples) * np.sum(slt_matrix ** 2)
    return tf_power


# 测试代码
if __name__ == "__main__":
    # 生成测试信号
    sfreq = 250
    duration = 10  # 秒
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq

    # 生成包含多个频率成分的信号
    signal = (np.sin(2 * np.pi * 5 * t) +  # 5 Hz (Theta)
              np.sin(2 * np.pi * 10 * t) +  # 10 Hz (Alpha)
              0.5 * np.random.randn(n_samples))  # 噪声

    # 创建频率数组 (1-40 Hz)
    frequencies = np.linspace(1, 40, 40)

    # 应用 Superlet 变换
    print("计算 Superlet 变换...")
    slt_matrix = superlet_transform(signal, sfreq, frequencies, c1=6, order=6)

    print(f"SLT 矩阵形状: {slt_matrix.shape}")
    print(f"频率数组: {frequencies[:5]}...{frequencies[-5:]}")

    # 提取频段特征
    bands = {
        'Delta': (0.5, 3),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30)
    }

    print("\n频段特征提取:")
    for band_name, band_range in bands.items():
        features = extract_band_features(slt_matrix, frequencies, band_range)
        print(f"{band_name} ({band_range[0]}-{band_range[1]} Hz): "
              f"Mean={features['mean']:.4f}, Energy={features['energy']:.4f}")

    # 计算时频功率
    tf_power = compute_time_frequency_power(slt_matrix)
    print(f"\n时频功率: {tf_power:.4f}")