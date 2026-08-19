from scipy.signal import stft as scipy_stft
import numpy as np


def motorTS(motor_p, motor_rpm, N_theta = 360*4, N_t = 2048, t_total = 0.1):
    """永磁电机的时空阶次分析"""
    fs = motor_rpm * motor_p /60
    mu0 = 4*np.pi*1e-7

    theta = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)
    t = np.linspace(0, t_total, N_t, endpoint=False)
    THETA, TIME = np.meshgrid(theta, t)

    # 气隙电磁力计算
    harmonics = [
        [motor_p, 0.80, 1, 0],  # 主极基波 v=p
        [5 * motor_p, 0.12, 1, np.pi / 6],
        [7 * motor_p, 0.08, 1, np.pi / 3],
        [-motor_p, 0.05, -1, 0],
        [3 * motor_p, 0.06, 3, np.pi / 2]
    ]

    B_gap = np.zeros_like(THETA)
    for v, Bamp, k, phi in harmonics:
        omega = 2 * np.pi * k * fs
        B_gap += Bamp * np.cos(v * THETA - omega * TIME + phi)

    # 3. 计算径向电磁力密度 Prad = B^2/(2μ0)
    P_rad = B_gap ** 2 / (2 * mu0)

    # 4. 二维FFT 时空谐波分解
    # FFT维度：axis0=时间维度，axis1=空间角度维度
    fft2d = np.fft.fft2(P_rad)
    fft_shift = np.fft.fftshift(fft2d)
    amp2d = np.abs(fft_shift) / (N_t * N_theta)

    # 5. 生成坐标：时间频率 & 空间阶次
    df = 1 / t_total  # 频率分辨率
    freq_axis = np.fft.fftfreq(N_t, d=t_total / N_t)
    freq_axis = np.fft.fftshift(freq_axis)

    d_order = 2 * np.pi / (2 * np.pi)  # 空间阶次分辨率（每圈谐波次数）
    sp_order_axis = np.fft.fftfreq(N_theta, d=1 / N_theta)
    sp_order_axis = np.fft.fftshift(sp_order_axis)
    return freq_axis, sp_order_axis, amp2d

def compute_stft(signal, fs, nperseg=256, noverlap=None, nfft=None,
                 window='hann', boundary='zeros', padded=True, axis=-1):
    """
    计算短时傅里叶变换 (STFT)

    参数:
        signal: 输入信号 (1D numpy 数组)
        fs: 采样频率 (Hz)
        nperseg: 每个分段的样本数 (默认 256)
        noverlap: 相邻分段的重叠样本数 (默认 nperseg//2)
        nfft: FFT 点数 (默认 nperseg)
        window: 窗函数类型 (默认 'hann')
        boundary: 边界处理方式 (默认 'zeros')
        padded: 是否填充信号 (默认 True)
        axis: 计算 STFT 的轴 (默认 -1)

    返回:
        f: 频率数组 (Hz)
        t: 时间数组 (s)
        Zxx: STFT 结果 (复数矩阵，形状为 (频率, 时间))
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg

    f, t, Zxx = scipy_stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
                           nfft=nfft, window=window, boundary=boundary,
                           padded=padded, axis=axis)
    return f, t, Zxx

def motorSTFT(signal, fs, nperseg=256, noverlap=None, nfft=None,
                           window='hann', boundary='zeros', padded=True, axis=-1):
    """
    计算 STFT 并返回幅度谱 (dB)

    参数:
        signal: 输入信号\n
        fs: 采样频率 (Hz)\n
        nperseg: 每个分段的样本数\n
        noverlap: 重叠样本数\n
        nfft: FFT 点数\n
        window: 窗函数类型\n
        boundary: 边界处理方式\n
        padded: 是否填充\n
        axis: 计算轴\n

    返回:
        f: 频率数组 (Hz)\n
        t: 时间数组 (s)\n
        magnitude_db: 幅度谱 (dB)\n
    """
    f, t, Zxx = compute_stft(signal, fs, nperseg, noverlap, nfft,
                             window, boundary, padded, axis)
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

    return f, t, magnitude_db


def motorFFT(signal, fs):
    """对时序信号进行傅里叶变换计算"""
    N = len(signal)

    # FFT 计算
    fft_result = np.fft.fft(signal)  # 复数频谱
    fft_magnitude = np.abs(fft_result)  # 幅度谱

    # 只取正频率（单边谱）
    fft_magnitude = fft_magnitude[:N // 2]

    # 频率轴
    freqs = np.fft.fftfreq(N, d=1 / fs)
    freqs = freqs[:N // 2]

    # 幅度归一化（使峰值等于实际振幅）
    fft_magnitude = fft_magnitude / (N / 2)
    return freqs, fft_magnitude