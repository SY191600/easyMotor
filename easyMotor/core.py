from scipy.signal import stft as scipy_stft
import numpy as np
from scipy import signal
import numpy as np
import pandas as pd
from fractions import Fraction


class MotorToothHarmonicCalculator:
    """同步电机齿谐波计算器"""
    
    def __init__(self, speed_rpm=750, pole_pairs=4, stator_slots=84,
                  max_gcd=3.5, d=2):
        """
        初始化电机参数
        
        Args:
            speed_rpm: 转速 (rpm)
            pole_pairs: 极对数 p
            stator_slots: 定子槽数 Q1
            max_gcd: 最大公因子 (用于分数槽绕组)
            d: 参数d
        """
        self.speed_rpm = speed_rpm
        self.p = pole_pairs
        self.Q1 = stator_slots
        self.f = pole_pairs * speed_rpm /60 
        f = Fraction(self.Q1/2/3/self.p)
        if f.denominator == 0:
            self.d = 1
        else:
            self.d = f.denominator
        print(f"分数槽绕组的分母 d = {self.d}")
        
        # 派生参数
        self.pole_number = 2 * pole_pairs  # 极数
        self.sync_speed = 60 * self.f / pole_pairs  # 同步转速
        self.base_freq = 2 * self.f  # 基础频率 = 100 Hz
        
        # k1定子齿谐波取值范围
        self.k1_values = [0, -1, 1, -2, 2, -3, 3, -4, 4, 
                          -5, 5, -6, 6, -7, 7, -8, 8, -9, 9]
    
    def calc_mu(self, r):
        """
        计算μ值 (与r相关的参数)
        
        公式: μ = 4 + 8*r
        
        Args:
            r: 行序号 (0, 1, 2, ...)
        
        Returns:
            μ值
        """
        return self.p * ( 2*r + 1)  # μ = 4 + 8*r

    def calc_v(self, k1):
        """
        计算v值 (与k1相关的参数)
        
        公式: v = 12 * k1
        
        Args:
            k1: 定子齿谐波序号
        
        Returns:
            v值
        """
        return self.p * (6 * k1/self.d + 1)  # 6k1p = 12k1 (因为p=2)
    
    def calc_force_wave_order(self, mu, k1, v):
        """
        计算力波阶数 n
        
        公式推导:
        - k1 = 0: n = 8r
        - k1 < 0: n = 8r + 8 - 12*|k1| = 8r + 8 + 12*k1
        - k1 > 0: n = 8r - 12*k1
        
        Args:
            r: 行序号
            k1: 定子齿谐波序号
        
        Returns:
            力波阶数 n
        """
        if k1 >= 0:
            return mu - v
        else:
            return mu + v
    
    def calc_frequency(self, r, k1):
        """
        计算齿谐波频率
        
        公式: f = 100 * (r + I(k1 < 0))  (Hz)
        其中 I(k1 < 0) 是指示函数，k1<0时为1，否则为0
        
        Args:
            r: 行序号
            k1: 定子齿谐波序号
        
        Returns:
            频率 (Hz)
        """
        return self.base_freq * (r + (1 if k1 < 0 else 0))
    
    def calc_amplitude(self, n, method='inverse_square'):
        """
        计算力波幅值 (相对值)
        
        电机学理论中，径向电磁力幅值与力波阶数有关:
        - 方法 'inverse': 幅值 ∝ 1/|n|
        - 方法 'inverse_square': 幅值 ∝ 1/n² (推荐)
        - 方法 'constant': 幅值 = 1 (仅用于测试)
        
        Args:
            n: 力波阶数
            method: 幅值计算方法
        
        Returns:
            相对幅值
        """
        if n == 0:
            return 1.0  # n=0时幅值最大
        
        if method == 'inverse':
            return 1.0 / abs(n)
        elif method == 'inverse_square':
            return 1.0 / (n ** 2)
        elif method == 'constant':
            return 1.0
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def generate_harmonic_table(self, max_r=28, amplitude_method='inverse_square'):
        """
        生成完整的齿谐波表格
        
        Args:
            max_r: 最大行号
            amplitude_method: 幅值计算方法
        
        Returns:
            三个DataFrame: 力波阶数表、频率表、幅值表
        """
        # 力波阶数表 (对应原"齿谐波"sheet)
        order_data = []
        # 频率表 (对应原"齿谐波的频率"sheet)
        freq_data = []
        # 幅值表
        amp_data = []
        vv = []
        muu = []
        for r in range(max_r + 1):
            mu = self.calc_mu(r)
            muu.append(mu)

            order_row = []
            freq_row = []
            amp_row = []
            
            for k1 in self.k1_values:
                
                v = self.calc_v(k1)  # v值与k1无关，这里取k1=0计算
                n = self.calc_force_wave_order(mu, k1, v)
                f = self.calc_frequency(r, k1)
                amp = self.calc_amplitude(n, method=amplitude_method)
                if r == 0:
                    vv.append(v)
                order_row.append(n)
                freq_row.append(f)
                amp_row.append(round(amp, 6))
            
            order_data.append(order_row)
            freq_data.append(freq_row)
            amp_data.append(amp_row)
        
        df_order = pd.DataFrame(order_data)
        df_freq = pd.DataFrame(freq_data)
        df_amp = pd.DataFrame(amp_data)
        
        return df_order, df_freq, df_amp, muu, vv
    
    def get_harmonic_info(self, r, k1, v):
        """
        获取指定(r, k1)的完整谐波信息
        
        Args:
            r: 行序号
            k1: 定子齿谐波序号
        
        Returns:
            dict: 包含n, f, amp的字典
        """
        n = self.calc_force_wave_order(r, k1, v)
        f = self.calc_frequency(r, k1)
        amp = self.calc_amplitude(n)
        
        return {
            'r': r,
            'k1': k1,
            'mu': self.calc_mu(r),
            'force_wave_order': n,
            'frequency_hz': f,
            'amplitude_relative': round(amp, 6)
        }
    
    def find_dominant_harmonics(self, max_r=10, top_n=10):
        """
        找出幅值最大的前top_n个谐波分量
        
        Args:
            max_r: 搜索的最大行号
            top_n: 返回前n个
        
        Returns:
            DataFrame: 按幅值排序的谐波列表
        """
        harmonics = []
        for r in range(max_r + 1):
            for k1 in self.k1_values:
                v = self.calc_v(k1)
                info = self.get_harmonic_info(r, k1, v)
                harmonics.append(info)
        
        df = pd.DataFrame(harmonics)
        df = df.sort_values('amplitude_relative', ascending=False)
        return df.head(top_n)


def motorSTFT(x, fs):
    """
    x : 时域信号
    fs : 采样频率
    """

    # 采样频率
    # dt = np.mean(np.diff(t))
    # fs = 1.0 / dt

    # print(f"采样频率: {fs:.2f} Hz")
    # print(f"采样点数: {len(x)}")
    # print(f"信号时长: {t[-1] - t[0]:.4f} s")

    # 去除直流分量
    x = x - np.mean(x)

    # STFT
    f, tt, Zxx = signal.stft(
        x,
        fs=fs,
        window='hann',
        nperseg=1024,
        noverlap=768,
        scaling='psd'
    )

    # 转换成 dB
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    return f, tt, magnitude_db


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