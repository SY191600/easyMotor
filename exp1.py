from easyMotor import *
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
# --------------------------
# 1. 高斯白噪声
# --------------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
def white_noise(N, amp=1.0):
    return amp * np.random.randn(N)

def genData(T_total, fs):
    # ========== 参数设置 ==========
    t = np.linspace(0, T_total, int(fs * T_total), endpoint=False)
    N = len(t)
    # ========== 生成信号 ==========
    s_white = white_noise(N, amp=0.3)

    s_white = 0.3*np.sin(2*np.pi*100*t)
    return s_white
# --------------------------
# 2. 测试程序
# --------------------------
def test_motorFFT(data, fs):
    """测试FFT程序"""
    f, a = motorFFT(data, fs)
    plt.figure(figsize=(8, 4))
    plt.semilogx(f, a)
    plt.title(u"FFT频谱图")
    plt.show()

def test_motorSTFT(data, fs):
    """测试STFT程序"""
    f, t, magnitude_db = motorSTFT(data,fs)

    plt.figure(figsize=(8, 4))
    plt.contourf(t, f, np.abs(magnitude_db))
    plt.title(u"FFT频谱图")
    plt.show()

def test_motorTS():
    """测试电机的时空阶次计算程序"""
    p = 4  # 极对数
    rpm = 1500  # 转速 r/min
    freq_axis, sp_order_axis, amp2d = motorTS(p, rpm)

    fig, ax = plt.subplots(figsize=(10, 7))
    contour = ax.contourf(sp_order_axis, freq_axis, amp2d, levels=100, cmap=cm.jet)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("电磁力密度幅值 [Pa]")

    ax.set_xlabel("空间阶次 r（周向谐波次数）")
    ax.set_ylabel("频率 f [Hz]")
    ax.set_title("永磁电机电磁力时空阶次二维谱")
    ax.set_xlim([-40, 40])  # 根据电机阶次范围缩放
    ax.set_ylim([-800, 800])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    T_total = 2
    fs = 10000
    white = genData(T_total, fs)
    # 测试程序
    test_motorFFT(white, fs)
    test_motorSTFT(white, fs)
    test_motorTS()