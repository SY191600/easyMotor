import easyMotor

if __name__ == "__main__":
    motorTS = easyMotor.MotorToothHarmonicCalculator()
    motorTS.speed_rpm = 750
    motorTS.p = 4
    motorTS.Q1 = 84

    df_order, df_freq, df_amp, muu, vv = motorTS.generate_harmonic_table()

    # 绘图以table的形式显示df_order, df_freq, df_amp
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    plt.figure(figsize=(12, 8))
    plt.title("力波阶数 n")
    plt.axis('off')
    plt.table(cellText=df_order.values, rowLabels = muu, colLabels=vv,  loc='center')
    plt.figure(figsize=(12, 8))
    plt.title("频率 (Hz)")
    plt.axis('off')
    plt.table(cellText=df_freq.values, rowLabels = muu, colLabels=vv, loc='center')
    plt.figure(figsize=(12, 8))
    plt.title("幅值 (相对值)")
    plt.axis('off')
    plt.table(cellText=df_amp.values, rowLabels = muu, colLabels=vv, loc='center')
    plt.tight_layout()
    plt.show()
