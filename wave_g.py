import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import wave
import pylab as pl




def createWave(sample_rate=22050, fa=100, t_length=10, mode="square25"):
    T = sample_rate / fa  # 周期
    D_Omega = 2 * pi / T  # 数字角频率
    D_num = int(t_length * sample_rate)  # 对应的数字波形点数
    x = np.arange(0, D_num)
    y = np.ones(D_num)
    # 基波
    if mode == "square125":  # 方波占空比12.5%
        high = T // 8
        y[x % T > high] = -1
    elif mode == "square25":  # 方波占空比25%
        high = T // 4
        y[x % T > high] = -1
    elif mode == "square50":  # 方波占空比50%
        y = np.sin(D_Omega * x)
        y[y > 0] = 1
        y[y < 0] = -1
    elif mode == "square75":  # 方波占空比75%
        high = 3 * T // 4
        y[x % T > high] = -1
    elif mode == "triangle":  # 三角波
        for i in range(len(x)):
            y[i] = abs(1 - i % T / T * 2)
        y = y - 1  # 向下平移
    elif mode == "noise":  # 噪声
        y = np.random.normal(0, 1, D_num)  ## 我也不知道应该是什么分布的噪声，先用这个吧
    t = x / sample_rate
    return y, t


def saveWave(y, sample_rate, path=r'wave.wav'):
    file = wave.open(path, 'wb')
    file.setnchannels(1)  # 设置通道数
    file.setsampwidth(2)  # 设置采样宽
    file.setframerate(sample_rate)  # 设置采样
    file.setcomptype('NONE', 'not compressed')  # 设置采样格式  无压缩
    y = y * 32768
    y_data = y.astype(np.int16).tobytes()  # 将类型转为字节
    file.writeframes(y_data)
    file.close()

def reafWave():
    f = wave.open(r"wave.wav", "rb")
    # 读取格式信息
    # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采
    # 样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # 读取波形数据
    # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
    str_data = f.readframes(nframes)
    f.close()
    # 将波形数据转换成数组
    # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
    wave_data = np.fromstring(str_data, dtype=np.short)
    # 通过取样点数和取样频率计算出每个取样的时间。
    time = np.arange(0, nframes) / framerate

    fft_size = 512  # FFT处理的取样长度
    # N点FFT进行精确频谱分析的要求是N个取样点包含整数个取样对象的波形。因此N点FFT能够完美计算频谱对取样对象的要求是n*Fs/N（n*采样频率/FFT长度），
    # 因此对8KHZ和512点而言，完美采样对象的周期最小要求是8000/512=15.625HZ,所以15.625的n为10。
    xs = wave_data[:fft_size]  # 从波形数据中取样fft_size个点进行运算
    xf = np.fft.rfft(xs) / fft_size  # 利用np.fft.rfft()进行FFT计算，rfft()是为了更方便对实数信号进行变换，由公式可知/fft_size为了正确显示波形能量
    # rfft函数的返回值是N/2+1个复数，分别表示从0(Hz)到sampling_rate/2(Hz)的分。
    # 于是可以通过下面的np.linspace计算出返回值中每个下标对应的真正的频率：
    freqs = np.linspace(0, framerate / 2, fft_size / 2 + 1)
    # np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    # 在指定的间隔内返回均匀间隔的数字
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    # 最后我们计算每个频率分量的幅值，并通过 20*np.log10()将其转换为以db单位的值。为了防止0幅值的成分造成log10无法计算，我们调用np.clip对xf的幅值进行上下限处理

    # 绘图显示结果
    pl.figure(figsize=(8, 4))

    pl.subplot(311)
    pl.plot(time, wave_data)
    pl.xlabel("time(s)")
    pl.title(u"WaveForm And Freq")
    pl.subplot(312)
    pl.plot(time[:fft_size], xs)
    pl.xlabel(u"Time(S)")

    pl.subplot(313)
    pl.plot(freqs, xfp)
    pl.xlabel(u"Freq(Hz)")
    pl.subplots_adjust(hspace=0.4)
    pl.show()
# if __name__ == "__main__":
#     mode = ["square125", "square25", "square50", "square75", "triangle", "noise"]
#     for i in range(6):
#         y, t = createWave(sample_rate=22050, fa=100, t_length=0.05, mode=mode[i])
#         plt.subplot(2, 3, i + 1)
#         plt.plot(t, y)  # 画出一个周期的波形图
#     plt.show()
    # fig, ax_nstd = plt.subplots()
    # y1,t1 = createWave(sample_rate=22050, fa=100, t_length=0.05, mode='square125')
    # plt.plot(t1, y1)
    # y,t = createWave(sample_rate=22050, fa=100, t_length=0.05, mode='noise')
    # plt.plot(t, y)
    # plt.show()
# reafWave()