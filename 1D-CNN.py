import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame, Series
from keras import models, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# 一维卷积神经网络［通常与空洞卷积核（dilated kernel）一起使用］已经在音 频生成和机器翻译领域取得了巨大成功。另外，对于文本分类和时间序列预测等简单任务，小型的一维卷积神经网络可以替代RNN，而且速度更快
# 序列的一维卷积：可以识别序列中的局部模式。因为对每个序列段执行相同的输入变换，所以在句子中某个位置学到的模式稍后可以在其他位置被识别，这使得一维卷积神经网络具有平移不变性（对于时间平移而言）
# 一维卷积神经网络的工作原理：每个输出时间步都是利用输入序列在时间维度上的一小段得到的
# 序列的一维池化：从输入中提取一维序列段（即子序列），然后输出其最大值（最大池化）或平均值（平均池化）。与二维卷积神经网络一样，该运算也是 用于降低一维输入的长度（子采样）
# 实现一维卷积神经网络
# Keras 中的一维卷积神经网络是 Conv1D 层，其接口类似于 Conv2D
from keras.datasets import imdb
from keras.preprocessing import sequence
fname = 'jena_climate_2009_2016.csv'  # jena天气数据集（2009—2016 年的数据，每 10 分钟记录 14 个不同的量）
# 对于.csv数据格式等用pandas操作更加方便

df = pd.read_csv(fname)
# 数据类型不同，所以需要标准化
# 预处理数据的方法是，将每个时间序列减去其平均值，然后除以其标准差
df = df.drop(['Date Time'], axis=1)
float_data = df.iloc[:, :]
# print(train_data)
mean = float_data.mean(axis=0)
# print(mean)
float_data -= mean
std = float_data.std(axis=0)
float_data /= std


print(float_data)

# 生成时间序列样本及其目标的生成器
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
        # [0--min_index--lookback--max_index--delay--len(data)]
        #                       i
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)#lookback?
        else:
            if i + batch_size >= max_index:  # 表明取到最后一批（数量<batch_size）
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))  # 按小时批量抽取数据点，每个点包含14个特征
        # print(samples)
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)  # 6步（每小时）一个点索引
            samples[j] = data.iloc[indices, :]
            t = data.iloc[rows[j] + delay, :]
            targets[j] = t[1]  # 144步（24小时后的温度数组）
        yield samples, targets


# 准备训练生成器、验证生成器和测试生成器
# 因为这种方法允许操作更长 的序列，所以我们可以查看更早的数据（通过增大数据生成器的 lookback 参数）或查看分辨 率更高的时间序列（通过减小生成
# 器的 step 参数）。这里我们任意地将 step 减半，得到时间序列的长度变为之前的两倍，温度数据的采样频率变为每 30 分钟一个数据点。
lookback = 720
step = 3  # 每30分钟观测一次
delay = 144

train_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000,
    shuffle=True,
    step=step)
#生成的数据集到底在何处使用？？？

val_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=200001,
    max_index=300000,
    step=step)
test_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=300001,
    max_index=None,
    step=step)

val_steps = (300000 - 200001 - lookback) // 128
print(val_steps)
test_steps = (len(float_data) - 300001 - lookback) // 128


def acc_loss_plot(history):
    fig = plt.figure()
    #ax1 = fig.add_subplot(2, 1, 1)
    #acc = history.history['acc']
    #val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    #ax1.plot(epochs, acc, 'bo', label='Training acc')
    #ax1.plot(epochs, val_acc, 'b', label='Validation acc')
    #ax1.set_title('Training and validation accuracy')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(epochs, loss, 'bo', label='Training loss')
    ax2.plot(epochs, val_loss, 'b', label='Validation loss')
    ax2.set_title('Training and validation loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


###---------------------------------------------------------------------
from keras.optimizers import RMSprop


def build_1D_cnn_jena():
    model = Sequential()
    model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit(train_gen,steps_per_epoch=500, epochs=20,validation_data=val_gen,validation_steps=val_steps)
    return history


acc_loss_plot(build_1D_cnn_jena())

