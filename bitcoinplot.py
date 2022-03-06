import pandas
df = pandas.read_csv("data_bit_1y1h.csv", index_col=0)
#ライブラリ
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#matplotlib inline
from matplotlib.pylab import rcParams
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

rcParams["figure.figsize"] = 40, 10

#グラフを描画する
x = df.index.values
y = df["Close"].values
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.show()

#正規化
input_data = df["Close"].values.astype(float)
print("input_data : ", input_data.shape, type(input_data))

norm_scale = input_data.max()
input_data /= norm_scale
print(input_data[0:5])

#教師データの分割
def make_dataset(low_data, maxlen):

    data, target = [], []

    for i in range(len(low_data) - maxlen):
        data.append(low_data[i:i+maxlen])
        target.append(low_data[i+maxlen])

    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target

window_size = 12

X, y = make_dataset(input_data, window_size)
print("shape X : ", X.shape)
print("shape y : ", y.shape)

#学習データとテストデータの分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#train:学習データ、test:テストデータ

print("X_train : ", X_train.shape)
print("X_test : ", X_test.shape)
print("y_train : ", y_train.shape)
print("y_test : ", y_test.shape)

#ネットワークの構築
lstm_model = Sequential()
lstm_model.add(LSTM(400, batch_input_shape=(None, window_size, 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

#コンパイル
lstm_model.compile(loss="mean_squared_error", optimizer=Adam(), metrics = ["accuracy"])
lstm_model.summary()

#学習用パラメータ
batch_size = 30
n_epoch = 1500

#学習
hist = lstm_model.fit(X_train, y_train,
                     epochs=n_epoch,
                     validation_data=(X_test, y_test),
                     verbose=0,
                     batch_size=batch_size)

#損失値の遷移をプロット
plt.plot(hist.history["loss"], lw=1, label="train set")
plt.plot(hist.history["val_loss"], lw=1, label="test set")
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

#予想
y_pred_train = lstm_model.predict(X_train)
y_pred_test = lstm_model.predict(X_test)
#train:学習データ、test:テストデータ
#X:データ、y:答え

# RMSEで評価
# 参考：https://deepage.net/deep_learning/2016/09/17/tflearn_rnn.html
def rmse(y_pred, y_true):
    return np.sqrt(((y_true - y_pred) ** 2).mean())
print("RMSE Score")
print("  train : " , rmse(y_pred_train, y_train))
print("  test : " , rmse(y_pred_test, y_test))

# 推定結果のプロット
plt.plot(X[:, 0, 0], color='blue',  label="observed")  # 元データ
plt.plot(y_pred_train, color='red',  lw=1, label="train")   # 予測値（学習）
plt.plot(range(len(X_train),len(X_test)+len(X_train)),y_pred_test, color='green', lw=1,  label="test")   # 予測値（検証）
plt.legend()
#plt.xticks(np.arange(0, 145, 12)) # 12ヶ月ごとにグリッド線を表示
plt.grid()
plt.show()
