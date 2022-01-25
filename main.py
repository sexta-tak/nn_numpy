#%%
import numpy as np
from model.twonet import TwoLayerNet
from utils.evaluate import accuracy
from utils.optimizer import SGD

#%%
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

train = X_train/255
test = X_test/255
train = train.reshape(-1, 28*28)
test = test.reshape(-1, 28*28)
train_labels = lb.fit_transform(y_train)
test_labels = lb.fit_transform(y_test)

#%%
x = train
t = train_labels

# x = x.reshape(-1,1,28,28) # 配列形式の変形

epochs = 10
batch_size = 500

optimizer = SGD(lr=0.01)

# 繰り返し回数
xsize = x.shape[0]
iter_num = np.ceil(xsize / batch_size).astype(np.int)


# CNNのオブジェクト生成
net = TwoLayerNet(input_size=28*28, hidden_size=100, output_size=10)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(epochs):
    print("epoch=%s"%epoch)

    # シャッフル
    idx = np.arange(xsize)
    np.random.shuffle(idx)

    for it in range(iter_num):
        """
        ランダムなミニバッチを順番に取り出す
        """
        mask = idx[batch_size*it : batch_size*(it+1)]

        # ミニバッチの生成
        x_train = x[mask]
        t_train = t[mask]

        loss = net.loss(x, t)

        # 更新
        optimizer.update(net)

    ## 学習経過の記録

    # 訓練データにおけるloss
#     print("calculating train_loss")    
    train_loss.append(loss)
    print(loss)

#     print("calculating test_loss")
    # テストデータにおけるloss
    test_loss.append(net.loss(test, test_labels))

#     print("calculating train_accuracy")
    # 訓練データにて精度を確認
    acc = net.accuracy(x, t)
    train_accuracy.append(acc)
    print(acc)
    
#     print("calculating test_accuracy")
    # テストデータにて精度を算出
    test_accuracy.append(net.accuracy(test, test_labels))

#%%
import pandas as pd
import matplotlib.pyplot as plt

# lossとaccuracyのグラフ化
df_log = pd.DataFrame({"train_loss":train_loss,
             "test_loss":test_loss,
             "train_accuracy":train_accuracy,
             "test_accuracy":test_accuracy})

df_log.plot(style=['r-', 'r--', 'b-', 'b--'])
plt.ylim([0,3])
plt.ylabel("Accuracy or loss")
plt.xlabel("epochs")
plt.show()