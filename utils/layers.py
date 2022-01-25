import numpy as np
from collections import OrderedDict
from activation import softmax, relu
from loss import cross_entropy_error

class Module:
    def __init__(self, required_grad=True):
        self.params = OrderedDict()
        self.layers = OrderedDict()
        if required_grad:
            self.grad = OrderedDict()

class Affine(Module):
    def __init__(self, in_features, out_features, weight_init_std=0.01, bias=True):
        super().__init__()

        self.bias = bias
        W = weight_init_std * np.random.randn(in_features, out_features)
        b = np.zeros(out_features) if bias else None
        self.params = {"W":W, "b":b}
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.grads = {"W":None, "b":None}

    def forward(self, x):
        # テンソル対応(画像形式のxに対応させる)
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.params["W"]) + self.params["b"]

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.params["W"].T)
        self.grads["W"] = np.dot(self.x.T, dout)
        if self.bias:
            self.grads["b"] = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        
        # 初期値
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        """
        順伝播
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        """
        逆伝播
        伝播する値をバッチサイズで割ること
        dout=1は、他のレイヤと同じ使い方ができるように設定しているダミー変数
        """
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx