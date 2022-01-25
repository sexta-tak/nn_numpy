import numpy as np
from collections import OrderedDict
from ..utils.layers import Module, Affine, ReLU, SoftmaxWithLoss

class TwoLayerNet(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        affine1 = Affine(input_size, hidden_size)
        relu1 = ReLU()
        affine2 = Affine(hidden_size, output_size)
        relu2 = ReLU()
        self.layers = {
            "affine1":affine1, 
            "relu1":relu1, 
            "affine2":affine2, 
            "relu2":relu2}

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]