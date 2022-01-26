import numpy as np
from ..utils.layers import Affine, Convolution, ReLU, MaxPooling, SoftmaxWithLoss, Module

class SimpleConvNet(Module):
    def __init__(self, input_ch, num_classes):
        super().__init__()
        conv1 = Convolution(input_ch=input_ch, output_ch=3, filter_size=3, stride=1, padding=1) #3*28*28
        relu1 = ReLU()
        pool1 = MaxPooling(filter_size=2, stride=2, padding=0) #3*14*14
        affine1 = Affine(in_features=3*14*14, out_features=100)
        relu2 = ReLU()
        affine2 = Affine(in_features=100, out_features=num_classes)

        self.layers = {
            "conv1":conv1, 
            "relu1":relu1, 
            "pool1":pool1, 
            "affine1":affine1, 
            "relu2":relu2, 
            "affine2":affine2
            }

        self.__init_weight(self.layers)

        self.last_layer = SoftmaxWithLoss()

    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    def loss(self, x, t):
        y = self.forward(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=500):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.forward(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def __init_weight(self, layers):
        for layer in layers.values():
            if not layer.is_params:
                continue

            if not hasattr(layer, "params"):
                self.__init_weight(layer)

            else:
                scale = np.sqrt(2.0 / layer.params["W"].size)
                layer.params["W"] = scale * np.random.standard_normal(layer.params["W"].shape)