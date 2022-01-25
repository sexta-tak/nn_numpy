class SGD:
    """
    Stochastic Gradient Descent
    """
    def __init__(self, lr=0.01):
        """
        lr : 学習係数 learning rate
        """
        self.lr = lr
        
    def update(self, net):
        """
        重みの更新
        """
        dout = net.last_layer.backward()
        keys = list(net.layers.keys())
        keys.reverse()
        for key in keys:
            dout = net.layers[key].backward(dout)

        for ley in keys:
            for param in net.layers[key].params.keys():
                net.layers[key].params[param] -= self.lr * net.layers[key].grads[param]