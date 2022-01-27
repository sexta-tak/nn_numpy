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
        dout = 1
        dout = net.last_layer.backward(dout)
        layers = list(net.layers.keys())
        layers.reverse()
        for layer in layers:
            dout = net.layers[layer].backward(dout)

        self._update(net, layers)

    def _update(self, net, layers):
        for layer in layers:
            if not net.layers[layer].is_params:
                continue

            if not any(net.layers[layer].params):
                self._update(net, layer)

            else:
                for param in net.layers[layer].params.keys():
                    if net.layers[layer].params[param] is not None:
                        net.layers[layer].params[param] -= self.lr * net.layers[layer].grads[param]