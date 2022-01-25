import numpy as np

def accuracy(self, y, t, batch_size=100):
    if t.ndim != 1 : t = np.argmax(t, axis=1)
    
    acc = 0.0
    y = np.argmax(y, axis=1)
    acc += np.sum(y == t) 
    
    return acc / batch_size