import numpy as np


class Relu(object):
    """
    name:
        Relu 激活函数
    """
    def __init__(self):
        super(Relu, self).__init__()
        self.X = None

    def forward(self, X):
        self.X = X
        return np.where(X < 0, 0, X)

    def backward(self, radius):
        return np.multiply(np.where(self.X < 0, 0, 1), radius)

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])
