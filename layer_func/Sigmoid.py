import numpy as np


class Sigmoid(object):
    """
        name:
            Sigmoid 激活函数
        """
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.output = None

    def forward(self, X):
        self.output = 1.0 / (1.0 + np.exp(-X))
        return self.output

    def backward(self, radius):
        return np.multiply(self.output * (1.0 - self.output), radius)

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])
