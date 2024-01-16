import numpy as np


class Softmax(object):
    """
    name:
        Softmax 概率归一化层
    """
    def __init__(self):
        super(Softmax, self).__init__()
        self.X = None

    def forward(self, X):
        self.X = X - np.max(X, axis=1, keepdims=True)
        return np.exp(self.X) / np.sum(np.exp(self.X), axis=1, keepdims=True)

    def backward(self, radius):
        radius = np.expand_dims(radius, axis=1)  # shape = [batch, 1, label]

        exp_sum = np.exp(self.X) / np.sum(np.exp(self.X), axis=1, keepdims=True)
        # shape = [batch, label, label]
        grad = np.apply_along_axis(np.diag, 1, exp_sum) - np.einsum('ijk, ikn -> ijn', np.expand_dims(exp_sum, axis=2), np.expand_dims(exp_sum, axis=1))
        grad = np.squeeze(np.einsum('ijk, ikn -> ijn', radius, np.transpose(grad, (0, 2, 1))), axis=1)

        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])
