import numpy as np


class AvgPool(object):
    """
    name：
        均值池化层
    param：
        kernel_size：卷积核大小
        stride：卷积核步长
    """
    def __init__(self, kernel_size=2, stride=2):
        super(AvgPool, self).__init__()
        self.shape = None
        self.size = None
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        self.shape = X.shape
        self.size = X.shape[2] // self.stride
        output = np.zeros((X.shape[0], X.shape[1], self.size, self.size))

        kernel = np.ones((1, self.shape[1], self.kernel_size, self.kernel_size)) / (self.kernel_size ** 2)
        for i in range(self.size):
            for j in range(self.size):
                output[:, :, i, j] = np.sum(X[:, :, i * self.stride: (i + 1) * self.stride, j * self.stride: (j + 1) * self.stride] * kernel, axis=(2, 3))

        return output

    def backward(self, radius):
        kernel = np.ones((1, self.shape[1], self.kernel_size, self.kernel_size)) / (self.kernel_size ** 2)
        grad = np.zeros(self.shape)
        for i in range(self.size):
            for j in range(self.size):
                grad[:, :, i * self.stride: (i + 1) * self.stride, j * self.stride: (j + 1) * self.stride] = radius[:, :, i: i + 1, j: j + 1] * kernel

        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])
