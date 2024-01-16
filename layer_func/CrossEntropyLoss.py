import numpy as np


class CrossEntropyLoss(object):
    """
        name：
            交叉熵损失函数层
        """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.X = None
        self.y = None
        self.N = None
        self.loss = None

    def forward(self, X, y):
        self.X = X
        self.y = y
        self.N = X.shape[0]
        epsilon = 1e-6
        loss = np.sum(-y * np.log(np.where(X < epsilon, epsilon, X))) / self.N
        self.loss = loss
        return loss

    def backward(self):
        epsilon = 1e-6
        return -(self.y / (self.X + epsilon)) / self.N

    def __call__(self, *args, **kwargs):
        return self.forward(args[0], args[1])
