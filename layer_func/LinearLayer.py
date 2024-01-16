import numpy as np


class LinearLayer(object):
    """
        name：
            线性层
        param：
            in_num：输入特征数
            out_num：输出特征数
            bias：是否含偏置
            dropout：是否 dropout
            drop_prob：dropout 比例
        """
    def __init__(self, in_num, out_num, bias=True, dropout=False, drop_prob=0.5):
        super(LinearLayer, self).__init__()

        self.in_num = in_num
        self.out_num = out_num
        self.param = {'w': np.random.randn(in_num, out_num), 'b': None}
        self.grad = {'gw': np.zeros((in_num, out_num)), 'gb': None}
        self.grad_m = {'mw': np.zeros((in_num, out_num)), 'mb': None}
        self.grad_v = {'vw': np.zeros((in_num, out_num)), 'vb': None}
        self.bias = bias
        if self.bias:
            self.param['b'] = np.zeros(out_num)
            self.grad['gb'] = np.zeros(out_num)
            self.grad_m['mb'] = np.zeros(out_num)
            self.grad_v['vb'] = np.zeros(out_num)
        self.X = None

        self.dropout = dropout
        self.dropout_state = dropout
        self.drop_prob = drop_prob
        self.mask = None
        self.t = 0

    def forward(self, X):
        self.X = X
        if self.dropout:
            self.mask = np.where(np.random.rand(self.in_num, self.out_num) < self.drop_prob, 0.0, 1.0)
            return np.matmul(self.X, self.param['w'] * self.mask) + self.bias if self.bias else np.matmul(self.X, self.param['w'] * self.mask)
        else:
            return np.matmul(self.X, self.param['w']) + self.bias if self.bias else np.matmul(self.X, self.param['w'])

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])

    def backward(self, radius, reg=False, lamda=0.01):
        if reg:
            self.grad['gw'] = (np.matmul(self.X.T, radius) + lamda * 2.0 * self.param['w'] / self.X.shape[0])* self.mask if self.dropout else np.matmul(self.X.T, radius) + lamda * 2.0 * self.param['w'] / self.X.shape[0]
        else:
            self.grad['gw'] = np.matmul(self.X.T, radius) * self.mask if self.dropout else np.matmul(self.X.T, radius)
        if self.param['b'] is not None:
            self.grad['gb'] = np.mean(radius, axis=0)
        return np.matmul(radius, (self.param['w'] * self.mask).T) if self.dropout else np.matmul(radius, self.param['w'].T)

    def step_grad(self, lr=0.001, adam=False, beta_1=0.9, beta_2=0.999):
        if adam:
            delta = 1e-8
            self.t += 1
            self.grad_m['mw'] = beta_1 * self.grad_m['mw'] + (1 - beta_1) * self.grad['gw']
            self.grad_v['vw'] = beta_2 * self.grad_v['vw'] + (1 - beta_2) * (self.grad['gw']**2)
            m_tilde = self.grad_m['mw'] / (1 - beta_1 ** self.t)
            v_tilde = self.grad_v['vw'] / (1 - beta_2 ** self.t)
            self.param['w'] -= lr * m_tilde / (np.sqrt(v_tilde) + delta)
            if self.param['b'] is not None:
                self.grad_m['mb'] = beta_1 * self.grad_m['mb'] + (1 - beta_1) * self.grad['gb']
                self.grad_v['vb'] = beta_2 * self.grad_v['vb'] + (1 - beta_2) * (self.grad['gb'] ** 2)
                m_tilde = self.grad_m['mb'] / (1 - beta_1 ** self.t)
                v_tilde = self.grad_v['vb'] / (1 - beta_2 ** self.t)
                self.param['b'] -= lr * m_tilde / (np.sqrt(v_tilde) + delta)
        else:
            self.param['w'] -= lr * self.grad['gw']
            if self.param['b'] is not None:
                self.param['b'] -= lr * self.grad['gb']

    def clear_grade(self):
        self.grad['gw'] = np.zeros((self.in_num, self.out_num))
        if self.param['b'] is not None:
            self.grad['gb'] = np.zeros(self.out_num)
