import numpy as np


class Conv2d(object):
    """
    name：
        多通道2D卷积层
    param：
        kernel_size：卷积核大小
        stride：卷积核步长
        in_chanel：输入通道数
        out_chanel：输出通道数
        padding：是否 padding
        kernel：自定义初始化卷积核
    """
    def __init__(self, kernel_size=3, stride=1, in_chanel=1, out_chanel=3, padding=True, kernel=None):
        super(Conv2d, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = np.random.randn(out_chanel, in_chanel, kernel_size, kernel_size) if kernel is None else kernel
        self.stride = stride
        self.padding = padding
        self.in_chanel = in_chanel
        self.out_chanel = out_chanel
        self.X = None
        self.kernel_grade = np.zeros((out_chanel, in_chanel, kernel_size, kernel_size), dtype=np.float64)

        self.t = 0
        self.kernel_grade_m = np.zeros((out_chanel, in_chanel, kernel_size, kernel_size), dtype=np.float64)
        self.kernel_grade_v = np.zeros((out_chanel, in_chanel, kernel_size, kernel_size), dtype=np.float64)

    def forward(self, X):
        self.X = X if len(X.shape) == 4 else np.expand_dims(X, axis=0)  # self.X.shape = [batch, C, H, W]
        N, C, H, W, = self.X.shape
        assert C == self.in_chanel  # 判断输入通道数是否一致
        pad_width = (self.kernel_size - 1) // 2 if self.padding else 0
        H_new = (H + 2 * pad_width - self.kernel_size) // self.stride + 1
        W_new = (W + 2 * pad_width - self.kernel_size) // self.stride + 1
        pad_X = np.zeros((N, self.in_chanel, H + self.kernel_size - 1, W + self.kernel_size - 1)) if self.padding else np.zeros_like(self.X, dtype=np.float64)
        pad_X[:, :,  pad_width: pad_width + H, pad_width: pad_width + W] = self.X

        pad_X = np.expand_dims(pad_X, axis=1)

        cov_X = np.zeros((N, self.out_chanel, H_new, W_new))

        for i in range(H_new):
            for j in range(W_new):
                cov_X[:, :, i, j] = np.sum(np.expand_dims(self.kernel, axis=0) * pad_X[:, :, :, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size], axis=(2, 3, 4))

        return cov_X

    def backward(self, radius):
        pad_width = (self.kernel_size - 1) // 2 if self.padding else 0
        N, C, H, W = radius.shape

        N_X, C_X, H_X, W_X = self.X.shape

        pad_r = np.zeros((N, C, H + self.kernel_size - 1, W + self.kernel_size - 1)) if self.padding else np.zeros_like(radius, dtype=np.float64)
        pad_r[:, :, pad_width: pad_width + H, pad_width: pad_width + W] = radius
        grad_X = np.zeros((N_X, C_X, H_X + self.kernel_size - 1, W_X + self.kernel_size - 1)) if self.padding else np.zeros_like(self.X, dtype=np.float64)

        pad_X = np.zeros((N_X, C_X, H_X + self.kernel_size - 1, W_X + self.kernel_size - 1)) if self.padding else np.zeros_like(self.X, dtype=np.float64)
        pad_X[:, :, pad_width: pad_width + H_X, pad_width: pad_width + W_X] = self.X

        for i in range(radius.shape[2]):
            for j in range(radius.shape[3]):
                # print((np.expand_dims(self.kernel, axis=0)).shape)
                # print((pad_r[:, :, pad_width + i, pad_width + j].reshape(-1, self.out_chanel, 1, 1, 1)).shape)
                # print((np.sum(pad_r[:, :, pad_width + i, pad_width + j].reshape(-1, self.out_chanel, 1, 1, 1) * np.expand_dims(self.kernel, axis=0), axis=1)).shape)
                # print((grad_X[:, :, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size]).shape)
                grad_X[:, :, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size] += np.sum(pad_r[:, :, pad_width + i, pad_width + j].reshape(-1, self.out_chanel, 1, 1, 1) * np.expand_dims(self.kernel, axis=0), axis=1)
                self.kernel_grade += np.sum(np.expand_dims(pad_X[:, :, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size], axis=1) * pad_r[:, :, pad_width + i, pad_width + j].reshape(-1, self.out_chanel, 1, 1, 1), axis=0)

        grad_X = grad_X[:, :, pad_width: pad_width + H_X, pad_width: pad_width + W_X] if self.padding else grad_X

        return grad_X

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])

    def step_grad(self, lr=0.001, adam=False, beta_1=0.9, beta_2=0.999):
        if adam:
            delta = 1e-8
            self.t += 1
            self.kernel_grade_m = beta_1 * self.kernel_grade_m + (1 - beta_1) * self.kernel_grade
            self.kernel_grade_v = beta_2 * self.kernel_grade_v + (1 - beta_2) * (self.kernel_grade**2)
            m_tilde = self.kernel_grade_m / (1 - beta_1 ** self.t)
            v_tilde = self.kernel_grade_v / (1 - beta_2 ** self.t)
            self.kernel -= lr * m_tilde / (np.sqrt(v_tilde) + delta)
        else:
            self.kernel -= self.kernel_grade * lr

    def clear_grade(self):
        self.kernel_grade = np.zeros_like(self.kernel)


# kernel = np.random.randn(10, 3, 3, 3)  # (O, I, H, W)
# arr = np.random.randn(3, 3, 5, 5)  # (B, I, H, W)
# conv = Conv2d(in_chanel=3, out_chanel=10, padding=False, kernel=kernel, stride=1)
# brr = conv(arr)
# print(brr.shape)
# grad_X = conv.backward(np.random.randn(3, 10, 3, 3))  # (B, O, H', W')
# print('gradient of Input\' shape :', grad_X.shape)
# print('gradient of kernel\' shape :', conv.kernel_grade.shape)
