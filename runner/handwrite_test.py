from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

from model import Classifier
from model import Classifier_conv


# Linear test
def standardizeCols(X, mu=None, sigma=None):
    if mu is None:
        mu = np.mean(X, axis=0)
    if sigma is None:
        sigma = np.std(X, axis=0, ddof=1)
    return (X - mu) / sigma, mu, sigma


image_0 = np.array(Image.open('../handwrite/handwrite_0.png').convert('L')).reshape((1, -1), order='F')
image_1 = np.array(Image.open('../handwrite/handwrite_1.png').convert('L')).reshape((1, -1), order='F')
image_2 = np.array(Image.open('../handwrite/handwrite_2.png').convert('L')).reshape((1, -1), order='F')
image_3 = np.array(Image.open('../handwrite/handwrite_3.png').convert('L')).reshape((1, -1), order='F')
image_4 = np.array(Image.open('../handwrite/handwrite_4.png').convert('L')).reshape((1, -1), order='F')
image_5 = np.array(Image.open('../handwrite/handwrite_5.png').convert('L')).reshape((1, -1), order='F')
image_6 = np.array(Image.open('../handwrite/handwrite_6.png').convert('L')).reshape((1, -1), order='F')
image_7 = np.array(Image.open('../handwrite/handwrite_7.png').convert('L')).reshape((1, -1), order='F')
image_8 = np.array(Image.open('../handwrite/handwrite_8.png').convert('L')).reshape((1, -1), order='F')
image_9 = np.array(Image.open('../handwrite/handwrite_9.png').convert('L')).reshape((1, -1), order='F')

image = np.concatenate((image_0, image_1, image_2, image_3, image_4,
                        image_5, image_6, image_7, image_8, image_9), axis=0)

# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.axis('off')
#     plt.imshow(image[i].reshape((16, 16), order='F'), cmap='gray')
# plt.show()

data = io.loadmat('../digits.mat')
X, y = data['X'], data['y']
X, mu, sigma = standardizeCols(X)

image_test, _, _ = standardizeCols(image, mu, sigma)

label = np.eye(10)

# 线性模型预测
cls = Classifier.Classifier(256, 128, 10, lr=0.01)
cls.load_param('../best_param/best_linear_model.pickle')
y_pred = cls.predict(image_test, label)
print(y_pred)


# CNN模型参数测试
image_test = image_test.reshape((-1, 16, 16), order='F')
cls = Classifier_conv.Classifier_conv(out_num=10, lr=0.01)
cls.load_param('../best_param/best_conv_model.pickle')

y_pred = cls.predict(image_test.reshape(-1, 1, 16, 16), label)
print(y_pred)
