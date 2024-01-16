import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from model import Classifier_conv
from layer_func.gauss_noise import gauss_noise


# One-Hot Label
def linearInd2Binary(y, nLabels):
    return np.eye(nLabels)[y]


# BatchNorm
def standardizeCols(X, mu=None, sigma=None):
    if mu is None:
        mu = np.mean(X, axis=0)
    if sigma is None:
        sigma = np.std(X, axis=0, ddof=1)
    return (X - mu) / sigma, mu, sigma


# Load data
data = io.loadmat('../digits.mat')
X, y = data['X'], data['y']
Xvalid, yvalid = data['Xvalid'], data['yvalid']
Xtest, ytest = data['Xtest'], data['ytest']
# 将 0 标签由 10 改为 0
y = np.where(y == 10, 0, y).reshape(-1)
yvalid = np.where(yvalid == 10, 0, yvalid).reshape(-1)
ytest = np.where(ytest == 10, 0, ytest).reshape(-1)

ytest = linearInd2Binary(ytest, 10)
yvalid = linearInd2Binary(yvalid, 10)

# # print(X[0, :].reshape(16, 16, order='F'))  # order = 'C' 按行排布 'F' 按列排布
# for i in range(18):
#     plt.subplot(3, 6, i + 1)
#     plt.imshow(X[idx[0][i], :].reshape(16, 16, order='F'), cmap='gray')
#     plt.title(str(y[idx[0][i], 0]))
# plt.show()

# Data preparation
n, d = X.shape
nLabels = np.max(y) + 1
yExpanded = linearInd2Binary(y, nLabels)
t = Xvalid.shape[0]
t2 = Xtest.shape[0]

X, mu, sigma = standardizeCols(X)

Xvalid, _, _ = standardizeCols(Xvalid, mu, sigma)
Xtest, _, _ = standardizeCols(Xtest, mu, sigma)

# # 引入高斯噪声丰富训练集
# X_noise, y_noise = gauss_noise(X, yExpanded, sigma=0.2)
# # plt.imshow(X_noise[0].reshape(16, 16, order='F'), cmap='gray')
# # plt.title('label : ' + str(np.argmax(y_noise, axis=1)[0]))
# # plt.show()
# X = np.concatenate((X, X_noise), axis=0)
# print(X.shape)
# yExpanded = np.concatenate((yExpanded, y_noise), axis=0)


X = X.reshape((-1, 16, 16), order='F')
Xvalid = Xvalid.reshape((-1, 16, 16), order='F')
Xtest = Xtest.reshape((-1, 16, 16), order='F')

cls = Classifier_conv.Classifier_conv(out_num=10, lr=0.01)
cls.fit(X.reshape(-1, 1, 16, 16), yExpanded)
cls.train(25, 5, if_eval=True, X_eval=Xvalid.reshape(-1, 1, 16, 16), y_eval=yvalid, save_path='../best_param/1_2_6.pickle')
# cls = Classifier_conv.Classifier_conv(10, in_chanel=1, out_chanel=20, stride=2, padding=False, lr=0.001)
# cls.fit(np.concatenate((X, Xvalid), axis=0).reshape((-1, 1, 16, 16)), np.concatenate((yExpanded, yvalid), axis=0))
# cls.load_param()
# cls.train(25, 20, if_eval=False)

cls.load_param('../best_param/1_2_6.pickle')

# Evaluate test error
y_pred, acc = cls.predict(Xtest.reshape(-1, 1, 16, 16), ytest)
# for i, idx in enumerate(np.where(np.argmax(ytest, axis=1) != y_pred)[0][:35]):
#     plt.subplot(5, 7, i + 1)
#     plt.imshow(Xtest[idx, :].reshape(16, 16, order='F'), cmap='gray')
#     plt.title('pred : ' + str(y_pred[idx]) + ' label : ' + str(np.argmax(ytest, axis=1)[idx]))
# plt.show()
#
# 特征提取
features = cls.show_features(Xtest.reshape(-1, 1, 16, 16)[:10, :, :, :])
print(features.shape)

plt.subplot(5, 5, 1)
plt.axis('off')
plt.imshow(Xtest[5], cmap='gray')
for i in range(20):
    plt.subplot(5, 5, i + 6)
    plt.axis('off')
    plt.imshow(features[5, i, :, :], cmap='gray')
plt.show()

# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, label='train_loss')
# plt.scatter(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, s=5)
# plt.plot(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, label='eval_loss')
# plt.scatter(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, s=5)
# plt.legend()
# plt.title('Loss')
# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, label='train_acc')
# plt.scatter(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, s=5)
# plt.plot(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, label='eval_acc')
# plt.scatter(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, s=5)
# plt.legend()
# plt.title('Accuracy')
# plt.suptitle('Conv acc = {}'.format(acc))
# plt.show()
