import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from model import Classifier
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

# Data preparation
n, d = X.shape
nLabels = np.max(y) + 1
yExpanded = linearInd2Binary(y, nLabels)
t = Xvalid.shape[0]
t2 = Xtest.shape[0]

X, mu, sigma = standardizeCols(X)
Xvalid, _, _ = standardizeCols(Xvalid, mu, sigma)
Xtest, _, _ = standardizeCols(Xtest, mu, sigma)

# 引入高斯噪声丰富训练集
X_noise, y_noise = gauss_noise(X, yExpanded, sigma=0.4)
X = np.concatenate((X, X_noise), axis=0)
yExpanded = np.concatenate((yExpanded, y_noise), axis=0)
X, mu, sigma = standardizeCols(X)

Xvalid_noise, yvalid_noise = gauss_noise(Xvalid, yvalid, sigma=0.3)
Xvalid = np.concatenate((Xvalid, Xvalid_noise), axis=0)
yvalid = np.concatenate((yvalid, yvalid_noise), axis=0)
Xvalid, _, _ = standardizeCols(Xvalid, mu, sigma)

idx = np.random.permutation(10000)
Xvalid = Xvalid[idx]
yvalid = yvalid[idx]

# # 加入部分验证集进行训练（交叉验证）最终模型未选用
# for num in range(3):
#     # idx = np.random.permutation(5000)
#     # Xvalid = Xvalid[idx]
#     # yvalid = yvalid[idx]
#     for i in range(5):
#         cls.load_param()
#         cls.fit(np.concatenate((X, Xvalid[: i * 1000], Xvalid[(i + 1) * 1000:]), axis=0), np.concatenate((yExpanded, yvalid[: i * 1000], yvalid[(i + 1) * 1000:]), axis=0))
#         best_loss = cls.train(50, 3, if_eval=True, X_eval=Xvalid[i * 1000: (i + 1) * 1000], y_eval=yvalid[i * 1000: (i + 1) * 1000], fine_tune=False, best_loss=best_loss)

# 模型训练
# cls = Classifier.Classifier(256, 128, 10, lr=0.01)
# cls.fit(np.concatenate((X, Xvalid[: 8000]), axis=0), np.concatenate((yExpanded, yvalid[: 8000]), axis=0))
# best_loss = cls.train(50, 20, if_eval=True, X_eval=Xvalid[8000:], y_eval=yvalid[8000:], reg=True, lamda=0.001)
# cls.load_param()
# cls.set_lr(lr=0.001)
# best_loss = cls.train(50, 20, if_eval=True, X_eval=Xvalid[8000:], y_eval=yvalid[8000:], fine_tune=True, best_loss=best_loss)

# 最佳模型参数测试
cls = Classifier.Classifier(256, 128, 10, lr=0.01)
cls.load_param('../best_param/best_linear_model.pickle')

# Evaluate test error
y_pred, acc = cls.predict(Xtest, ytest)
for i, idx in enumerate(np.where(np.argmax(ytest, axis=1) != y_pred)[0][:20]):
    plt.subplot(4, 5, i + 1)
    plt.axis('off')
    plt.imshow(Xtest[idx, :].reshape(16, 16, order='F'), cmap='gray')
    plt.title('pred : ' + str(y_pred[idx]) + ' label : ' + str(np.argmax(ytest, axis=1)[idx]))
plt.show()

# # Show train Loss and Acc
# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, label='train_loss')
# plt.scatter(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, s=5)
# plt.plot(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, label='eval_loss')
# plt.scatter(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, s=5)
# plt.legend()
# plt.title('Loss')
# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, label='train_loss')
# plt.scatter(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, s=5)
# plt.plot(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, label='eval_loss')
# plt.scatter(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, s=5)
# plt.legend()
# plt.title('Accuracy')
# plt.suptitle('acc = {}'.format(acc))
# plt.show()
