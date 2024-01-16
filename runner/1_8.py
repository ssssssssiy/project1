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

# 1.2.1 nHidden
acc_arr = []
for i in [5, 10, 20, 50, 100, 150, 200, 250, 350, 500, 750, 1000, 1500, 2000, 5000]:
    cls = Classifier.Classifier(256, i, 10, lr=0.01)
    cls.fit(X, yExpanded)
    cls.train(100, 3, if_eval=True, X_eval=Xvalid, y_eval=yvalid, save_path='../best_param/1_2_1.pickle')
    cls.load_param('../best_param/1_2_1.pickle')

# Evaluate test error
    y_pred, acc = cls.predict(Xtest, ytest)
    acc_arr.append(acc)

plt.plot(range(15), acc_arr)
plt.scatter(range(15), acc_arr, s=5)
plt.xticks(range(15), (5, 10, 20, 50, 100, 150, 200, 250, 350, 500, 750, 1000, 1500, 2000, 5000))
plt.xlabel('nHidden')
plt.ylabel('accuracy')
plt.show()

# # 1.2.2 adam & SGD
# cls = Classifier.Classifier(256, 128, 10, lr=0.01)
# cls.fit(X, yExpanded)
# cls.train(500, 50, if_eval=True, X_eval=Xvalid, y_eval=yvalid, save_path='../best_param/1_2_2.pickle')
# cls.load_param('../best_param/1_2_2.pickle')
# _, acc = cls.predict(Xtest, ytest)
# print(acc)
#
# # cls = Classifier.Classifier(256, 128, 10, lr=1.0)
# # cls.fit(X, yExpanded)
# # cls.train(500, 50, if_eval=True, X_eval=Xvalid, y_eval=yvalid, save_path='../best_param/1_2_2.pickle', adam=False)
# # cls.load_param('../best_param/1_2_2.pickle')
# # _, acc = cls.predict(Xtest, ytest)
# # print(acc)
#
# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, label='train_loss')
# plt.scatter(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, s=5)
# plt.plot(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, label='eval_loss')
# plt.scatter(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, s=5)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, label='train_acc')
# plt.scatter(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, s=5)
# plt.plot(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, label='eval_acc')
# plt.scatter(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, s=5)
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.legend()
#
# # plt.suptitle('SGD acc={}'.format(acc))
# plt.suptitle('Adam acc={}'.format(acc))
# plt.show()

# # 1.2.3 regulation
# cls = Classifier.Classifier(256, 128, 10, lr=0.01)
# cls.fit(X, yExpanded)
# cls.train(500, 50, if_eval=True, X_eval=Xvalid, y_eval=yvalid, save_path='../best_param/1_2_3.pickle', reg=False, lamda=0.01)
# cls.load_param('../best_param/1_2_3.pickle')
# _, acc = cls.predict(Xtest, ytest)
# print(acc)
# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, label='train_loss')
# plt.scatter(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, s=5)
# plt.plot(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, label='eval_loss')
# plt.scatter(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, s=5)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, label='train_acc')
# plt.scatter(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, s=5)
# plt.plot(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, label='eval_acc')
# plt.scatter(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, s=5)
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.legend()
# plt.suptitle('No regulation acc={}'.format(acc))
# plt.show()

# # 1.2.5 bias
# cls = Classifier.Classifier(256, 128, 10, lr=0.01)
# cls.fit(X, yExpanded)
# cls.train(500, 5, if_eval=True, X_eval=Xvalid, y_eval=yvalid, save_path='../best_param/early.pickle')
# cls.load_param('../best_param/early.pickle')
# _, acc = cls.predict(Xtest, ytest)
# print(acc)
#
# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, label='train_loss')
# plt.scatter(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, s=5)
# plt.plot(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, label='eval_loss')
# plt.scatter(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, s=5)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, label='train_acc')
# plt.scatter(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, s=5)
# plt.plot(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, label='eval_acc')
# plt.scatter(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, s=5)
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.legend()
#
# plt.suptitle('full bias acc={}'.format(acc))
# plt.show()

# # 1.2.6 drop out
# cls = Classifier.Classifier(256, 128, 10, lr=0.01)
# cls.fit(X, yExpanded)
# cls.train(50, 10, if_eval=True, X_eval=Xvalid, y_eval=yvalid, save_path='../best_param/1_2_4.pickle')
# cls.load_param('../best_param/1_2_4.pickle')
# _, acc = cls.predict(Xtest, ytest)
# print(acc)
# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, label='train_loss')
# plt.scatter(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, s=5)
# plt.plot(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, label='eval_loss')
# plt.scatter(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, s=5)
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, label='train_acc')
# plt.scatter(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, s=5)
# plt.plot(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, label='eval_acc')
# plt.scatter(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, s=5)
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.suptitle('drop_prob = 0.8 acc = {}'.format(acc))
# plt.show()

# # 1.2.7
# # 引入高斯噪声丰富训练集
# X_noise, y_noise = gauss_noise(X, yExpanded, sigma=0.3)
# # for i, idx in enumerate(range(0, 5000, 500)):
# #     plt.subplot(2, 5, i + 1)
# #     plt.axis('off')
# #     plt.imshow(X[idx].reshape(16, 16, order='F'), cmap='gray')
# #     plt.title('label : ' + str(np.argmax(yExpanded, axis=1)[idx]))
# # plt.show()
# #
# # for i, idx in enumerate(range(0, 5000, 500)):
# #     plt.subplot(2, 5, i + 1)
# #     plt.axis('off')
# #     plt.imshow(X_noise[idx].reshape(16, 16, order='F'), cmap='gray')
# #     plt.title('label : ' + str(np.argmax(y_noise, axis=1)[idx]))
# # plt.show()
#
# # cls = Classifier.Classifier(256, 128, 10, lr=0.01)
# # cls.fit(X, yExpanded)
# # cls.train(50, 20, if_eval=True, X_eval=Xvalid, y_eval=yvalid, save_path='../best_param/1_2_5.pickle')
# # cls.load_param('../best_param/1_2_5.pickle')
# # _, acc = cls.predict(Xtest, ytest)
# cls = Classifier.Classifier(256, 128, 10, lr=0.01)
# cls.fit(np.concatenate((X, X_noise), axis=0), np.concatenate((yExpanded, y_noise), axis=0))
# cls.train(50, 10, if_eval=True, X_eval=Xvalid, y_eval=yvalid, save_path='../best_param/1_2_5.pickle')
# cls.load_param()
# _, acc = cls.predict(Xtest, ytest)
#
# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, label='train_loss')
# plt.scatter(range(1, len(cls.train_epoch_loss) + 1), cls.train_epoch_loss, s=5)
# plt.plot(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, label='eval_loss')
# plt.scatter(range(1, len(cls.eval_epoch_loss) + 1), cls.eval_epoch_loss, s=5)
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, label='train_acc')
# plt.scatter(range(1, len(cls.train_epoch_acc) + 1), cls.train_epoch_acc, s=5)
# plt.plot(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, label='eval_acc')
# plt.scatter(range(1, len(cls.eval_epoch_acc) + 1), cls.eval_epoch_acc, s=5)
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.suptitle('Original Training Set acc={}'.format(acc))
# plt.show()
