import numpy as np
import layer_func.LinearLayer as Linear
import layer_func.CrossEntropyLoss as Loss
from layer_func import Softmax
from layer_func import Relu
from layer_func import Conv2d
import layer_func.AveragePooling as AvgPool
import pickle


class Classifier_conv(object):
    def __init__(self, out_num, lr=0.001):
        super(Classifier_conv, self).__init__()
        self.out_num = out_num
        self.lr = lr

        self.x_train = None
        self.y_train = None
        self.sample = None

        self.conv = Conv2d.Conv2d(in_chanel=1, out_chanel=10, kernel_size=3, stride=1, padding=True, kernel=None)  # B x 10 x 16 x 16
        self.relu = Relu.Relu()  # B x 10 x 16 x 16
        self.avg_pool = AvgPool.AvgPool(kernel_size=2, stride=2)  # B x 10 x 8 x 8
        self.dropout = Linear.LinearLayer(10 * 8 * 8, 256, dropout=True, drop_prob=0.2)
        self.linear = Linear.LinearLayer(256, out_num=10)
        self.softmax = Softmax.Softmax()
        self.loss_fn = Loss.CrossEntropyLoss()

        self.train_epoch_loss = []
        self.eval_epoch_loss = []

        self.best_loss = None

        self.train_epoch_acc = []
        self.eval_epoch_acc = []

        self.state = 1  # 1 : train, 0 : eval

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y
        self.sample = X.shape[0]

    def backward(self):
        radius = self.loss_fn.backward()
        radius = self.softmax.backward(radius) / 1000.0
        radius = self.linear.backward(radius)
        radius = self.dropout.backward(radius).reshape(radius.shape[0], 10, 8, 8)
        radius = self.avg_pool.backward(radius)
        radius = self.relu.backward(radius)
        _ = self.conv.backward(radius)

    def step_grad(self, adam=False):
        self.linear.step_grad(self.lr, adam=adam)
        self.dropout.step_grad(self.lr, adam=adam)
        self.conv.step_grad(self.lr, adam=adam)

    def clear_grad(self):
        self.linear.clear_grade()
        self.dropout.clear_grade()
        self.conv.clear_grade()

    def train(self, batch_size, epoch_num, if_eval=True, X_eval=None, y_eval=None, drop=True, shuffle=True, save_path=None, best_loss=None):
        batch_num = self.sample // batch_size

        save_path = '../best_param/best_conv_model.pickle' if save_path is None else save_path
        self.best_loss = 99999.0 if best_loss is None else best_loss
        for i in range(epoch_num):
            batch_loss = []
            # if not i and not i % 5:
            #     self.lr /= 2.0
            if shuffle:
                idx = np.random.permutation(self.sample)
                self.x_train = self.x_train[idx]
                self.y_train = self.y_train[idx]
            for j in range(batch_num):
                self.clear_grad()

                output = self.conv(self.x_train[batch_size * j: batch_size * (j + 1), :, :])
                output = self.relu(output)
                output = self.avg_pool(output).reshape(output.shape[0], 10 * 8 * 8)
                output = self.dropout(output)
                output = self.linear(output) / 1000.0  # make a "BatchNorm"
                # output = self.linear(output)
                output = self.softmax(output)
                loss = self.loss_fn(output, self.y_train[batch_size * j: batch_size * (j + 1), :])

                batch_loss.append(loss)

                self.backward()
                self.step_grad(adam=True)

                if not (i * batch_num + j) % 50:
                    print('[eopch : {} batch : {}] Loss : {}'.format(i, j, loss))

            self.train_epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.train_epoch_acc.append(self.predict(self.x_train, self.y_train)[1])
            if if_eval:  # 验证集调参
                self.eval_epoch_loss.append(self.valid(batch_size, X_eval, y_eval))
                self.eval_epoch_acc.append(self.predict(X_eval, y_eval)[1])

                if self.eval_epoch_loss[i] < self.best_loss:
                    self.best_loss = self.eval_epoch_loss[i]
                    self.save_param(save_path)
                if i and self.eval_epoch_loss[i] >= self.eval_epoch_loss[i - 1]:  # 提前停止
                    # break
                    continue

            else:  # 训练集 + 验证集 组成新训练集
                if self.train_epoch_loss[i] < self.best_loss:
                    self.best_loss = self.train_epoch_loss[i]
                    self.save_param(save_path)

                if i and self.train_epoch_loss[i] >= self.train_epoch_loss[i - 1]:  # 提前停止
                    # break
                    continue
        return self.best_loss

    def valid(self, batch_size, X_eval, y_eval):
        self.model_eval()  # 切换为评估模式
        eval_num = X_eval.shape[0]
        batch_num = eval_num // batch_size

        batch_loss = []
        for i in range(batch_num):
            output = self.conv(X_eval[batch_size * i: batch_size * (i + 1), :, :])
            output = self.relu(output)
            output = self.avg_pool(output).reshape(output.shape[0], 10 * 8 * 8)
            output = self.dropout(output)
            output = self.linear(output) / 1000.0  # make a "BatchNorm"
            # output = self.linear(output)
            output = self.softmax(output)
            loss = self.loss_fn(output, y_eval[batch_size * i: batch_size * (i + 1), :])
            batch_loss.append(loss)
        self.model_train()  # 切换为训练模式

        return sum(batch_loss) / len(batch_loss)

    def predict(self, X, y):
        self.model_eval()  # 切换为评估模式
        sample = X.shape[0]
        output = self.conv(X)
        output = self.relu(output)
        output = self.avg_pool(output).reshape(sample, 10 * 8 * 8)
        output = self.dropout(output)
        output = self.linear(output)
        yhat = np.argmax(output, axis=1)
        label = np.argmax(y, axis=1)
        correct = np.sum(yhat == label)
        print('accuracy : {} / {}'.format(correct, sample))
        self.model_train()  # 切换为训练模式
        return yhat, correct / sample

    def model_train(self):
        self.state = 1  # default
        self.dropout.dropout = self.linear.dropout_state

    def model_eval(self):
        self.state = 0
        self.dropout.dropout = False

    def show_features(self, X):
        self.model_eval()
        output = self.conv(X)
        return output

    def save_param(self, save_path='../best_param/best_conv_model.pickle'):
        best_param = {'conv': self.conv.kernel, 'linear': self.linear.param, 'dropout': self.dropout.param}
        file = open(save_path, 'wb')
        pickle.dump(best_param, file)
        file.close()

    def load_param(self, load_path='../best_param/best_conv_model.pickle'):
        file = open(load_path, 'rb')
        best_param = pickle.load(file)
        self.conv.kernel = best_param['conv']
        self.linear.param = best_param['linear']
        self.dropout.param = best_param['dropout']

    def set_lr(self, lr):
        self.lr = lr
