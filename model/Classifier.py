import numpy as np
from layer_func import LinearLayer as Linear
from layer_func import CrossEntropyLoss as Loss
from layer_func import Softmax
from layer_func import Relu
from layer_func import Sigmoid
import pickle


class Classifier(object):
    def __init__(self, in_num, hidden_num, out_num, lr=0.001):
        super(Classifier, self).__init__()
        self.in_num = in_num
        self.hidden_num = hidden_num
        self.out_num = out_num
        self.lr = lr

        self.x_train = None
        self.y_train = None
        self.sample = None

        self.linear_1 = Linear.LinearLayer(in_num, hidden_num, dropout=False)
        self.relu = Relu.Relu()
        # self.sigmoid = Sigmoid.Sigmoid()
        self.linear_2 = Linear.LinearLayer(hidden_num, out_num, dropout=True, drop_prob=0.2)
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

    def backward(self, reg=False, lamda=0.01):
        radius = self.loss_fn.backward()
        radius = self.softmax.backward(radius) / 100.0
        # radius = self.softmax.backward(radius)
        radius = self.linear_2.backward(radius, reg=reg, lamda=lamda) / (1 - self.linear_2.drop_prob) if self.linear_2.dropout else self.linear_2.backward(radius, reg=reg, lamda=lamda)
        radius = self.relu.backward(radius)
        # radius = self.sigmoid.backward(radius)
        _ = self.linear_1.backward(radius, reg=reg, lamda=lamda) / (1 - self.linear_1.drop_prob) if self.linear_1.dropout else self.linear_1.backward(radius, reg=reg, lamda=lamda)

    def step_grad(self, adam=False):
        self.linear_2.step_grad(self.lr, adam=adam)
        self.linear_1.step_grad(self.lr, adam=adam)

    def clear_grad(self):
        self.linear_2.clear_grade()
        self.linear_1.clear_grade()

    def train(self, batch_size, epoch_num, reg=False, lamda=0.01, if_eval=True, X_eval=None, y_eval=None, drop=True, shuffle=True, fine_tune=False, save_path=None, adam=True, best_loss=None):
        batch_num = self.sample // batch_size
        save_path = '../best_param/best_linear_model.pickle' if save_path is None else save_path
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

                output = self.linear_1(self.x_train[batch_size * j: batch_size * (j + 1), :] / (1 - self.linear_1.drop_prob)) if self.linear_1.dropout else self.linear_1(self.x_train[batch_size * j: batch_size * (j + 1), :])
                output = self.relu(output)
                # output = self.sigmoid(output)
                output = self.linear_2(output / (1 - self.linear_2.drop_prob)) / 100.0 if self.linear_2.dropout else self.linear_2(output) / 100.0  # make a "BatchNorm"
                # output = self.linear_2(output / (1 - self.linear_2.drop_prob)) if self.linear_2.dropout else self.linear_2(output)  # make a "BatchNorm"

                output = self.softmax(output)
                loss = self.loss_fn(output, self.y_train[batch_size * j: batch_size * (j + 1), :]) + lamda * (np.linalg.norm(self.linear_1.param['w'], ord='fro') + np.linalg.norm(self.linear_2.param['w'], ord='fro')) / batch_size if reg else self.loss_fn(output, self.y_train[batch_size * j: batch_size * (j + 1), :])
                batch_loss.append(loss)
                self.backward(reg=reg, lamda=lamda)
                self.fine_tune(adam=adam) if fine_tune else self.step_grad(adam=adam)

                if not (i * batch_num + j) % 20:
                    print('[eopch : {} batch : {}] Loss : {}'.format(i, j, loss))

            self.train_epoch_loss.append(sum(batch_loss) / len(batch_loss))

            self.train_epoch_acc.append(self.predict(self.x_train, self.y_train)[1])
            if if_eval:  # 验证集调参
                self.eval_epoch_loss.append(self.valid(batch_size, X_eval, y_eval, reg=reg, lamda=lamda))

                self.eval_epoch_acc.append(self.predict(X_eval, y_eval)[1])

                if i and self.eval_epoch_loss[i] >= self.eval_epoch_loss[i - 1]:  # 提前停止
                    # break
                    continue
                else:
                    if self.eval_epoch_loss[i] < self.best_loss:
                        self.best_loss = self.eval_epoch_loss[i]
                        self.save_param(save_path)
            else:  # 训练集 + 验证集 组成新训练集
                if i and self.train_epoch_loss[i] >= self.train_epoch_loss[i - 1]:  # 提前停止
                    # break
                    continue
                else:
                    if self.train_epoch_loss[i] < self.best_loss:
                        self.best_loss = self.train_epoch_loss[i]
                        self.save_param(save_path)
        return self.best_loss

    def fine_tune(self, adam):
        self.linear_2.step_grad(self.lr, adam=adam)

    def valid(self, batch_size, X_eval, y_eval, reg=False, lamda=0.001):
        self.model_eval()  # 切换为评估模式
        eval_num = X_eval.shape[0]
        batch_num = eval_num // batch_size

        batch_loss = []
        for i in range(batch_num):
            output = self.linear_1(X_eval[batch_size * i: batch_size * (i + 1), :])
            output = self.relu(output)
            # output = self.sigmoid(output)
            output = self.linear_2(output) / 100.0  # make a "BatchNorm"
            # output = self.linear_2(output)

            output = self.softmax(output)
            loss = self.loss_fn(output, y_eval[batch_size * i: batch_size * (i + 1), :]) + lamda * (np.linalg.norm(self.linear_1.param['w'], ord='fro') + np.linalg.norm(self.linear_2.param['w'], ord='fro')) / batch_size if reg else self.loss_fn(output, y_eval[batch_size * i: batch_size * (i + 1), :])
            batch_loss.append(loss)
        self.model_train()  # 切换为训练模式

        return sum(batch_loss) / len(batch_loss)

    def predict(self, X, y):
        self.model_eval()  # 切换为评估模式
        sample = X.shape[0]
        output = self.linear_1(X)
        output = self.relu(output)
        # output = self.sigmoid(output)
        output = self.linear_2(output)
        yhat = np.argmax(output, axis=1)
        label = np.argmax(y, axis=1)
        correct = np.sum(yhat == label)
        print('accuracy : {} / {}'.format(correct, sample))
        self.model_train()  # 切换为训练模式
        return yhat, correct / sample

    def model_train(self):
        self.state = 1  # default 1: train
        self.linear_1.dropout = self.linear_1.dropout_state
        self.linear_2.dropout = self.linear_2.dropout_state

    def model_eval(self):
        self.state = 0  # 0: eval
        self.linear_1.dropout = False
        self.linear_2.dropout = False

    def save_param(self, save_path='../best_param/best_linear_model.pickle'):
        best_param = {'linear_1': self.linear_1.param, 'linear_2': self.linear_2.param}
        file = open(save_path, 'wb')
        pickle.dump(best_param, file)
        file.close()

    def load_param(self, load_path='../best_param/best_linear_model.pickle'):
        file = open(load_path, 'rb')
        best_param = pickle.load(file)
        self.linear_1.param = best_param['linear_1']
        self.linear_2.param = best_param['linear_2']

    def set_lr(self, lr):
        self.lr = lr
