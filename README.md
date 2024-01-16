# project1
work for AI security class
使用 python 完成，主要功能仅使用 numpy 等 python 基本库。
### 文件结构（主要内容）：

```
│  digits.mat
│  report.pdf
│
├─best_param............................最佳参数
│      best_conv_model.pickle			CNN模型最佳参数
│      best_linear_model.pickle			线性模型最佳参数
│
├─handwrite.............................自制16x16手写数字图像
│      handwrite_0.png
│      handwrite_1.png
│      handwrite_2.png
│      handwrite_3.png
│      handwrite_4.png
│      handwrite_5.png
│      handwrite_6.png
│      handwrite_7.png
│      handwrite_8.png
│      handwrite_9.png
│
├─layer_func.............................layer及功能函数
│     AveragePooling.py
│     Conv2d.py
│     CrossEntropyLoss.py
│     gauss_noise.py
│     LinearLayer.py
│     Relu.py
│     Sigmoid.py
│     Softmax.py
│
├─model..................................模型
│     Classifier.py
│     Classifier_conv.py
│
├─runner.................................实验及模型训练
│      1_8.py
│      9.py
│      Conv2D_Model.py
│      handwrite_test.py.................训练最佳模型在自制图像上的测试
│      Linear_Model.py
```

#### Layer

##### 线性层 LinearLayer [1.2.1]

实现了前向、后向过程

（可选参数）实现了偏置 $ b $ [1.2.6]；实现了 drop out [1.2.7]； （取代动量法）实现了 Adam 参数更新 [1.2.2]；实现了参数 $ W $ 的 $ L_2 $ 正则化 [1.2.4]

##### 激活函数 Relu

实现了 Relu 激活函数： $ y = max(0, x) $ 的前向、后向过程

##### 概率归一化 Softmax [1.2.5]

实现了归一化概率分布 Softmax： $ p(y_i) = \frac{exp(z_i)}{\sum_{j=1}^nexp(z_j)} $ 的前向、后向过程，为了避免溢出做了平移处理： $ z_i = z_i - max_n(z_j) $

##### 交叉熵损失函数 CrossEntropyLoss [1.2.5]

实现了交叉熵损失函数 $ Loss = \frac{1}{N}\sum_{i=1}^n{-y_i}log(\hat{y_i}) $ 的前向、后向过程

（可选参数）在训练时对可学习参数进行 Adam 参数更新；对线性层进行参数 $ W $ 的 $ L_2 $ 正则化

##### 2D卷积层 Conv2d [1.2.10]

实现了2D卷积的前向、后向过程；

（可选参数）实现了卷积核大小、步长、padding自定义；实现了多通道卷积，可自定义输入输出通道数；实现了 Adam 参数更新

##### 均值池化层 AveragePooling

实现了均值池化前向、后向过程

（可选参数）实现了卷积核大小、步长

#### 数据处理

批归一化

二值化处理

#### 其他功能

批训练，向量化、矩阵化计算过程 [1.2.3]

提前停止训练过程 [1.2.4]

模型训练时可选 fine-tune 对最后一层（线性层）进行微调 [1.2.8]

为训练集加入高斯噪声，扩展训练集；通过验证集确定超参数后，将验证集合并入训练集进行训练 [1.2.9]

### 模型选择

未经修改的 matlab 参考代码在经过 100000 步迭代后，错误率仍然高达 32%

最终选择了以下两个模型

#### 线性模型

##### 模型选择：

Linear(256 --> 128, bias, drop_prob=0.2) ==> Relu ==> Linear(128 --> 10, bias, drop_prob=0.2)  ==> Softmax ==> CrossEntropyLoss

##### 训练集选择：

为训练集 (5000) 和验证集 (5000) 增加高斯噪声 (sigma=0.4) 合并后，选取验证集中的 20% (2000) 作为新的验证集，其余部分 (18000) 作为新的训练集投入训练

##### 训练过程：

batch_size = 50, epoch = 20, lr = 0.01, **reg**=True, lamda=0.001

batch_size = 50, epoch = 20, lr = 0.001, **fine_tune**=True

训练过程中保存 eval_loss 最小的参数作为最佳模型参数

##### 模型结果：

在测试集上最高正确率：981 / 1000


##### 最佳模型参数保存地址：

best_param/best_linear_model.pickle

##### 复现方法：

运行 runner/Linear_Model.py

##### 分析：

```
对数据集进行了增广和重新划分。
第一次训练进行正则化约束，第二次训练进行 fine-tune，两次训练均使用 drop oup，增强了模型的泛化能力。
可以发现在测试集上预测错误的图像确实不易区分，对线性模型来说难以识别，譬如上图中2行1列,3行1列,4行3列的图像，人类也不好轻易识别。
由于该任务比较简单，线性模型在该任务上能达到较好的表现。
```

#### CNN 模型

##### 模型选择：

Conv2D(1x16x16 --> 10x16x16, kernel_size=3, stride=1, padding=True) ==> Relu

==> AveragePooling(10x16x16 --> 10x8x8, kernel_size=2, stride=2) ==> Linear(640 --> 256, drop_prob=0.2) 

==> Linear(256 --> 10, bias) ==> Softmax ==> CrossEntropyLoss

##### 训练集选择：

为训练集 (5000) 和验证集 (5000) 增加高斯噪声 (sigma=0.4) 合并后，选取验证集中的 20% (2000) 作为新的验证集，其余部分 (18000) 作为新的训练集投入训练

##### 训练过程：

batch_size = 25, epoch = 20, lr = 0.01

batch_size = 25, epoch = 20, lr = 0.001

##### 模型结果

在测试集上最高正确率：984 / 1000

##### 最佳模型参数保存地址：

best_param/best_conv_model.pickle

##### 复现方法：

运行 runner/Conv2D_Model.py

##### 分析：

```
对数据集进行了增广和重新划分。
使用了均值池化等手段，增强了模型的泛化性；调整了学习率，使模型能充分得到训练。
可以发现在测试集上预测错误的图像在形态上有一些混淆特征，不易被卷积网络识别。
卷积模型能够更好的提取图像的信息，比线性模型能力更强，表现更好，在复杂的图像分类任务上更能体现优势。
```

##### 线性模型预测结果：

accuracy ：8 / 10	 ypred ：[5, 1, 2, 3, 4, 5, 8, 7, 8, 9]

##### CNN 模型预测结果：

accuracy ：10 / 10	ypred ：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]，优于线性模型
