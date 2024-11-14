print("3.1. 线性回归")

import torch
from torch import nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams # 导入 Matplotlib 的 rcParams 模块

# 设置默认字体为支持中文的字体（例如：SimHei黑体）
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他支持中文的字体
#Microsoft YaHei是微软雅黑，是微软公司为其Windows操作系统开发的一种无衬线字体。它在中文显示和印刷中表现良好，被广泛用于各种文档和网页设计中。

# 生成数据集
def generate_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的数据样本

    参数：
    w: 真正的权重（权重向量）
    b: 真正的偏置（标量）
    num_examples: 样本数量

    返回：
    features: 生成的输入特征 X
    labels: 生成的标签 y
    """
    #pytorch默认不能用len()函数来测量tensor的长度，所以要使用shape函数来测量tensor的形状，如
    # length = t.shape[0]  # 使用 .shape 属性
    # length = t.size(0)  # 使用 .size() 方法
    # 但是对于一维的tensor，可使用len()函数来测量tensor的长度

    # 生成特征 X，服从均值为0，标准差为1的正态分布
    X = torch.normal(0, 1, (num_examples, len(w)))    # 生成一个形状为 (num_examples, len(w)) 的张量，服从均值为0，标准差为1的正态分布
    #normal函数的参数    normal(mean, std, out=None) mean: 均值  std: 标准差  out: 输出张量，可选
    # 生成标签 y，使用矩阵乘法加上偏置，并添加噪声项
    y = torch.matmul(X, w) + b  #matmul函数用于矩阵乘法，X是一个形状为 (num_examples, len(w)) 的张量，w是一个形状为 (len(w), 1) 的张量
    # 添加噪声，噪声服从均值为0，标准差为0.01的正态分布
    y += torch.normal(0, 0.01, y.shape)
    # 返回特征和标签
    return X, y.reshape((-1, 1))    # reshape((-1, 1)) 实际上是将 y 的形状从 (num_examples,) 变为 (num_examples, 1)
# 其中 - 1 是一个占位符，表示让PyTorch自动计算该维度的大小，以确保张量中的元素总数不变。
#  (3,) ：tensor([1, 2, 3])
#  (3,1) : tensor([[1],
#         [2],
#         [3]])
# 如果y的形状是 (num_examples,) ，那么 y.reshape((-1, 1)) 的形状就是 (num_examples, 1) ，这里需要保证1*num_examples=num_examples，否则会报错。
# 相当于将y分开为2D的张量，2D张量的每一行只有一个元素，-1来计算出该有的组数

# 设置真实的参数
true_w = torch.tensor([2, -3.4])    # 权重向量
true_b = 4.2      # 偏置
# 生成数据
features, labels = generate_data(true_w, true_b, 1000)

# 在 torch.matmul(X, w) 中，X 和 w 的形状分别为 (num_examples, len(w)) 和 (len(w),)
# PyTorch 支持广播机制（broadcasting）来自动扩展张量的维度以进行计算。
# 在 PyTorch 中，当执行 torch.matmul(X, w) 时，w 会被自动视为一个二维列向量，其形状从 (len(w),) 扩展到 (len(w), 1)。这是为了符合矩阵乘法的规则。然而，实际的计算过程并不需要显式地扩展 w，因为 PyTorch 的广播机制会自动处理这个转换。

# 查看一下数据的形状
print("特征的形状:", features.shape)
print("标签的形状:", labels.shape)

# 绘制数据的分布（仅绘制第一个特征和标签的关系）
plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), 1)
plt.show()
#scatter函数用于绘制散点图，其中第一个参数是x轴的数据，第二个参数是y轴的数据，第三个参数是点的大小
#features[:, 0].detach().numpy()表示将features[:, 0]从tensor转换为numpy数组，detach()函数用于分离张量，以便在计算图中分离出该张量，numpy()函数用于将张量转换为numpy数组
# 在二维平面上绘制散点图时，使用每个样本的第一个特征作为 x 轴的值。

# features[:, 0]: 使用切片语法从 features 中提取所有行的第一列数据。在 PyTorch 中，: 表示选取该维度上的所有元素，0 表示选取第一个元素（在这个上下文中是第一列）。
# .detach(): 从当前计算图中分离出张量，并返回一个新的张量，这个新的张量不会有梯度（gradient）。这通常用于在不需要进行梯度计算的情况下操作张量，比如在绘图时。
# .numpy(): 将 PyTorch 张量转换为 NumPy 数组。这是因为 plt.scatter 函数（来自 Matplotlib 库）通常期望接收 NumPy 数组作为输入。


# 定义批量读取数据的函数
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器

    参数：
    data_arrays: 包含特征和标签的元组 (features, labels)
    batch_size: 批量大小
    is_train: 是否为训练模式

    返回：
    一个数据迭代器
    """
    # 创建数据集
    #torch.utils.data.TensorDataset(*data_arrays)将多个张量组合成一个数据集,数据集是一个元组，元组中的每个元素都是一个张量
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    # 创建数据迭代器
    # torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)
    # dataset: 数据集
    # batch_size: 批量大小
    # shuffle: 是否打乱数据顺序
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)   #

# 设置批量大小
batch_size = 10
# 创建数据迭代器
data_iter = load_array((features, labels), batch_size)
#(features, labels)是个元组，元组中的每个元素都是一个张量
#data_iter是一个数据迭代器，它可以用于在训练过程中迭代地获取数据
'''
批量大小（batch_size）和训练模式（is_train）是机器学习和深度学习中常用的概念，特别是在使用数据迭代器（如PyTorch的DataLoader）时。
批量大小（batch_size）
批量大小指的是在每次迭代（或称为一个批次、一个batch）中，从数据集中提取并用于训练或评估模型的样本数量。例如，如果数据集包含1000个样本，并且批量大小设置为100，那么在一次完整的训练周期（epoch）中，模型将会分10次（1000/100=10）处理这些数据，每次处理100个样本。
批量大小的选择对模型的训练效率和性能有影响：
较小的批量大小：可能导致训练过程更加不稳定，因为每次更新的梯度基于较少的样本。但这也可能有助于模型找到更好的全局最小值，并减少过拟合的风险。此外，较小的批量大小通常意味着更快的训练速度（尤其是在使用GPU时），因为每次迭代所需的计算量较少。
较大的批量大小：可以使训练过程更加稳定，因为每次更新的梯度基于更多的样本，从而更接近整个数据集的梯度。然而，这也可能导致模型陷入不太好的局部最小值，并增加过拟合的风险。此外，较大的批量大小可能需要更多的内存资源，并可能导致训练速度变慢。
训练模式（is_train）
训练模式是一个布尔标志，用于指示模型当前是否处于训练状态。在训练模式下，模型通常会启用某些特定的行为，如启用dropout层、batch normalization层的训练模式等。这些行为有助于在训练过程中正则化模型，防止过拟合，并促进模型学习到更鲁棒的特征表示。
在评估或测试模式下（即非训练模式），这些特定的训练时行为通常会被禁用或修改，以确保模型在评估时能够稳定、一致地输出预测结果。例如，dropout层在评估模式下会保持所有神经元激活，而不是随机丢弃一部分；batch normalization层会使用在训练过程中计算得到的运行均值和方差，而不是基于当前批次的统计量。
在PyTorch中，可以通过调用模型的.train()和.eval()方法来切换训练和评估模式。在自定义的数据加载函数中，如你提供的load_array函数，is_train参数用于控制数据迭代器是否应该在每个epoch开始时打乱数据顺序。在训练模式下，打乱数据顺序有助于模型更好地泛化到未见过的数据；而在评估模式下，通常不需要打乱数据顺序。
'''

# 定义线性回归模型
def linreg(X, w, b):
    """线性回归模型

    参数：
    X: 输入特征
    w: 权重参数
    b: 偏置参数

    返回：
    y_hat: 预测值
    """
    return torch.matmul(X, w) + b

# 定义损失函数（平方损失）
def squared_loss(y_hat, y):
    """计算平方损失

    参数：
    y_hat: 预测值
    y: 真实标签

    返回：
    损失值
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2    #即为(y_hat - y)^2/2

'''
损失函数（Loss Function）的计算，具体来说是平方损失函数（Squared Loss Function）的一个变体。
在机器学习和统计学习中，损失函数是用来衡量模型预测值（在这里是y_hat）与实际值（在这里是y）之间差距的函数。平方损失函数是一种常见的损失函数，其形式为(y_hat - y) ^ 2，即预测值与实际值之差的平方。
至于为什么要除以2，这通常是为了数学上的便利。在一些优化算法（如梯度下降法）中，对损失函数求导是必不可少的步骤。平方损失函数除以2之后的导数形式会更简洁，从而简化计算过程。具体来说，如果损失函数是(y_hat - y) ^ 2 / 2，那么其关于y_hat的导数就是y_hat - y，比不除以2时的导数2 * (y_hat - y)更简洁。
然而，需要注意的是，除以2并不会改变损失函数的基本性质（如凸性、最小值点等），因此在实际应用中，是否除以2通常取决于具体的算法实现和数学上的便利性。在一些情况下，为了保持与文献或其他实现的一致性，也可能会选择不除以2。
'''

# 定义优化算法（小批量随机梯度下降）
def sgd(params, lr, batch_size):
    """小批量随机梯度下降优化算法

    参数：
    params: 模型参数列表
    lr: 学习率
    batch_size: 批量大小

    返回：
    无返回值，直接更新参数
    """
    with torch.no_grad():   # 上下文管理器，用于禁用梯度计算
        #.no_grad()上下文管理器用于禁用梯度计算，这是因为在训练过程中，我们不需要计算梯度，因为我们已经在计算损失时使用了.backward()方法。
        for param in params:
            # 更新参数
            param -= lr * param.grad / batch_size
            # 清零梯度
            param.grad.zero_()
# with torch.no_grad():上下文管理器用于在不需要梯度信息的情况下禁用梯度计算，从而提高计算效率和减少内存消耗。在训练过程中，我们确实需要调用.backward()来计算梯度，但在某些情况下（如模型评估或预测），我们可以禁用梯度计算以获得更好的性能。
# 在with torch.no_grad():上下文内执行代码时，主要的区别是内存消耗和计算效率。由于禁用了梯度计算，因此不会跟踪操作历史，从而减少了内存消耗。此外，由于没有额外的梯度计算开销，代码的执行速度可能会更快。然而，需要注意的是，在这个上下文内无法获取梯度信息，因此不能用于训练过程中的梯度更新。

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 设置学习率和迭代次数
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

# 开始训练模型
for epoch in range(num_epochs):
    for X, y in data_iter:
        # 前向传播计算预测值
        y_hat = net(X, w, b)
        # 计算损失
        l = loss(y_hat, y)
        # 反向传播计算梯度
        l.sum().backward()
        # 更新参数
        sgd([w, b], lr, batch_size)
    # 每个epoch结束后，计算并输出损失
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 查看训练得到的参数与真实参数的差距
print(f'估计的w: {w.reshape(true_w.shape)}')
print(f'真实的w: {true_w}')
print(f'估计的b: {b}')
print(f'真实的b: {true_b}')

# 使用训练好的模型进行预测
def predict(X, w, b):
    """使用线性模型进行预测

    参数：
    X: 输入特征
    w: 权重参数
    b: 偏置参数

    返回：
    y_hat: 预测值
    """
    return torch.matmul(X, w) + b

# 生成新的样本用于预测
new_features = torch.tensor([[5.0, -2.0], [1.5, 3.0]])
# 进行预测
preds = predict(new_features, w, b)
print("预测结果:", preds)

# 定义一个计时器类，用于记录运行时间
class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]     #返回最后一次运行的时间

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()   #cumsum()函数用于计算数组中元素的累积和。tolist()函数用于将数组转换为列表。

# 比较使用for循环和矢量化计算的效率
n = 10000
a = torch.ones(n)
b = torch.ones(n)

# 使用for循环计算
c = torch.zeros(n)
timer = Timer()    # 开始计时
for i in range(n):
    c[i] = a[i] + b[i]
print(f'使用for循环计算时间: {timer.stop():.5f} sec')

# 使用矢量化计算
timer.start()     # 开始计时
d = a + b
print(f'使用矢量化计算时间: {timer.stop():.5f} sec')

# 计算正态分布概率密度函数
def normal(x, mu, sigma):
    """计算正态分布的概率密度函数

    参数：
    x: 自变量
    mu: 均值
    sigma: 标准差

    返回：
    概率密度值
    """
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * torch.exp(-0.5 / sigma**2 * (x - mu)**2)

# 绘制正态分布曲线
x = torch.arange(-7.0, 7.0, 0.01)     # 生成一个从 -7.0 到 7.0，步长为 0.01 的张量
# 参数列表 (均值, 标准差)
params = [(0, 1), (0, 2), (3, 1)]     # 均值和标准差的参数列表,(0, 1)表示均值为0，标准差为1的正态分布，(0, 2)表示均值为0，标准差为2的正态分布，(3, 1)表示均值为3，标准差为1的正态分布
# 绘制曲线
plt.figure(figsize=(8, 6))    # 设置图像大小
for mu, sigma in params:
    y = normal(x, mu, sigma)    # 画出不同的正态分布曲线，但范围是从 -7.0 到 7.0，且在一个图中
    plt.plot(x.numpy(), y.numpy(), label=f'均值 {mu}, 标准差 {sigma}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()    # 显示图例    也就是label
plt.show()

# 在代码的最后，用注释梳理一下本章用过的函数
# 本章用到的函数和类：
# torch.normal(mean, std, size): 生成正态分布的张量
#   参数：
#     mean: 均值，可以是标量或张量
#     std: 标准差，可以是标量或张量
#     size: 输出张量的大小
# torch.matmul(a, b): 矩阵乘法
#   参数：
#     a: 张量，形状为 (n, m)
#     b: 张量，形状为 (m, p)
# torch.utils.data.TensorDataset(*tensors): 将张量组合成数据集
#   参数：
#     *tensors: 任意数量的张量，这些张量的第一个维度大小应该相同
# torch.utils.data.DataLoader(dataset, batch_size, shuffle): 数据迭代器
#   参数：
#     dataset: 数据集
#     batch_size: 每个批量的大小
#     shuffle: 是否在每个epoch开始时打乱数据
# torch.zeros(size, requires_grad): 生成全零张量
#   参数：
#     size: 张量的大小
#     requires_grad: 是否需要计算梯度
# torch.ones(size): 生成全一张量
#   参数：
#     size: 张量的大小
# tensor.backward(): 计算梯度
#   无参数
# tensor.grad.zero_(): 清零梯度
#   无参数
# torch.no_grad(): 关闭梯度计算的上下文管理器
#   用于在不需要计算梯度时包裹代码块，节省内存和加速
# torch.arange(start, end, step): 生成等差数列张量
#   参数：
#     start: 起始值
#     end: 结束值
#     step: 步长
# torch.exp(tensor): 计算指数
#   参数：
#     tensor: 输入张量
# matplotlib.pyplot: 绘图库，用于绘制图形
# class Timer: 计时器类，用于测量代码运行时间
#   方法：
#     start(): 启动计时器
#     stop(): 停止计时器并记录时间
#     avg(): 返回平均时间
#     sum(): 返回时间总和
#     cumsum(): 返回累计时间列表


'''
在深度学习中，向前传播、反向传播和梯度计算是训练神经网络的核心步骤。下面我将逐一解释这些概念，并阐述它们之间的联系和重要性。

1. 向前传播（Forward Propagation）
概念：向前传播是神经网络中信息从输入层流向输出层的过程。在这个过程中，每个神经元接收前一层神经元的输出作为输入，并根据其权重和激活函数计算出自己的输出。这个输出再作为下一层神经元的输入，依此类推，直到达到输出层。
目的：向前传播的目的是为了根据当前的网络参数（权重和偏置）计算出神经网络对于给定输入的预测输出。
实例：假设我们有一个简单的两层神经网络，输入层有两个节点，隐藏层有一个节点，输出层有一个节点。当我们向输入层提供一组数据时（比如[0.05, 0.1]），这些数据会乘以输入层到隐藏层的权重，然后加上偏置，再经过激活函数得到隐藏层的输出。这个输出又会乘以隐藏层到输出层的权重，加上偏置，再经过激活函数得到最终的预测输出。

2. 反向传播（Backward Propagation）
概念：反向传播是在向前传播之后进行的一个过程，它的目的是根据预测输出和真实标签之间的误差来更新神经网络的参数（权重和偏置）。这个过程是通过计算损失函数对每个参数的梯度来实现的。
目的：反向传播的目的是通过调整神经网络的参数来减小预测输出与真实标签之间的误差，从而提高网络的预测准确性。
实例：继续上面的例子，假设我们的真实标签是[0.01, 0.99]，而神经网络的预测输出是[0.75, 0.25]。显然，这两者之间存在较大的误差。为了减小这个误差，我们需要通过反向传播来更新网络的参数。具体来说，我们会计算损失函数（比如均方误差）对每个参数的梯度，然后用这些梯度来更新参数的值。这样，在下一次向前传播时，网络就有可能产生更准确的预测输出。

3. 梯度计算（Gradient Calculation）
概念：梯度计算是反向传播过程中的一个关键步骤。它涉及到计算损失函数对每个参数的偏导数（即梯度），这些梯度指示了为了最小化损失函数应该如何调整参数的值。
目的：梯度计算的目的是为参数更新提供方向和大小的指导。通过沿着梯度的反方向更新参数（即梯度下降），我们可以逐渐减小损失函数的值，从而优化网络的性能。
实例：在反向传播过程中，我们会使用链式法则来计算损失函数对每个参数的梯度。以权重参数为例，我们需要计算损失函数关于该权重的偏导数。这个偏导数可以通过先计算损失函数关于下一层神经元输入的偏导数（即δ值），然后乘以该输入关于当前权重的偏导数来得到。这个过程会一直回溯到输入层，从而得到所有参数的梯度值。

4. 联系
向前传播、反向传播和梯度计算是相互依赖、相辅相成的三个步骤。向前传播提供了神经网络的预测输出，为我们评估网络性能提供了基础；反向传播则根据预测误差来指导如何调整网络参数以改进性能；而梯度计算则为参数更新提供了具体的数学依据和方向指导。这三个步骤循环进行，不断优化神经网络的参数和性能，直到达到满意的训练效果为止。
'''



# 梯度就是导数吗？
# 不完全准确。梯度是一个向量，它表示函数在某一点处沿着各个自变量方向的变化率。对于单变量函数，梯度就是该函数的导数。但对于多变量函数（如神经网络中的损失函数，它依赖于多个权重和偏置），梯度是一个包含所有偏导数的向量。
#
# 什么叫沿着梯度的反方向更新参数（即梯度下降）？什么又叫梯度下降？
# 梯度下降是一种优化算法，用于最小化损失函数。损失函数的梯度指示了函数值增加最快的方向。为了最小化损失函数，我们沿着梯度的反方向（即函数值减少最快的方向）更新参数。这个过程称为“沿着梯度的反方向更新参数”或简称为“梯度下降”。
#
# 关于反向传播和梯度计算的可视化解释
# 假设我们有一个简单的三层神经网络：输入层有2个节点，隐藏层有2个节点，输出层有1个节点。损失函数使用均方误差（MSE），真实标签为y_true，网络参数为W1（输入层到隐藏层的权重）、b1（输入层到隐藏层的偏置）、W2（隐藏层到输出层的权重）和b2（隐藏层到输出层的偏置）。
# 步骤1：向前传播
# 输入层到隐藏层：z1 = X * W1 + b1，a1 = sigmoid(z1) （sigmoid是激活函数）
# 隐藏层到输出层：z2 = a1 * W2 + b2，a2 = sigmoid(z2) （a2是网络的预测输出）
# 步骤2：计算损失
# 损失函数：L = 0.5 * (y_true - a2)^2
# 步骤3：反向传播和梯度计算
# 我们从输出层开始，逐步向输入层回溯，计算每个参数的梯度。
# 对于输出层：
# δ2 = ∂L/∂z2 = (a2 - y_true) * sigmoid_derivative(z2) （δ2是输出层的误差项）
# 对于隐藏层到输出层的权重和偏置：
# ∂L/∂W2 = δ2 * a1.T （.T表示转置）
# ∂L/∂b2 = δ2 （注意这里δ2是一个向量，所以∂L/∂b2也是对应每个样本的梯度向量）
# 对于隐藏层：
# δ1 = ∂L/∂z1 = (δ2 * W2.T) * sigmoid_derivative(z1) （δ1是隐藏层的误差项，通过链式法则从δ2传递回来）
# 对于输入层到隐藏层的权重和偏置：
# ∂L/∂W1 = δ1 * X.T
# ∂L/∂b1 = δ1
# 步骤4：参数更新（梯度下降）
# 使用计算出的梯度来更新参数：
# W1 = W1 - learning_rate * ∂L/∂W1
# b1 = b1 - learning_rate * ∂L/∂b1
# W2 = W2 - learning_rate * ∂L/∂W2
# b2 = b2 - learning_rate * ∂L/∂b2 （learning_rate是学习率，一个正的小数）
# 这个过程在训练过程中会反复进行，每次迭代都会使用一批新的训练样本，并更新网络参数以最小化损失函数。随着迭代的进行，网络的预测能力会逐渐提高。
#
#
# sigmoid激活函数
# sigmoid激活函数通常是指逻辑函数（logistic function），其数学形式为 1 / (1 + e^(-x))。这个函数可以将任意范围的输入映射到0到1之间，常用于二分类问题的输出层，以表示概率。sigmoid激活函数的主要目的是引入非线性，使得神经网络可以学习并逼近复杂的非线性函数。
# 然而，有时“sigmoid激活函数”也可能泛指具有S形曲线的激活函数，如tanh等。这些函数都具有类似的特性，即能够将输入值映射到一个有限的输出范围内。
#
# learning_rate的作用
# learning_rate（学习率）是梯度下降算法中的一个关键参数，它决定了在每次迭代中参数更新的步长大小。合适的学习率能够使算法快速收敛到损失函数的最小值。
# 学习率过大的影响：如果学习率设置得过大，可能会导致算法在最小值附近震荡而无法收敛，甚至可能使损失函数值增大。
# 学习率过小的影响：如果学习率设置得过小，虽然可以保证算法的收敛性，但会导致收敛速度非常慢，需要更多的迭代次数才能达到较好的效果。
#
#
# δ2 不是L对W2的偏导数，而是L对z2的偏导数。z2是隐藏层到输出层的加权输入，是W2和a1的函数。因此，当我们需要计算L对W2的偏导数时，我们会使用链式法则，通过δ2（即∂L/∂z2）和a1来计算。具体来说，∂L/∂W2 = δ2 * a1.T，这里的*表示逐元素的乘法（对于向量或矩阵而言，实际上是外积）。这样做的原因是z2是关于W2和a1的函数，所以我们需要将误差项δ2与a1相乘来得到关于W2的梯度。
# 求出的偏导数（梯度）将用于更新网络的权重和偏置。在梯度下降算法中，我们会沿着梯度的反方向更新参数，以减小损失函数的值。具体来说，对于权重W和偏置b，更新公式通常为：W = W - learning_rate * ∂L/∂W 和 b = b - learning_rate * ∂L/∂b。
# 在反向传播过程中，我们需要求出损失函数关于每一层权重和偏置的偏导数。对于上述的三层神经网络示例，我们需要求出∂L/∂W1、∂L/∂b1、∂L/∂W2和∂L/∂b2这四个偏导数。这些偏导数将用于更新网络的参数，以最小化损失函数并提高网络的预测性能。
#
# 在表达式 W = W - learning_rate * ∂L/∂W 中，∂L/∂W 表示损失函数 L 关于权重矩阵 W 的偏导数。这个偏导数不是一个函数，而是在当前权重 W 和其他相关变量（如 b、输入数据 X、真实标签 y_true 等）取值下计算出来的一个具体的矩阵。这个矩阵与权重矩阵 W 具有相同的维度。