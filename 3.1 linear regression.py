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
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

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
    with torch.no_grad():
        for param in params:
            # 更新参数
            param -= lr * param.grad / batch_size
            # 清零梯度
            param.grad.zero_()

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
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

# 比较使用for循环和矢量化计算的效率
n = 10000
a = torch.ones(n)
b = torch.ones(n)

# 使用for循环计算
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'使用for循环计算时间: {timer.stop():.5f} sec')

# 使用矢量化计算
timer.start()
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
x = torch.arange(-7.0, 7.0, 0.01)
# 参数列表 (均值, 标准差)
params = [(0, 1), (0, 2), (3, 1)]
# 绘制曲线
plt.figure(figsize=(8, 6))
for mu, sigma in params:
    y = normal(x, mu, sigma)
    plt.plot(x.numpy(), y.numpy(), label=f'均值 {mu}, 标准差 {sigma}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
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
