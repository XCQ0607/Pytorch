# 3.2. 线性回归的从零开始实现
print("3.2. 线性回归的从零开始实现")

# 导入必要的库
import torch
from torch import nn
import numpy as np
import random

# 设置随机种子以确保结果可重复
torch.manual_seed(0)

# ----------------------------------------------------------------------
print("\n----------- 生成数据集示例 -----------\n")
# 定义生成数据集的函数
def synthetic_data(w, b, num_examples):
    """
    生成 y = Xw + b + 噪声 的数据

    参数：
    w (Tensor): 权重向量
    b (float): 偏置项
    num_examples (int): 样本数

    返回：
    features (Tensor): 特征张量
    labels (Tensor): 标签张量
    """
    # 正态分布随机生成特征
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 矩阵乘法计算标签
    y = torch.matmul(X, w) + b
    # 加入噪声
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 真实的权重和偏置
true_w = torch.tensor([2, -3.4, 5.6])
true_b = 4.2
# 生成数据集
features, labels = synthetic_data(true_w, true_b, 1000)

# 输出数据集的部分内容
print("特征样本：", features[0])
print("标签样本：", labels[0])

# ----------------------------------------------------------------------
print("\n----------- 数据迭代器示例 -----------\n")
# 定义数据迭代器
def data_iter(batch_size, features, labels):
    """
    创建一个数据迭代器

    参数：
    batch_size (int): 批量大小
    features (Tensor): 特征张量
    labels (Tensor): 标签张量

    生成：
    每次迭代返回一小批量特征和标签
    """
    num_examples = len(features)
    # 生成索引并随机打乱
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 获取一个批量的索引
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        # 返回对应的特征和标签
        yield features[batch_indices], labels[batch_indices]

# 设置批量大小
batch_size = 10
# 获取并打印一个批量的数据
for X, y in data_iter(batch_size, features, labels):
    print("批量特征：", X)
    print("批量标签：", y)
    break

# ----------------------------------------------------------------------
print("\n----------- 初始化模型参数示例 -----------\n")
# 初始化模型参数
w = torch.normal(0, 0.01, size=(3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print("初始化的权重：", w)
print("初始化的偏置：", b)

# ----------------------------------------------------------------------
print("\n----------- 定义模型示例 -----------\n")
# 定义线性回归模型
def linreg(X, w, b):
    """
    线性回归模型

    参数：
    X (Tensor): 输入特征
    w (Tensor): 权重
    b (Tensor): 偏置

    返回：
    y_hat (Tensor): 预测输出
    """
    return torch.matmul(X, w) + b

# 模型测试
X_sample = features[:5]
print("模型输入：", X_sample)
print("模型输出：", linreg(X_sample, w, b))

# ----------------------------------------------------------------------
print("\n----------- 定义损失函数示例 -----------\n")
# 定义损失函数（均方误差）
def squared_loss(y_hat, y):
    """
    计算预测值和真实值之间的平方损失

    参数：
    y_hat (Tensor): 预测值
    y (Tensor): 真实值

    返回：
    loss (Tensor): 损失值
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 损失函数测试
y_hat_sample = linreg(X_sample, w, b)
loss_sample = squared_loss(y_hat_sample, labels[:5])
print("预测值：", y_hat_sample)
print("真实值：", labels[:5])
print("损失值：", loss_sample)

# ----------------------------------------------------------------------
print("\n----------- 定义优化算法示例 -----------\n")
# 定义优化算法（小批量随机梯度下降）
def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降

    参数：
    params (list): 模型参数列表 [w, b]
    lr (float): 学习率
    batch_size (int): 批量大小
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            # 梯度清零
            param.grad.zero_()

# 优化算法测试
# 假设我们有一些梯度
w.grad = torch.ones_like(w)
b.grad = torch.ones_like(b)
# 执行一次更新
sgd([w, b], lr=0.03, batch_size=10)
print("更新后的权重：", w)
print("更新后的偏置：", b)

# ----------------------------------------------------------------------
print("\n----------- 训练模型示例 -----------\n")
# 设置训练参数
lr = 0.03  # 学习率
num_epochs = 5  # 迭代周期
net = linreg  # 模型
loss = squared_loss  # 损失函数

# 开始训练
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 前向传播计算预测值
        y_hat = net(X, w, b)
        # 计算损失
        l = loss(y_hat, y)
        # 反向传播计算梯度
        l.sum().backward()
        # 更新参数
        sgd([w, b], lr, batch_size)
    # 计算并打印每个epoch的损失
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# ----------------------------------------------------------------------
print("\n----------- 结果分析示例 -----------\n")
# 比较学到的参数和真实参数
print("学到的权重：", w.reshape(true_w.shape))
print("真实的权重：", true_w)
print("权重估计误差：", true_w - w.reshape(true_w.shape))
print("学到的偏置：", b)
print("真实的偏置：", true_b)
print("偏置估计误差：", true_b - b)

# ----------------------------------------------------------------------
print("\n----------- 可选的改进示例 -----------\n")
# 增加L1正则化项
def l1_penalty(w):
    """
    L1正则化项

    参数：
    w (Tensor): 权重

    返回：
    Tensor: L1正则化值
    """
    return torch.abs(w).sum()

# 修改损失函数，加入正则化
def squared_loss_with_l1(y_hat, y, w, lambda_):
    """
    带有L1正则化的平方损失

    参数：
    y_hat (Tensor): 预测值
    y (Tensor): 真实值
    w (Tensor): 权重
    lambda_ (float): 正则化系数

    返回：
    loss (Tensor): 损失值
    """
    return squared_loss(y_hat, y) + lambda_ * l1_penalty(w)

# 重新初始化参数
w = torch.normal(0, 0.01, size=(3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 重新训练模型，带有L1正则化
lambda_ = 0.1  # 正则化系数
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = net(X, w, b)
        l = squared_loss_with_l1(y_hat, y, w, lambda_)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = squared_loss_with_l1(net(features, w, b), labels, w, lambda_)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# ----------------------------------------------------------------------
print("\n----------- 优化算法改进示例 -----------\n")
# 使用Adam优化器
# 重新初始化参数
w = torch.normal(0, 0.01, size=(3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# 定义Adam优化器
optimizer = torch.optim.Adam([w, b], lr=0.03)

# 训练模型
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = net(X, w, b)
        l = loss(y_hat, y)
        optimizer.zero_grad()  # 梯度清零
        l.sum().backward()
        optimizer.step()  # 更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# ----------------------------------------------------------------------
print("\n----------- 总结 -----------\n")
print("在本次代码示例中，我们从零开始实现了线性回归模型，进行了数据集的生成、模型的定义、损失函数的定义、优化算法的实现和模型的训练。我们还展示了如何加入L1正则化项和使用PyTorch内置的优化器。")

# ----------------------------------------------------------------------

# 总结使用到的函数：
# 1. torch.normal(mean, std, size): 生成正态分布的张量
#    - mean (float or Tensor): 均值
#    - std (float or Tensor): 标准差
#    - size (int...): 张量的形状
#    示例：X = torch.normal(0, 1, (1000, 3))

# 2. torch.matmul(tensor1, tensor2): 矩阵乘法
#    - tensor1 (Tensor): 第一个张量
#    - tensor2 (Tensor): 第二个张量
#    示例：y = torch.matmul(X, w)

# 3. torch.zeros(size, requires_grad): 生成全零张量
#    - size (int...): 张量的形状
#    - requires_grad (bool): 是否需要计算梯度
#    示例：b = torch.zeros(1, requires_grad=True)

# 4. torch.tensor(data, requires_grad): 生成张量
#    - data: 数据
#    - requires_grad (bool): 是否需要计算梯度
#    示例：true_w = torch.tensor([2, -3.4, 5.6])

# 5. torch.abs(tensor): 计算张量的绝对值
#    - tensor (Tensor): 输入张量
#    示例：l1_penalty = torch.abs(w).sum()

# 6. torch.optim.Adam(params, lr): Adam优化器
#    - params (iterable): 待优化的参数
#    - lr (float): 学习率
#    示例：optimizer = torch.optim.Adam([w, b], lr=0.03)

# 7. optimizer.zero_grad(): 梯度清零
#    示例：optimizer.zero_grad()

# 8. optimizer.step(): 执行优化步骤，更新参数
#    示例：optimizer.step()

# 以上函数在代码中被使用到，参数的选择可以根据具体需求进行调整。
