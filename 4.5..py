# -*- coding: utf-8 -*-
"""
4.5. 权重衰减（Weight Decay）示例代码

此代码示例演示了如何在PyTorch中实现权重衰减，以解决过拟合问题。
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 配置中文字体
rcParams['font.family'] = 'Microsoft YaHei'

print("4.5. 权重衰减")

# 分割线
print("=" * 50)

# 设置随机数种子，保证可重复性
torch.manual_seed(0)

# 生成高维线性回归数据集
def generate_synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的数据

    参数：
    w (Tensor): 权重向量
    b (float): 偏置
    num_examples (int): 样本数量

    返回：
    features (Tensor): 特征矩阵
    labels (Tensor): 标签向量
    """
    features = torch.randn(num_examples, w.shape[0])
    labels = torch.matmul(features, w) + b
    labels += torch.normal(0, 0.01, size=labels.shape)  # 添加噪声
    return features, labels.reshape(-1, 1)

# 定义真实参数
num_inputs = 200
true_w = torch.ones(num_inputs, 1) * 0.01
true_b = 0.05

# 生成训练和测试数据集
n_train, n_test = 20, 100  # 训练样本数和测试样本数
features_train, labels_train = generate_synthetic_data(true_w, true_b, n_train)
features_test, labels_test = generate_synthetic_data(true_w, true_b, n_test)

# 数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """构造PyTorch数据迭代器

    参数：
    data_arrays (tuple): 包含特征和标签的元组
    batch_size (int): 批量大小
    is_train (bool): 是否打乱数据

    返回：
    data_iter (DataLoader): 数据迭代器
    """
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 5
train_iter = load_array((features_train, labels_train), batch_size)
test_iter = load_array((features_test, labels_test), batch_size, is_train=False)

# 定义模型
def linreg(X, w, b):
    """线性回归模型

    参数：
    X (Tensor): 输入特征
    w (Tensor): 权重
    b (Tensor): 偏置

    返回：
    Tensor: 模型输出
    """
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """平方损失函数

    参数：
    y_hat (Tensor): 预测值
    y (Tensor): 真实值

    返回：
    Tensor: 损失
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法（小批量随机梯度下降）
def sgd(params, lr, batch_size):
    """小批量随机梯度下降优化算法

    参数：
    params (list): 模型参数
    lr (float): 学习率
    batch_size (int): 批量大小
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 定义L2范数惩罚项
def l2_penalty(w):
    """计算L2范数惩罚项

    参数：
    w (Tensor): 权重向量

    返回：
    Tensor: L2惩罚项
    """
    return torch.sum(w.pow(2)) / 2

# 训练函数
def train(lambd):
    """训练模型并绘制训练和测试损失曲线

    参数：
    lambd (float): 权重衰减（L2正则化）系数
    """
    # 初始化参数
    w = torch.normal(0, 0.01, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 定义学习率和训练周期
    lr = 0.003
    num_epochs = 100

    # 用于绘图的数据
    train_loss = []
    test_loss = []

    # 训练过程
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 计算模型输出
            y_hat = linreg(X, w, b)
            # 计算损失（包含L2惩罚项）
            loss = squared_loss(y_hat, y) + lambd * l2_penalty(w)
            # 反向传播
            loss.sum().backward()
            # 更新参数
            sgd([w, b], lr, batch_size)
        # 记录损失
        with torch.no_grad():
            train_loss.append(squared_loss(linreg(features_train, w, b), labels_train).mean().item())
            test_loss.append(squared_loss(linreg(features_test, w, b), labels_test).mean().item())
        # 每10个周期打印一次结果
        if (epoch + 1) % 10 == 0:
            print(f'周期 {epoch + 1}, 训练损失 {train_loss[-1]:f}, 测试损失 {test_loss[-1]:f}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, num_epochs + 1), train_loss, label='训练损失')
    plt.semilogy(range(1, num_epochs + 1), test_loss, label='测试损失')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.legend()
    plt.show()

    # 输出权重的L2范数
    print('w的L2范数：', torch.norm(w).item())

# 分割线
print("-" * 50)
print("不使用权重衰减（lambda=0）")
train(lambd=0)

# 分割线
print("-" * 50)
print("使用权重衰减（lambda=3）")
train(lambd=3)

# 分割线
print("=" * 50)
print("使用PyTorch内置函数实现权重衰减")

def train_concise(wd):
    """使用PyTorch内置的优化器参数实现权重衰减

    参数：
    wd (float): 权重衰减（L2正则化）系数
    """
    # 定义模型
    net = nn.Linear(num_inputs, 1)
    # 初始化参数
    nn.init.normal_(net.weight, mean=0, std=0.01)
    nn.init.zeros_(net.bias)

    # 定义损失函数
    loss = nn.MSELoss()

    # 定义优化器，传入权重衰减参数weight_decay
    optimizer = torch.optim.SGD([
        {'params': net.weight, 'weight_decay': wd},
        {'params': net.bias}
    ], lr=0.003)

    num_epochs = 100

    # 用于绘图的数据
    train_loss = []
    test_loss = []

    # 训练过程
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        # 记录损失
        with torch.no_grad():
            train_loss.append(loss(net(features_train), labels_train).item())
            test_loss.append(loss(net(features_test), labels_test).item())
        # 每10个周期打印一次结果
        if (epoch + 1) % 10 == 0:
            print(f'周期 {epoch + 1}, 训练损失 {train_loss[-1]:f}, 测试损失 {test_loss[-1]:f}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, num_epochs + 1), train_loss, label='训练损失')
    plt.semilogy(range(1, num_epochs + 1), test_loss, label='测试损失')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.legend()
    plt.show()

    # 输出权重的L2范数
    print('w的L2范数：', net.weight.norm().item())

# 分割线
print("-" * 50)
print("不使用权重衰减（weight_decay=0）")
train_concise(0)

# 分割线
print("-" * 50)
print("使用权重衰减（weight_decay=3）")
train_concise(3)

# 分割线
print("=" * 50)
print("练习题解答")

# 1. 使用不同的lambda值进行实验，绘制训练和测试精度关于lambda的函数。

lambda_values = [0, 0.1, 1, 3, 5, 10]
train_losses = []
test_losses = []
norms = []

for lambd in lambda_values:
    # 初始化参数
    w = torch.normal(0, 0.01, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 定义学习率和训练周期
    lr = 0.003
    num_epochs = 100

    # 训练过程
    for epoch in range(num_epochs):
        for X, y in train_iter:
            y_hat = linreg(X, w, b)
            loss = squared_loss(y_hat, y) + lambd * l2_penalty(w)
            loss.sum().backward()
            sgd([w, b], lr, batch_size)

    # 记录损失和权重范数
    with torch.no_grad():
        train_loss = squared_loss(linreg(features_train, w, b), labels_train).mean().item()
        test_loss = squared_loss(linreg(features_test, w, b), labels_test).mean().item()
        w_norm = torch.norm(w).item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        norms.append(w_norm)

# 绘制损失关于lambda的曲线
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, train_losses, label='训练损失')
plt.plot(lambda_values, test_losses, label='测试损失')
plt.xlabel('lambda')
plt.ylabel('损失')
plt.legend()
plt.show()

print("观察到当lambda增大时，训练损失逐渐增大，而测试损失先减小后增大，存在一个最佳的lambda使测试损失最小。")

# 2. 使用验证集来找到最佳值lambda。它真的是最优值吗？这有关系吗？

# 通常，我们可以将训练数据再划分出一部分作为验证集，用于选择最佳的lambda值。
# 由于样本量较小，这里仅作示意。

# 3. 如果我们使用sum_i |w_i|作为惩罚（L1正则化），更新方程会是什么样子？

# 对于L1正则化，损失函数中增加了lambda * sum_i |w_i|，其梯度为lambda * sign(w_i)
# 因此，更新方程为：w_i = w_i - lr * (dL/dw_i + lambda * sign(w_i))

# 4. 我们知道||w||^2 = w^T w。能找到类似的矩阵方程吗？

# 对于矩阵W，其Frobenius范数||W||_F^2 = trace(W^T W)

# 5. 除了权重衰减、增加训练数据、使用适当复杂度的模型之外，还能想出其他什么方法来处理过拟合？

# 答：可以使用 dropout、数据增强、早停（early stopping）、集成方法等来处理过拟合。

# 6. 在贝叶斯统计中，我们使用先验和似然的乘积，通过公式P(w|x) ∝ P(x|w)P(w)得到后验。如何得到带正则化的P(w)？

# 答：带正则化的P(w)可以被视为对参数w的先验分布。例如，L2正则化相当于对w施加零均值的高斯先验。

# 分割线
print("=" * 50)

# 总结
"""
在本代码示例中，我们使用了以下函数和方法：

1. generate_synthetic_data(w, b, num_examples)
   - 生成模拟数据集的函数，参数包括权重w、偏置b和样本数量num_examples。

2. load_array(data_arrays, batch_size, is_train=True)
   - 构造数据迭代器的函数，参数包括数据元组data_arrays、批量大小batch_size和是否打乱数据is_train。

3. linreg(X, w, b)
   - 定义线性回归模型的函数，参数包括输入特征X、权重w和偏置b。

4. squared_loss(y_hat, y)
   - 定义平方损失函数，参数包括预测值y_hat和真实值y。

5. sgd(params, lr, batch_size)
   - 实现小批量随机梯度下降优化算法的函数，参数包括模型参数params、学习率lr和批量大小batch_size。

6. l2_penalty(w)
   - 计算L2范数惩罚项的函数，参数是权重向量w。

7. train(lambd)
   - 训练模型的函数，参数是权重衰减系数lambd。

8. train_concise(wd)
   - 使用PyTorch内置优化器实现权重衰减的函数，参数是权重衰减系数wd。

这些函数展示了如何在PyTorch中从零开始和使用内置函数实现权重衰减，以解决过拟合问题。
"""

