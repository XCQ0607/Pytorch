# 3.3. 线性回归的简洁实现
print("3.3. 线性回归的简洁实现")

# 导入必要的库
import numpy as np
import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.nn import init

# 分割线
print("-" * 50)

# 生成数据集
print("生成数据集")

# 设置真实参数
true_w = torch.tensor([2, -3.4, 5.6])  # 真实权重
true_b = 4.2                          # 真实偏置

# 特征数量
num_features = len(true_w)

# 生成特征X和标签y
num_samples = 1000  # 样本数量

features = torch.randn(num_samples, num_features)  # 生成标准正态分布的特征，均值为0，标准差为1，形状为(num_samples, num_features)，也可normal(mean, std ,size)      生成均值为mean，标准差为std的正态分布，形状为size
labels = torch.matmul(features, true_w) + true_b   # 线性关系
labels += torch.randn(labels.shape) * 0.01         # 添加均值为0，标准差为0.01的噪声

# 打印生成的数据集的一部分
print("特征样本：", features[0:5])       #<tensor>[i:j,k:l] 表示从第i行到第j行，第k列到第l列的子矩阵,包含i，不包含j，包含k，不包含l
print("标签样本：", labels[0:5])

# 分割线
print("-" * 50)

# 读取数据集
print("读取数据集")

# 定义一个函数来加载数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。

    参数：
    data_arrays: 包含特征和标签的元组，例如 (features, labels)
    batch_size: 批量大小
    is_train: 是否在每个epoch中随机打乱数据，默认为True

    返回：
    PyTorch的数据迭代器
    """
    dataset = data.TensorDataset(*data_arrays)  # 创建数据集
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)  # 返回数据迭代器
    # data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train) 创建一个数据迭代器，dataset是数据集，batch_size是批量大小，shuffle=is_train表示是否在每个epoch中随机打乱数据
    #批量大小就是每次从数据集中取多少个样本进行训练

batch_size = 32  # 批量大小
data_iter = load_array((features, labels), batch_size)

# 读取并打印第一个小批量样本
for X, y in data_iter:
    print("小批量特征：", X)
    print("小批量标签：", y)
    break  # 只打印第一个小批量

# 分割线
print("-" * 50)

# 定义模型
print("定义模型")

# 定义一个包含多个全连接层的模型
net = nn.Sequential()

# nn.Sequential() 可以按顺序添加子模块，参数可以是OrderedDict或者多个子模块
# 这里我们使用add_module()方法逐个添加子模块

# 添加第一个线性层
net.add_module('linear1', nn.Linear(in_features=num_features, out_features=16, bias=True))  #add_module('名称', 子模块)
# 参数解释：
# in_features: 输入特征的大小
# out_features: 输出特征的大小 大小指的是特征的数量，也就是行数
# bias: 是否包含偏置项，默认为True

#Linearshi线性层

# 添加激活函数
net.add_module('relu1', nn.ReLU())
# # nn.ReLU() 是一种激活函数，不需要参数

# 添加第二个线性层
net.add_module('linear2', nn.Linear(16, 1))
# 输出为1，因为是回归问题，预测一个标量

# 查看模型结构
print("模型结构：", net)


# 分割线
print("-" * 50)

# 初始化模型参数
print("初始化模型参数")

# 使用自定义的初始化函数
def init_weights(m):
    """初始化模型参数。

    参数：
    m: 模型的层
    """
    if isinstance(m, nn.Linear):    # isinstance() 函数用于判断对象是否属于指定的类或其子类,参数：(类名, 类的父类)
        # 对权重使用正态分布初始化
        # init.normal_(tensor, mean=0.0, std=1.0)   tensor: 要初始化的张量，mean: 正态分布的均值，std: 正态分布的标准差
        init.normal_(m.weight.data, mean=0.0, std=0.01)
        # 对偏置使用常数初始化
        # init.constant_(tensor, val)   tensor: 要初始化的张量，val: 初始化的常数值
        init.constant_(m.bias.data, val=0.0)
#这个函数会对nn.Linear类型的层进行特殊处理，通过init.normal_方法对权重使用正态分布进行初始化，并通过init.constant_方法对偏置使用常数进行初始化。

net.apply(init_weights)  # 对网络中的所有模块应用初始化权重函数

# 打印初始化的参数
print("初始化的第一层权重：", net.linear1.weight.data)
print("初始化的第一层偏置：", net.linear1.bias.data)
print("初始化的第二层权重：", net.linear2.weight.data)
print("初始化的第二层偏置：", net.linear2.bias.data)

# 分割线
print("-" * 50)

# 定义损失函数
print("定义损失函数")

# 使用均方误差损失函数
# nn.MSELoss(reduction='mean')
# 参数解释：
# reduction: 指定应用于输出的归约方式，'mean'表示求平均，'sum'表示求和，'none'表示不归约
loss = nn.MSELoss(reduction='mean')

# 也可以使用其他损失函数，例如Huber损失函数
# nn.SmoothL1Loss(reduction='mean', beta=1.0)
# 参数解释：
# beta: 损失的阈值，默认值为1.0
# loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)

# 分割线
print("-" * 50)

# 定义优化算法
print("定义优化算法")

# 使用小批量随机梯度下降（SGD）
# optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
# 参数解释：
# params: 待优化的参数
# lr: 学习率
# momentum: 动量因子，默认为0
# weight_decay: 权重衰减（L2正则化），默认为0
# nesterov: 是否使用Nesterov动量，默认为False
learning_rate = 0.03
optimizer = optim.SGD(net.parameters(), lr=learning_rate)   # 使用SGD优化器，参数包括net.parameters()（待优化的参数）和学习率

# 也可以使用其他优化器，例如Adam
# optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# 参数解释：
# betas: 用于计算梯度和梯度平方的运行平均系数
# eps: 为了增加数值计算的稳定性而加到分母里的项

# 分割线
print("-" * 50)

# 训练模型
print("训练模型")

num_epochs = 5  # 迭代周期数
# 用于记录每个epoch的损失
loss_history = []

for epoch in range(num_epochs):
    for X, y in data_iter:
        optimizer.zero_grad()  # 梯度清零
        y_pred = net(X).squeeze()  # 前向传播，得到预测值，并将输出形状从 (batch_size, 1) 变为 (batch_size,)    squeeze() 用于去除形状为1的维度
        l = loss(y_pred, y)  # 计算损失
        l.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新参数
    l_epoch = loss(net(features).squeeze(), labels) # 计算整个数据集上的损失
    loss_history.append(l_epoch.item()) # 计算整个数据集上的损失
    print(f'epoch {epoch + 1}, loss {l_epoch:f}')

# 分割线
print("-" * 50)

# 查看训练结果
print("查看训练结果")

# 估计的参数
w_est = net.linear1.weight.data #net.linear1表示第一个线性层，weight表示权重，data表示取出数据
b_est = net.linear1.bias.data   #net.linear1.bias表示第一层的偏置，data表示取出数据

print('估计的第一层权重：\n', w_est)
print('估计的第一层偏置：\n', b_est)

# 因为有两层，需要综合考虑参数
# 这里我们只简单地打印最终的输出层参数

w_out = net.linear2.weight.data
b_out = net.linear2.bias.data

print('估计的输出层权重：\n', w_out)
print('估计的输出层偏置：\n', b_out)

# 分割线
print("-" * 50)

# 访问梯度
print("访问梯度")

# 打印权重和偏置的梯度
print('第一层权重的梯度：', net.linear1.weight.grad)
print('第一层偏置的梯度：', net.linear1.bias.grad)
print('输出层权重的梯度：', net.linear2.weight.grad)
print('输出层偏置的梯度：', net.linear2.bias.grad)

# 分割线
print("-" * 50)

# 绘制损失随epoch的变化
print("绘制损失曲线")

import matplotlib.pyplot as plt

plt.plot(range(1, num_epochs+1), loss_history)  # 绘制损失随epoch的变化    参数：range(1, num_epochs+1)表示从1到num_epochs的整数序列，loss_history表示每个epoch的损失
#横坐标范围为1到num_epochs，纵坐标为每个epoch的损失
#需要横坐标为整数：
plt.xticks(range(1, num_epochs+1))  # 设置横坐标刻度为整数
plt.xlabel('Epoch') # 横坐标为Epoch
plt.ylabel('Loss')  # 纵坐标为Loss
plt.title('Loss over Epochs')   # 标题为Loss over Epochs
plt.show()   # 显示图像

# 分割线
print("-" * 50)



# 总结
print("总结：")
print("本次代码示例中，我们使用PyTorch实现了线性回归的简洁实现，并扩展为包含一个隐藏层的神经网络。")
print("主要使用的函数和模块包括：")
print("1. nn.Sequential：用于按顺序构建模型，可以使用add_module()方法添加子模块")
print("2. nn.Linear：全连接层，参数包括in_features, out_features, bias等")
print("3. nn.ReLU：激活函数，不需要参数")
print("4. nn.MSELoss：均方误差损失函数，参数包括reduction")
print("5. optim.SGD：优化器，参数包括params, lr, momentum等")
print("6. data.DataLoader：数据迭代器，参数包括dataset, batch_size, shuffle等")
print("7. init.normal_：用于初始化参数，参数包括tensor, mean, std")
print("8. net.apply：用于对网络中的所有层应用初始化函数")
print("9. optimizer.zero_grad()：梯度清零")
print("10. loss.backward()：反向传播计算梯度")
print("11. optimizer.step()：更新参数")
print("12. matplotlib.pyplot：用于绘制损失曲线")


# 输入层、隐藏层、输出层和线性层在神经网络中各自扮演着不同的角色，它们之间的主要区别可以归纳如下：
# 位置和功能：
# 输入层：位于神经网络的最前端，负责接收来自外界的原始数据，并将其转化为神经网络可以处理的格式。输入层的神经元数量通常与输入数据的特征数量相匹配。
# 隐藏层：位于输入层和输出层之间，可以有一层或多层。它的主要作用是进行特征提取和转换，通过非线性激活函数对输入数据进行处理，以捕捉和表示数据中的复杂模式。
# 输出层：位于神经网络的最末端，负责将经过隐藏层处理后的数据转化为人类或其他系统可以理解的输出形式。输出层的神经元数量和激活函数的选择通常取决于具体的任务类型，如分类、回归等。
# 线性层与非线性层：
# 线性层：在神经网络中，线性层通常是指其输出是输入的线性组合的层，即不使用非线性激活函数或仅使用线性激活函数的层。线性层的输出可以表示为输入与权重矩阵的乘积加上偏置项。在某些情况下，如回归任务的输出层，可能会使用线性层。
# 非线性层：与线性层相对，非线性层在神经元中引入了非线性激活函数，如sigmoid、tanh、ReLU等。这些非线性函数使得神经网络能够学习和表示输入与输出之间的非线性关系。隐藏层通常都是非线性的，因为它们需要捕捉数据中的复杂特征。
# 结构和连接：
# 输入层和输出层通常具有固定的结构，即神经元的数量和连接方式是根据具体任务确定的。
# 隐藏层的结构则更加灵活，可以根据需要调整神经元的数量和层数，以及不同层之间的连接方式。这种灵活性使得神经网络能够适应各种不同的数据和任务。
# 综上所述，输入层、隐藏层、输出层和线性层在神经网络中的位置、功能、线性/非线性特性以及结构方面存在明显的区别。这些区别使得它们能够共同协作，实现神经网络对复杂数据的处理和学习能力。
#
# 只有隐藏层到输出层的层才可能是线性层。这是因为在隐藏层到输出层的层中，如果没有使用任何激活函数，那么该层的输出将是输入的线性组合，从而成为线性层。
# 然而，需要注意的是，线性层在神经网络中并不常见，因为它限制了网络能够表示和学习的复杂性。大多数神经网络在隐藏层中使用非线性激活函数，以便能够捕捉并表示输入与输出之间的非线性关系。线性层通常只在某些特定的场景或结构中使用，如线性回归问题或某些特定的神经网络结构。

'''
优化器的作用：
在机器学习中，优化器（如SGD、Adam等）的主要目的是更新模型参数，使得模型的损失函数（即模型预测值与真实值之间的差异）最小化。这通常通过计算损失函数对模型参数的梯度（即方向导数），并使用这些梯度来更新参数。

SGD（随机梯度下降）：
SGD是最早和最常用的优化器之一。在每次迭代中，SGD根据当前批次的损失函数梯度来更新模型参数。
作用：根据当前批次的梯度来更新模型参数。
参数：optimizer = optim.SGD(net.parameters(), lr=learning_rate)，其中net.parameters()返回模型的所有参数，lr是学习率，它决定了每次参数更新的步长。
使用场景：对于大规模数据集，SGD及其变种（如带动量的SGD）通常更高效。
区别：SGD更新是固定的，每次迭代都使用同样的学习率。这可能导致在接近最优解时，步长过大而错过最优解。

Adam（Adaptive Moment Estimation）：
Adam是一种自适应的学习率优化算法，它结合了梯度的一阶矩（均值）和二阶矩（未中心化的方差）来调整每个参数的学习率。
作用：自适应地调整每个参数的学习率，以更快地收敛到最优解。
参数：optimizer = torch.optim.Adam([w, b], lr=0.03)，其中[w, b]是模型的参数，lr是学习率，尽管在Adam中，这个学习率的影响通常比SGD中的小，因为它会自动调整每个参数的学习率。
使用场景：对于需要快速收敛且对超参数不敏感的任务，Adam通常是一个很好的选择。
区别：与SGD相比，Adam能够自动调整学习率，并在迭代过程中逐渐减少步长，这使得它在某些情况下更加高效和稳定。
总结：SGD和Adam都是常用的优化器，它们都可以用来最小化损失函数。SGD通过固定的学习率进行更新，而Adam通过自适应地调整每个参数的学习率来提高训练效率。选择哪种优化器取决于具体任务和数据集的特点。
'''