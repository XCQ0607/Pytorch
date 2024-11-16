print("3.7. softmax回归的简洁实现")

# 导入必要的库
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
from torch.nn import functional as F

# 分割线
print("--------------------------------------------------")
print("数据加载和预处理")

# 定义超参数
batch_size = 256  # 批量大小
num_epochs = 20   # 迭代周期数
learning_rate = 0.1  # 学习率

# 定义数据预处理方式
transform = transforms.Compose([
    transforms.ToTensor()
])

# 下载并加载训练数据集
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, transform=transform, download=True)

# 下载并加载测试数据集
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, transform=transform, download=True)

# 创建数据加载器
train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 分割线
print("--------------------------------------------------")
print("模型定义与初始化")

# 定义模型
net = nn.Sequential(        #Sequential是一个有序的容器，神经网络模块将按照在传入Sequential的顺序依次被添加到计算图中执行
    nn.Flatten(),          # 展平输入
    nn.Linear(784, 10)     # 全连接层，输入784维，输出10维  784=28*28    10个类别
)

# 初始化模型参数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 用正态分布初始化权重
        nn.init.zeros_(m.bias)                       # 将偏置初始化为0

net.apply(init_weights)  # 应用初始化函数

# 在这段代码中，net.apply(init_weights) 的作用是对模型 net 中的每个子模块应用 init_weights 函数来进行参数初始化。让我们逐步解析代码中的各个部分：
# 1. 模型定义：
# net = nn.Sequential(
#     nn.Flatten(),          # 展平输入
#     nn.Linear(784, 10)     # 全连接层，输入784维，输出10维
# )
# nn.Sequential：这是一个有序的容器，用来将神经网络的各个层按照给定的顺序串联起来。神经网络的输入数据会按照这些层的顺序依次流经网络中的每一层。
# nn.Flatten()：该层将输入的多维数据展平为一维。在这里，输入是一个 28x28 的二维图像（784 个特征），通过 Flatten 后变成一个包含 784 个元素的一维张量。
# nn.Linear(784, 10)：这是一个全连接层（也叫线性层），输入大小为 784（展平后的图像像素数），输出大小为 10，表示模型最终会输出 10 个类别的预测结果。
#
# 2. init_weights 函数：
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0, std=0.01)  # 用正态分布初始化权重
#         nn.init.zeros_(m.bias)                       # 将偏置初始化为0
# init_weights(m) 是一个自定义的初始化函数，接受一个模块 m 作为输入。
# if isinstance(m, nn.Linear):：这行代码检查输入的模块 m 是否是一个 nn.Linear 层，即判断该模块是否为全连接层。如果是 nn.Linear 层，那么对其权重和偏置进行初始化。
# nn.init.normal_(m.weight, mean=0, std=0.01)：使用正态分布（均值为 0，标准差为 0.01）初始化权重。
# nn.init.zeros_(m.bias)：将该层的偏置初始化为 0。
#
# 3. net.apply(init_weights) 的作用：
# net.apply(init_weights)
# apply 是 PyTorch 中 nn.Module 类的一个方法。它会遍历 net 模型中的所有子模块，并对每个子模块调用 init_weights 函数。
# 由于 net 是一个 nn.Sequential 容器，net.apply(init_weights) 会依次遍历 net 中的每个子层：首先是 nn.Flatten()，然后是 nn.Linear(784, 10)。但是，在 init_weights 函数中，只有 nn.Linear 层会被初始化。对于 nn.Flatten()，它不是一个可训练的层，因此不会进行初始化。
# 在这个例子中，net.apply(init_weights) 主要会对 nn.Linear(784, 10) 层的权重进行初始化。具体来说，它会：
# 使用正态分布（均值为 0，标准差为 0.01）初始化 Linear 层的权重。
# 将 Linear 层的偏置初始化为 0。
# 总结：
# net.apply(init_weights) 会对 net 中的每个子模块（在这里主要是 nn.Linear 层）应用 init_weights 函数进行初始化。在这个例子中，init_weights 函数会初始化 Linear 层的权重为正态分布（均值 0，标准差 0.01），并将偏置初始化为 0。

# 分割线
print("--------------------------------------------------")
print("损失函数定义")

# 定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()

# 分割线
print("--------------------------------------------------")
print("优化器定义")

# 定义优化器
# 定义优化器，使用随机梯度下降（SGD）优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 分割线
print("--------------------------------------------------")
print("模型训练")

# 定义训练函数
def train(net, train_loader, test_loader, loss_fn, num_epochs, optimizer):
    for epoch in range(num_epochs):
        net.train()  # 设置模型为训练模式
        for X, y in train_loader:
            optimizer.zero_grad()        # 梯度清零
            y_hat = net(X)               # 前向传播
            loss = loss_fn(y_hat, y)     # 计算损失
            loss.backward()              # 反向传播
            optimizer.step()             # 更新参数
        # 在每个epoch后评估模型
        test_accuracy = evaluate_accuracy(net, test_loader)
        print(f'第{epoch+1}轮，测试集准确率：{test_accuracy:.4f}')

# 定义模型评估函数
def evaluate_accuracy(net, data_loader):
    net.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for X, y in data_loader:
            y_hat = net(X)
            predictions = y_hat.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
    return correct / total

# 开始训练
train(net, train_loader, test_loader, loss_fn, num_epochs, optimizer)

# 分割线
print("--------------------------------------------------")
print("超参数调整实验")

# 试验不同的学习率
learning_rates = [0.01, 0.1, 0.5]
for lr in learning_rates:
    print(f"\n使用学习率：{lr}")
    # 重新初始化模型和优化器
    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train(net, train_loader, test_loader, loss_fn, num_epochs=5, optimizer=optimizer)

# 试验不同的批量大小
batch_sizes = [64, 256, 512]
for bs in batch_sizes:
    print(f"\n使用批量大小：{bs}")
    # 重新定义数据加载器
    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=bs, shuffle=False)
    # 重新初始化模型和优化器
    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    train(net, train_loader, test_loader, loss_fn, num_epochs=5, optimizer=optimizer)

# 分割线
print("--------------------------------------------------")
print("增加迭代周期数，观察测试准确率下降")

# 训练50个迭代周期
num_epochs_long = 50
net.apply(init_weights)  # 重新初始化模型
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  # 重新定义优化器,learning_rate=0.1
train(net, train_loader, test_loader, loss_fn, num_epochs=num_epochs_long, optimizer=optimizer)

# 分割线
print("--------------------------------------------------")
print("引入正则化方法（如权重衰减）解决过拟合问题")
#过拟合：在训练集上表现很好，但在测试集上表现很差

# 定义带权重衰减的优化器
optimizer_wd = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)  # weight_decay=0.001是指L2正则化的系数，用于控制权重的衰减程度，防止权重过大

# 重新初始化模型
net.apply(init_weights)

# 训练50个迭代周期，使用权重衰减
train(net, train_loader, test_loader, loss_fn, num_epochs=num_epochs_long, optimizer=optimizer_wd)

# 分割线
print("--------------------------------------------------")
print("总结与回答问题")

'''
总结：
- 使用了torch.utils.data.DataLoader来加载数据，主要参数有：
    - dataset：数据集
    - batch_size：批量大小
    - shuffle：是否打乱数据
- 使用了nn.Sequential来构建模型，可以按顺序添加层
- 使用了nn.Linear定义全连接层，参数有：
    - in_features：输入特征数
    - out_features：输出特征数
    - bias：是否包含偏置，默认True
- 使用了nn.CrossEntropyLoss作为损失函数
- 使用了torch.optim.SGD作为优化器，参数有：
    - params：待优化参数
    - lr：学习率
    - weight_decay：权重衰减系数，用于正则化
- 在训练过程中，观察到增加迭代周期数后，测试准确率开始下降，这是由于过拟合造成的。解决方法包括：
    - 使用正则化方法，如权重衰减（L2正则化）
    - 使用早停法，监控验证集性能，在性能下降前停止训练
    - 增加数据集，或使用数据增强技术

'''

# 练习答案：
# 当增加迭代周期数时，模型在训练集上可能表现得越来越好，但在测试集上准确率可能会下降，这是因为模型开始过拟合训练数据。为了解决这个问题，可以使用正则化方法（如权重衰减），或者在模型开始过拟合前停止训练（早停法）。

