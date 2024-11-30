print("5.2. 参数管理")

import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------
# 示例：创建一个具有单隐藏层的多层感知机并访问参数

# 定义一个简单的多层感知机模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.hidden = nn.Linear(4, 8)   # 隐藏层，输入维度4，输出维度8
        self.output = nn.Linear(8, 1)   # 输出层，输入维度8，输出维度1

    def forward(self, X):
        X = F.relu(self.hidden(X))
        X = self.output(X)
        return X

# 初始化模型
net = SimpleMLP()

# 生成随机输入
X = torch.rand(2, 4)
print("输入X：", X)

# 执行前向传播
output = net(X)
print("网络输出：", output)

# --------------------------------------------------
# 访问参数

print("\n访问网络的参数：")
for name, param in net.named_parameters():
    print(f"参数名称：{name}, 参数形状：{param.shape}")

# 访问特定层的参数
print("\n访问特定层的参数：")
print("隐藏层权重：", net.hidden.weight)
print("隐藏层偏置：", net.hidden.bias)

# --------------------------------------------------
# 一次性访问所有参数

print("\n一次性访问所有参数：")
params = list(net.parameters())
for i, param in enumerate(params):
    print(f"参数 {i}: 形状 {param.shape}")

# --------------------------------------------------
# 从嵌套块中收集参数

print("\n从嵌套块中收集参数：")

# 定义嵌套块
def block1():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU()
    )

def block2():
    net = nn.Sequential()
    for _ in range(2):
        net.add_module('block1', block1())
    return net

class NestedMLP(nn.Module):
    def __init__(self):
        super(NestedMLP, self).__init__()
        self.net = nn.Sequential(
            block2(),
            nn.Linear(4, 1)
        )

    def forward(self, X):
        return self.net(X)

nested_net = NestedMLP()
output = nested_net(X)
print("嵌套网络输出：", output)

print("\n嵌套网络的参数：")
for name, param in nested_net.named_parameters():
    print(f"参数名称：{name}, 参数形状：{param.shape}")

# --------------------------------------------------
# 参数初始化

print("\n参数初始化：")

# 内置初始化方法
def init_weights_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=0.01)  # 正态分布初始化权重
        nn.init.zeros_(m.bias)  # 将偏置初始化为0

net.apply(init_weights_normal)
print("初始化后的隐藏层权重：", net.hidden.weight)

# 自定义初始化方法
print("\n自定义初始化：")

class CustomInit(nn.Module):
    def __init__(self):
        super(CustomInit, self).__init__()

    def forward(self, m):
        if type(m) == nn.Linear:
            print(f"正在初始化层：{m}")
            nn.init.uniform_(m.weight, a=-10, b=10)
            m.weight.data *= (torch.abs(m.weight.data) >= 5)
            nn.init.zeros_(m.bias)

net.apply(CustomInit())
print("自定义初始化后的隐藏层权重：", net.hidden.weight)

# --------------------------------------------------
# 参数绑定（共享参数）

print("\n参数绑定（共享参数）：")

# 定义共享层
shared_layer = nn.Linear(8, 8)

# 定义使用共享参数的网络
class SharedParamsNet(nn.Module):
    def __init__(self):
        super(SharedParamsNet, self).__init__()
        self.layer1 = nn.Linear(4, 8)
        self.shared1 = shared_layer
        self.layer2 = nn.Linear(8, 8)
        self.shared2 = shared_layer  # 共享参数
        self.output = nn.Linear(8, 1)

    def forward(self, X):
        X = F.relu(self.layer1(X))
        X = F.relu(self.shared1(X))
        X = F.relu(self.layer2(X))
        X = F.relu(self.shared2(X))
        X = self.output(X)
        return X

shared_net = SharedParamsNet()
output = shared_net(X)
print("使用共享参数的网络输出：", output)

# 验证共享参数是否相同
print("shared1 和 shared2 的权重是否相同：", shared_net.shared1.weight.data_ptr() == shared_net.shared2.weight.data_ptr())

# 修改共享参数，验证是否同步更新
shared_net.shared1.weight.data[0, 0] = 100
print("修改后的共享层权重第一行：", shared_net.shared2.weight.data[0])

# --------------------------------------------------
# 练习

print("\n练习：")

# 1. 使用 5.1 节中定义的 FancyMLP 模型，访问各个层的参数。

class FancyMLP(nn.Module):
    def __init__(self):
        super(FancyMLP, self).__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 常数参数
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X = X / 2
        return X.sum()

fancy_net = FancyMLP()
X_fancy = torch.rand(2, 20)
output = fancy_net(X_fancy)
print("FancyMLP 网络输出：", output)

# 访问各个层的参数
print("\nFancyMLP 网络的参数：")
for name, param in fancy_net.named_parameters():
    print(f"参数名称：{name}, 参数形状：{param.shape}")

# 2. 查看初始化模块文档以了解不同的初始化方法。

print("\n常用的初始化方法包括：")
print("- nn.init.normal_: 正态分布初始化")
print("- nn.init.uniform_: 均匀分布初始化")
print("- nn.init.constant_: 常数初始化")
print("- nn.init.xavier_uniform_: Xavier 均匀分布初始化")
print("- nn.init.xavier_normal_: Xavier 正态分布初始化")
print("- nn.init.kaiming_uniform_: Kaiming 均匀分布初始化")
print("- nn.init.kaiming_normal_: Kaiming 正态分布初始化")

# 3. 构建包含共享参数层的多层感知机并对其进行训练。在训练过程中，观察模型各层的参数和梯度。

print("\n构建包含共享参数的多层感知机并训练：")

# 简单的数据集
X_train = torch.randn(100, 4)
y_train = torch.randn(100, 1)

# 定义模型
class SharedMLP(nn.Module):
    def __init__(self):
        super(SharedMLP, self).__init__()
        self.shared_layer = nn.Linear(4, 4)
        self.fc = nn.Linear(4, 1)

    def forward(self, X):
        X = F.relu(self.shared_layer(X))
        X = self.shared_layer(X)  # 共享层重复使用
        X = self.fc(X)
        return X

model = SharedMLP()

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(3):
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 查看共享层的参数和梯度
print("\n共享层的权重：", model.shared_layer.weight)
print("共享层的权重梯度：", model.shared_layer.weight.grad)

# 4. 为什么共享参数是个好主意？

print("\n参数共享的好处：")
print("- 减少模型参数数量，降低过拟合风险")
print("- 节省内存，提高计算效率")
print("- 在某些结构中（如循环神经网络），参数共享是必要的")
print("- 捕获输入之间的模式，提高模型泛化能力")

# --------------------------------------------------
# 总结：

# 本示例演示了参数管理的多个方面，包括访问参数、参数初始化、自定义初始化、参数共享等。

# 主要使用的函数和类：

# - nn.Module: 所有神经网络模块的基类，用于自定义模型。
# - nn.Linear(in_features, out_features, bias=True): 线性层。
#   - 参数：
#     - in_features: 输入特征维度。
#     - out_features: 输出特征维度。
#     - bias: 是否包含偏置项，默认为 True。
# - nn.init.normal_(tensor, mean=0, std=1): 正态分布初始化张量。
#   - 参数：
#     - tensor: 要初始化的张量。
#     - mean: 均值。
#     - std: 标准差。
# - nn.init.uniform_(tensor, a=0, b=1): 均匀分布初始化张量。
#   - 参数：
#     - tensor: 要初始化的张量。
#     - a: 均匀分布下界。
#     - b: 均匀分布上界。
# - nn.init.constant_(tensor, val): 将张量初始化为常数。
#   - 参数：
#     - tensor: 要初始化的张量。
#     - val: 常数值。
# - torch.mm(input, mat2): 矩阵乘法。
#   - 参数：
#     - input: 张量，形状 (n, m)。
#     - mat2: 张量，形状 (m, p)。
# - torch.rand(*sizes): 生成均匀分布的随机张量。
#   - 参数：
#     - sizes: 张量的形状。
# - torch.randn(*sizes): 生成标准正态分布的随机张量。
#   - 参数：
#     - sizes: 张量的形状。
# - torch.optim.SGD(params, lr): 随机梯度下降优化器。
#   - 参数：
#     - params: 待优化的参数。
#     - lr: 学习率。

# 调用示例：

# - 定义模型：
#   net = SimpleMLP()
# - 访问参数：
#   for name, param in net.named_parameters():
#       print(name, param.shape)
# - 参数初始化：
#   net.apply(init_weights_normal)
# - 自定义初始化：
#   net.apply(CustomInit())
# - 参数共享：
#   shared_layer = nn.Linear(8, 8)
#   在多个层中使用 shared_layer

