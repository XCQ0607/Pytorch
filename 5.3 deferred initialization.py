print("5.3. 延后初始化")

# 导入必要的库
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# 定义一个获取网络的函数，使用延后初始化的 nn.LazyLinear
def get_net():
    net = nn.Sequential()
    # 使用 nn.LazyLinear 时，不需要指定输入特征数
    net.add_module('layer1', nn.LazyLinear(256))  # 输出特征数为256
    net.add_module('relu1', nn.ReLU())
    net.add_module('layer2', nn.LazyLinear(10))   # 输出特征数为10
    return net

# 初始化网络
net = get_net()

# 打印网络结构
print("网络结构:")
print(net)

# 尝试访问参数（在前向传播之前），需要检查参数是否已初始化
print("\n初始化前的参数:")
for name, param in net.named_parameters():
    try:
        print(f"参数 {name} 的形状: {param.shape}")
    except RuntimeError:
        print(f"参数 {name} 还未初始化")

# -----------------------------------------------------------------------------
# 将数据通过网络以触发参数初始化
X = torch.randn(2, 20)  # 输入数据，形状为(2, 20)
output = net(X)

# 查看参数形状（在前向传播之后）
print("\n初始化后的参数:")
for name, param in net.named_parameters():
    print(f"参数 {name} 的形状: {param.shape}")

# -----------------------------------------------------------------------------
# 练习1: 如果指定了第一层的输入尺寸，但没有指定后续层的尺寸，会发生什么？是否立即进行初始化？

# 定义一个函数来获取一个网络，第一层指定输入维度
def get_net_with_input_dim():
    net = nn.Sequential()
    net.add_module('layer1', nn.Linear(20, 256))  # 指定输入特征数为20，输出为256
    net.add_module('relu1', nn.ReLU())
    net.add_module('layer2', nn.LazyLinear(10))   # 使用 LazyLinear 延后初始化
    return net

net = get_net_with_input_dim()

# 打印网络结构
print("\n网络结构（第一层指定输入维度）:")
print(net)

# 查看参数形状（在前向传播之前）
print("\n初始化前的参数:")
for name, param in net.named_parameters():
    try:
        print(f"参数 {name} 的形状: {param.shape}")
    except RuntimeError:
        print(f"参数 {name} 还未初始化")

# 将数据通过网络
X = torch.randn(2, 20)
output = net(X)

# 查看参数形状（在前向传播之后）
print("\n初始化后的参数:")
for name, param in net.named_parameters():
    print(f"参数 {name} 的形状: {param.shape}")

# 结论：第一层因指定了输入尺寸而立即初始化，后续层仍然延后初始化，直到数据通过网络。

# -----------------------------------------------------------------------------
# 练习2: 如果指定了不匹配的维度会发生什么？

# 定义一个函数来获取一个网络，指定错误的维度
def get_net_with_wrong_dims():
    net = nn.Sequential()
    net.add_module('layer1', nn.Linear(20, 256))  # 输入20，输出256
    net.add_module('relu1', nn.ReLU())
    net.add_module('layer2', nn.Linear(128, 10))   # 输入128，输出10（输入维度错误）
    return net

net = get_net_with_wrong_dims()

# 打印网络结构
print("\n网络结构（指定错误的维度）:")
print(net)

# 查看参数形状
print("\n初始化的参数:")
for name, param in net.named_parameters():
    print(f"参数 {name} 的形状: {param.shape}")

# 将数据通过网络
X = torch.randn(2, 20)
try:
    output = net(X)
except Exception as e:
    print(f"\n发生错误: {e}")

# 结论：当维度不匹配时，会在数据通过网络时引发运行时错误。

# -----------------------------------------------------------------------------
# 练习3: 如果输入具有不同的维度，需要做什么？提示：查看参数绑定的相关内容。

# 第一次输入数据，尺寸为(2, 20)
net = get_net()  # 使用延后初始化的网络
X1 = torch.randn(2, 20)
output1 = net(X1)
print("\n第一次输入的输出:")
print(output1)

# 查看参数形状
print("\n参数形状:")
for name, param in net.named_parameters():
    print(f"参数 {name} 的形状: {param.shape}")

# 尝试使用不同尺寸的输入
X2 = torch.randn(2, 30)
try:
    output2 = net(X2)
except Exception as e:
    print(f"\n发生错误: {e}")

# 解决方法：需要重新初始化网络或设计能处理可变输入尺寸的网络。

# -----------------------------------------------------------------------------
# 重新初始化网络以处理新尺寸的输入
net = get_net()  # 重新获取一个新的网络
X2 = torch.randn(2, 30)
output2 = net(X2)
print("\n重新初始化后输入的输出:")
print(output2)

# 查看参数形状
print("\n参数形状:")
for name, param in net.named_parameters():
    print(f"参数 {name} 的形状: {param.shape}")

# 结论：对于不同的输入尺寸，需要重新初始化网络或采用能处理可变输入的模型设计。

# -----------------------------------------------------------------------------
# 总结：
# 在上述代码示例中，我们使用了以下函数和方法：

# 1. nn.Sequential(*args): 序列容器，按照添加的顺序将模块添加到一起，前向传播时依次通过各个模块。
#    - *args: 可变长度参数列表，包含各个子模块。

# 2. nn.LazyLinear(out_features, bias=True): 延后初始化的线性层，不需要指定输入特征数。
#    - out_features: 输出特征数。
#    - bias: 是否包含偏置项，默认 True。

# 3. nn.Linear(in_features, out_features, bias=True): 线性层，需要指定输入和输出特征数。
#    - in_features: 输入特征数。
#    - out_features: 输出特征数。
#    - bias: 是否包含偏置项，默认 True。

# 4. net.named_parameters(): 返回网络中所有参数的迭代器，包括名称和参数本身。

# 5. torch.randn(*sizes): 返回一个张量，包含标准正态分布的随机数。
#    - *sizes: 张量的形状。

# 注意：在使用 nn.LazyLinear 时，必须在前向传播之后才能访问参数的形状，否则会出现未初始化的错误。

