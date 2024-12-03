print("5.1. 层和块")

import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------
# 示例：自定义一个更复杂的块，包括多个层和非顺序连接

# 定义一个自定义块，包含多个层和复杂的前向传播逻辑
class ComplexBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ComplexBlock, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # 定义一个可学习的参数
        self.alpha = nn.Parameter(torch.rand(1))

    def forward(self, X):
        # 前向传播中包含控制流和非顺序操作
        out1 = F.relu(self.fc1(X))  #relu用于破坏线性性
        out2 = F.relu(self.fc2(out1))
        # 计算 out1 和 out2 的加权和
        out = self.alpha * out1 + (1 - self.alpha) * out2
        out = self.fc3(out)
        return out


"""
ComplexBlock 是一个自定义的神经网络模块（或称为“块”），它是 nn.Module 的子类。这个模块由三个全连接层（nn.Linear）和一个可学习的参数（alpha）组成。ComplexBlock 的设计允许它接收输入数据，并通过这些层进行前向传播，其中包含了控制流和非顺序操作。
具体来说，ComplexBlock 的结构如下：

输入层：接收输入数据 X。
第一个全连接层（fc1）：将输入数据映射到一个隐藏层空间，并通过 ReLU 激活函数。
第二个全连接层（fc2）：接收第一个全连接层的输出，并将其再次映射到隐藏层空间，同样通过 ReLU 激活函数。
加权和：计算第一个和第二个全连接层输出的加权和，权重由可学习的参数 alpha 和 (1 - alpha) 决定。
第三个全连接层（fc3）：将加权和的结果映射到输出空间。
ComplexBlock 的设计允许它在前向传播过程中进行更复杂的操作，比如通过可学习的 alpha 参数来控制两个不同路径（fc1 和 fc2）输出的贡献度。
在创建 ComplexBlock 的实例时，需要指定输入维度 input_dim、隐藏层维度 hidden_dim 和输出维度 output_dim。这些参数用于初始化全连接层的权重和偏置。
例如，要创建一个输入维度为 10，隐藏层维度为 20，输出维度为 1 的 ComplexBlock 实例，可以这样做：

block = ComplexBlock(input_dim=10, hidden_dim=20, output_dim=1)
然后，这个 block 实例就可以像其他 nn.Module 一样被用于构建和训练神经网络模型了。
"""


# 创建一个 ComplexBlock 的实例
input_dim = 20
hidden_dim = 64
output_dim = 10
block = ComplexBlock(input_dim, hidden_dim, output_dim)  # 实例化 ComplexBlock

# 生成一个随机输入
X = torch.rand(2, input_dim)    # 2 个样本，每个样本 20 维度

# 执行前向传播
output = block(X)
print("ComplexBlock 输出：", output)


# --------------------------------------------------
# 示例：实现一个函数，生成同一个块的多个实例，并构建更大的网络

def create_network(block_class, num_blocks, input_dim, hidden_dim, output_dim):
    """
    创建一个由多个相同块组成的网络

    参数：
    - block_class: 块的类，需要是 nn.Module 的子类。
    - num_blocks: 块的数量
    - input_dim: 输入维度
    - hidden_dim: 隐藏层维度
    - output_dim: 输出维度

    返回：
    - nn.Sequential 对象，包含多个块
    """
    layers = []  # 用于存储块的列表
    for i in range(num_blocks):
        # 对于第一个块，输入维度是给定的 input_dim
        # 对于后续的块，输入维度是 hidden_dim
        in_dim = input_dim if i == 0 else hidden_dim    # 输入维度
        #这行代码是Python语言中的条件表达式（也称为三元操作符）的一个例子。它用于根据条件来选择不同的值。具体来说，这行代码的意思是：
        # 如果变量 i 的值等于 0，那么变量 in_dim 将被赋值为 input_dim。
        # 如果变量 i 的值不等于 0，那么变量 in_dim 将被赋值为 hidden_dim。
        # 对于最后一个块，输出维度是给定的 output_dim
        out_dim = output_dim if i == num_blocks - 1 else hidden_dim    # 输出维度
        layers.append(block_class(in_dim, hidden_dim, out_dim)) # 添加块到列表
    # 使用 nn.Sequential 来组合所有块
    return nn.Sequential(*layers)   #*表示解包列表中的元素(解引用)


# 使用 create_network 函数创建一个网络，包含 3 个 ComplexBlock
num_blocks = 3
network = create_network(ComplexBlock, num_blocks, input_dim, hidden_dim, output_dim)
#ComplexBlock 是一个自定义的神经网络模块（或称为“块”）
#num_blocks = 3 表示要创建 3 个 ComplexBlock 实例。
#input_dim = 20 表示输入维度为 20。
#hidden_dim = 64 表示隐藏层维度为 64。
#output_dim = 10 表示输出维度为 10。


# 查看网络结构
print("网络结构：", network)

# 执行前向传播
output = network(X) # 执行前向传播
print("网络输出：", output)


# --------------------------------------------------
# 示例：展示在前向传播中使用控制流

class ControlFlowBlock(nn.Module):
    def __init__(self, input_dim):
        super(ControlFlowBlock, self).__init__()    # 调用父类的构造函数
        self.linear = nn.Linear(input_dim, input_dim)    # 线性层

    def forward(self, X):
        # 使用控制流，根据输入的 L1 范数决定前向传播的计算
        if X.norm().item() > 1:
            X = self.linear(X)
        else:
            X = X * 0.5
        return X


# 创建 ControlFlowBlock 的实例
control_block = ControlFlowBlock(input_dim)

# 执行前向传播
output = control_block(X)
print("ControlFlowBlock 输出：", output)

# --------------------------------------------------
# 总结：
# 本代码示例展示了如何自定义复杂的块，以及如何生成同一块的多个实例并构建更大的网络。
# 使用了以下函数和类：
# - nn.Module: 所有神经网络模块的基类。用于定义自定义的神经网络块。
# - nn.Linear(in_features, out_features, bias=True): 线性（全连接）层。
#   参数：
#   - in_features：输入的特征维度。
#   - out_features：输出的特征维度。
#   - bias：是否包含偏置项，默认值为 True。
# - nn.Parameter(data, requires_grad=True): 张量，表示可学习的参数。
#   参数：
#   - data：初始化参数的张量。
#   - requires_grad：是否计算梯度，默认值为 True。
# - torch.rand(*sizes): 返回一个张量，包含从均匀分布中抽取的随机数。
#   参数：
#   - sizes：张量的形状。
# - F.relu(input, inplace=False): ReLU 激活函数。
#   参数：
#   - input：输入张量。
#   - inplace：是否进行原地操作，默认值为 False。
# - nn.Sequential(*args): 顺序容器，用于将模块按顺序组合。
#   参数：
#   - *args：模块的实例。
# - torch.norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None): 计算张量的范数。
#   参数：
#   - input：输入张量。
#   - p：范数类型，默认值为 "fro"（Frobenius 范数）。
#   - dim：计算的维度。
#   - keepdim：是否保持维度。
# - torch.rand(形状): 生成指定形状的均匀分布随机张量。

# create_network 函数接受以下参数：
# - block_class: 块的类，需要是 nn.Module 的子类。
# - num_blocks: 要生成的块的数量。
# - input_dim: 网络的输入维度。
# - hidden_dim: 块的隐藏层维度。
# - output_dim: 网络的输出维度。

# 调用示例：
# network = create_network(ComplexBlock, 3, input_dim, hidden_dim, output_dim)
