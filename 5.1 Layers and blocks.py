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
        out1 = F.relu(self.fc1(X))
        out2 = F.relu(self.fc2(out1))
        # 计算 out1 和 out2 的加权和
        out = self.alpha * out1 + (1 - self.alpha) * out2
        out = self.fc3(out)
        return out


# 创建一个 ComplexBlock 的实例
input_dim = 20
hidden_dim = 64
output_dim = 10
block = ComplexBlock(input_dim, hidden_dim, output_dim)

# 生成一个随机输入
X = torch.rand(2, input_dim)

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
    layers = []
    for i in range(num_blocks):
        # 对于第一个块，输入维度是给定的 input_dim
        # 对于后续的块，输入维度是 hidden_dim
        in_dim = input_dim if i == 0 else hidden_dim
        # 对于最后一个块，输出维度是给定的 output_dim
        out_dim = output_dim if i == num_blocks - 1 else hidden_dim
        layers.append(block_class(in_dim, hidden_dim, out_dim))
    return nn.Sequential(*layers)


# 使用 create_network 函数创建一个网络，包含 3 个 ComplexBlock
num_blocks = 3
network = create_network(ComplexBlock, num_blocks, input_dim, hidden_dim, output_dim)

# 查看网络结构
print("网络结构：", network)

# 执行前向传播
output = network(X)
print("网络输出：", output)


# --------------------------------------------------
# 示例：展示在前向传播中使用控制流

class ControlFlowBlock(nn.Module):
    def __init__(self, input_dim):
        super(ControlFlowBlock, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)

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
