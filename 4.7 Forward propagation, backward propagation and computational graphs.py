print("4.7. 前向传播、反向传播和计算图")

import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型的输入和输出维度
input_size = 784  # 输入层维度，例如28x28的图像展开为784
hidden_size = 256  # 隐藏层神经元数量
output_size = 10  # 输出层维度，例如10分类问题


# 定义模型，包括输入层、隐藏层（添加了偏置项）、输出层
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        # 定义隐藏层，使用线性变换，添加偏置项
        self.hidden = nn.Linear(input_size, hidden_size, bias=True)
        # 定义激活函数，使用ReLU
        self.activation = nn.ReLU()
        # 定义输出层，使用线性变换，添加偏置项
        self.output = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        """
        前向传播函数，定义了数据如何通过网络。

        参数：
        x (torch.Tensor): 输入张量，形状为(batch_size, input_size)。

        返回：
        x (torch.Tensor): 输出张量，形状为(batch_size, output_size)。
        """
        # 计算隐藏层线性变换
        z = self.hidden(x)
        # 计算激活函数
        h = self.activation(z)
        # 计算输出层线性变换
        o = self.output(h)
        return o


# 创建模型实例
model = SimpleNeuralNetwork(input_size, hidden_size, output_size)

# 定义损失函数，使用交叉熵损失
criterion = nn.CrossEntropyLoss()

# 定义优化器，使用随机梯度下降，并添加L2正则化（权重衰减）
learning_rate = 0.01
lambda_l2 = 0.001  # L2正则化的权重
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)

# 模拟一个输入和标签
batch_size = 64
# 随机生成输入数据，形状为(batch_size, input_size)
inputs = torch.randn(batch_size, input_size)
# 随机生成标签，形状为(batch_size)
labels = torch.randint(0, output_size, (batch_size,))

# 前向传播
outputs = model(inputs)
# 计算损失
loss = criterion(outputs, labels)

# 反向传播
optimizer.zero_grad()  # 清零梯度
loss.backward()  # 计算梯度
optimizer.step()  # 更新参数

print("模型训练了一次迭代。")

# 打印分割线
print("-" * 50)

# 以下是针对练习的回答和代码示例

# 练习1：假设一些标量函数f的输入X是n x m矩阵。f相对于X的梯度维数是多少？
print("练习1：梯度维数计算")
n, m = 5, 3
X = torch.randn(n, m, requires_grad=True)
# 定义一个标量函数f
f = torch.sum(X ** 2)
# 计算梯度
f.backward()
# X的梯度维度与X相同
print(f"X的梯度维度：{X.grad.size()}")  # 输出：torch.Size([5, 3])

# 打印分割线
print("-" * 50)

# 练习2：向模型的隐藏层添加偏置项（已在上面的模型中完成）
print("练习2：添加偏置项的模型已经定义。")

# 绘制计算图和推导前向、反向传播方程（这里用注释表示）

# 前向传播方程：
# z = W1 * x + b1
# h = φ(z)
# o = W2 * h + b2
# L = l(o, y)
# s = λ/2 * (||W1||^2 + ||W2||^2)
# J = L + s

# 反向传播方程：
# 计算 ∂J/∂o = ∂L/∂o
# 计算 ∂J/∂W2 = ∂J/∂o * h^T + λW2
# 计算 ∂J/∂h = W2^T * ∂J/∂o
# 计算 ∂J/∂z = ∂J/∂h ⊙ φ'(z)
# 计算 ∂J/∂W1 = ∂J/∂z * x^T + λW1
# 其中⊙表示元素乘

# 打印分割线
print("-" * 50)

# 练习3：计算模型用于训练和预测的内存占用

print("练习3：内存占用计算")
# 计算参数数量
num_params = sum(p.numel() for p in model.parameters())
print(f"模型参数总数：{num_params}")

# 计算训练时的内存占用（参数 + 中间激活值）
# 假设每个参数和激活值都是32位浮点数（4字节）
memory_params = num_params * 4 / (1024 ** 2)  # 转换为MB
# 中间激活值包括z、h、o，大小与相应的层输出维度有关
memory_activations = batch_size * (hidden_size + hidden_size + output_size) * 4 / (1024 ** 2)
total_memory_training = memory_params + memory_activations
print(f"训练时内存占用约为：{total_memory_training:.2f} MB")

# 预测时不需要存储梯度，中间激活值也可能不需要全部保留
total_memory_inference = memory_params
print(f"预测时内存占用约为：{total_memory_inference:.2f} MB")

# 打印分割线
print("-" * 50)

# 练习4：计算二阶导数对计算图的影响和时间消耗

print("练习4：计算二阶导数")
# 在PyTorch中，计算二阶导数需要设置create_graph=True
# 这会导致计算图的构建更加复杂，消耗更多的内存和计算时间

# 示例：计算参数的二阶导数
loss = criterion(model(inputs), labels)
first_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
# 计算二阶导数
second_grad = []
for grad in first_grad:
    grad2 = torch.autograd.grad(grad.sum(), model.parameters(), retain_graph=True)
    second_grad.append(grad2)

print("已计算二阶导数。由于需要保留计算图，计算时间和内存占用显著增加。")

# 打印分割线
print("-" * 50)

# 练习5：将计算图划分到多个GPU上

print("练习5：多GPU计算")
if torch.cuda.device_count() > 1:
    print(f"检测到 {torch.cuda.device_count()} 个GPU，尝试进行模型并行。")
    # 将模型的不同部分放到不同的GPU上
    model.hidden.to('cuda:0')
    model.output.to('cuda:1')

    # 定义输入和移动到第一个GPU
    inputs = inputs.to('cuda:0')

    # 前向传播
    z = model.hidden(inputs)
    z = z.to('cuda:1')  # 将中间结果移动到第二个GPU
    h = model.activation(z)
    o = model.output(h)

    # 计算损失和反向传播
    labels = labels.to('cuda:1')
    loss = criterion(o, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("已在多个GPU上完成一次前向和反向传播。")
else:
    print("GPU数量不足，无法演示多GPU并行。")

# 优点：可以处理更大的模型和批量，减轻单个GPU的内存压力。
# 缺点：需要在GPU之间传输数据，增加了通信开销，可能降低计算效率。

# 打印分割线
print("-" * 50)

# 总结
print("总结：在本示例中，我们构建了一个包含偏置项的单隐藏层神经网络，并演示了前向传播和反向传播的过程。我们还回答了练习中的问题，包括计算梯度维度、推导方程、计算内存占用、讨论二阶导数对计算图的影响，以及如何在多GPU环境下进行模型并行。")


在以上代码示例中，我们使用了以下主要函数和类：

torch.nn.Module：所有神经网络模块的基类。自定义的神经网络需要继承它。
参数：
self：实例本身。
示例：
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, ...):
        super(SimpleNeuralNetwork, self).__init__()
        ...

# torch.nn.Linear：用于设置全连接层的模块。
# 参数：
# in_features：输入特征维度。
# out_features：输出特征维度。
# bias：是否包含偏置项，默认为True。
# 示例：
# self.hidden = nn.Linear(input_size, hidden_size, bias=True)
#
# torch.nn.ReLU：ReLU激活函数。
# 示例：
# self.activation = nn.ReLU()
#
# torch.optim.SGD：随机梯度下降优化器。
# 参数：
# params：待优化的参数。
# lr：学习率。
# weight_decay：权重衰减系数，即L2正则化项的λ。
# 示例：
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)
#
# torch.nn.CrossEntropyLoss：交叉熵损失函数，用于多分类问题。
# 示例：
# criterion = nn.CrossEntropyLoss()
#
# torch.autograd.grad：用于计算梯度。
# 参数：
# outputs：计算图的输出张量。
# inputs：需要计算梯度的输入张量。
# create_graph：是否构建用于计算高阶导数的计算图。
# retain_graph：是否保留计算图，以便多次使用。
# 示例：
# grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
