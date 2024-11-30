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
        super(SimpleNeuralNetwork, self).__init__() #调用父类的构造函数，这里显式传入self对象，并传入当前类的名称
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
model = SimpleNeuralNetwork(input_size, hidden_size, output_size)   #

# 定义损失函数，使用交叉熵损失
criterion = nn.CrossEntropyLoss()

# 定义优化器，使用随机梯度下降，并添加L2正则化（权重衰减）
learning_rate = 0.01
lambda_l2 = 0.001  # L2正则化的权重
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)

# 模拟一个输入和标签
batch_size = 64 # 批量大小
# 随机生成输入数据，形状为(batch_size, input_size)
inputs = torch.randn(batch_size, input_size)
# 随机生成标签，形状为(batch_size)
labels = torch.randint(0, output_size, (batch_size,))   #randint函数返回一个形状为(batch_size)的张量，张量中的元素是0到output_size-1之间的随机整数

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
f.backward()    #计算的偏导数，上面有requires_grad=True，计算f中的每个元素对f的偏导数
# X的梯度维度与X相同
print(f"X的梯度维度：{X.grad.size()}")  # 输出：torch.Size([5, 3])

# 打印分割线
print("-" * 50)

# 练习2：向模型的隐藏层添加偏置项（已在上面的模型中完成）
print("练习2：添加偏置项的模型已经定义。")

# 绘制计算图和推导前向、反向传播方程（这里用注释表示）

# 前向传播方程：
# z = W1 * x + b1
# h = φ(z)  #φ是激活函数
# o = W2 * h + b2
# L = l(o, y)   #l是损失函数
# s = λ/2 * (||W1||^2 + ||W2||^2)   #||w1||指的是w1的L2范数
#s是L2正则化项，λ是正则化系数
#J是目标函数，L是损失函数，s是L2正则化项

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
num_params = sum(p.numel() for p in model.parameters()) #p.numel()返回参数p中的元素个数,model.parameters()返回模型中的所有参数
print(f"模型参数总数：{num_params}")

# 计算训练时的内存占用（参数 + 中间激活值）
# 假设每个参数和激活值都是32位浮点数（4字节）
memory_params = num_params * 4 / (1024 ** 2)  # 转换为MB
# 中间激活值包括z、h、o，大小与相应的层输出维度有关
memory_activations = batch_size * (hidden_size + hidden_size + output_size) * 4 / (1024 ** 2)   #其实上面只有一个隐藏层
total_memory_training = memory_params + memory_activations
print(f"训练时内存占用约为：{total_memory_training:.2f} MB")

#两个隐藏层写法：
# class SimpleNeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleNeuralNetwork, self).__init__()
#         # 定义第一个隐藏层
#         self.hidden1 = nn.Linear(input_size, hidden_size, bias=True)
#         # 定义激活函数
#         self.activation1 = nn.ReLU()
#         # 定义第二个隐藏层
#         self.hidden2 = nn.Linear(hidden_size, hidden_size, bias=True)
#         # 定义激活函数
#         self.activation2 = nn.ReLU()
#         # 定义输出层
#         self.output = nn.Linear(hidden_size, output_size, bias=True)
#     def forward(self, x):
#         # 前向传播通过第一个隐藏层
#         x = self.activation1(self.hidden1(x))
#         # 前向传播通过第二个隐藏层
#         x = self.activation2(self.hidden2(x))
#         # 前向传播通过输出层
#         output = self.output(x)
#         return output

# 你可以根据需要扩展你的神经网络模型，添加更多的隐藏层。但是，你需要注意以下几点：
# 性能考虑：随着隐藏层数量的增加，模型的复杂性和计算需求也会增加。这可能导致训练时间变长，并且需要更多的计算资源。
# 梯度消失/爆炸：在深度神经网络中，随着层数的增加，梯度在反向传播过程中可能会逐渐消失（变得非常小）或爆炸（变得非常大），这会影响模型的训练效果。为了解决这个问题，可能需要使用特殊的初始化方法、激活函数（如ReLU、LeakyReLU等）或优化技术（如梯度裁剪、残差连接等）。
# 过拟合风险：增加模型的复杂性（如添加更多的隐藏层）可能会增加过拟合的风险，即模型在训练数据上表现良好，但在测试数据上表现不佳。为了缓解过拟合，可以使用正则化技术（如L1/L2正则化、dropout等）或增加训练数据量。

# batch_size 是指每批次处理的数据样本数量。
# hidden_size 通常代表隐藏层的单元数或神经元的数量。在这个表达式中，它出现了两次，这可能意味着在计算中有两个隐藏层，且这两个隐藏层的大小（即神经元的数量）是相同的。当然，具体是否如此需要查看完整的网络结构和上下文来确定。
# output_size 是指输出层的单元数或神经元的数量。
# 将这些值相加 (hidden_size + hidden_size + output_size)，我们得到了一个批次中所有样本在所有相关层（在这里是两个隐藏层和一个输出层）中的激活值总数。然后，乘以 batch_size，我们得到了一批数据中所有激活值的总数。

# 预测时不需要存储梯度，中间激活值也可能不需要全部保留
total_memory_inference = memory_params
print(f"预测时内存占用约为：{total_memory_inference:.2f} MB")
# 模型预测时内存占用等于模型参数总数乘以4除以1024^2的原因在于，预测时主要关注的是模型参数本身所占用的内存，而不需要额外存储如梯度或中间激活值等其他信息。这里的计算方式是基于以下假设：
# 参数存储：模型参数在预测时是需要被加载到内存中的。这些参数通常是浮点数，且在这个背景知识中被假设为32位浮点数，即每个参数占用4字节（32位 = 4字节）的内存空间。
# 内存计算：为了将参数的内存占用从字节转换为更常用的单位MB（兆字节），我们需要进行单位转换。由于1MB等于1024^2字节（即1MB = 1024KB = 1024 * 1024字节），因此我们通过乘以4（每个参数的字节数）然后除以1024^2来进行转换。


# 打印分割线
print("-" * 50)

# 练习4：计算二阶导数对计算图的影响和时间消耗

print("练习4：计算二阶导数")
# 在PyTorch中，计算二阶导数需要设置create_graph=True
# 这会导致计算图的构建更加复杂，消耗更多的内存和计算时间

# 示例：计算参数的二阶导数
loss = criterion(model(inputs), labels)
first_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)   #torch.autograd.grad用于计算梯度，create_graph=True表示构建用于计算高阶导数的计算图
# torch.autograd.grad 和 torch.grad 实际上在 PyTorch 中指的是同一个函数，它们没有区别。torch.grad 是 torch.autograd.grad 的别名，两者都用于自动计算梯度。
# 计算二阶导数
second_grad = []
for grad in first_grad:
    grad2 = torch.autograd.grad(grad.sum(), model.parameters(), retain_graph=True)  #计算gard.sum()对model.parameters()的梯度，retain_graph=True表示保留计算图，以便多次使用
    #model.parameters()实际上是一个生成器，每次调用都会返回一个新的参数，因此需要使用for循环来遍历所有的参数
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
    if torch.cuda.device_count() <= 1:
        print("GPU数量不足，无法演示多GPU并行。")

# 优点：可以处理更大的模型和批量，减轻单个GPU的内存压力。
# 缺点：需要在GPU之间传输数据，增加了通信开销，可能降低计算效率。

# 打印分割线
print("-" * 50)

# 总结
print("总结：在本示例中，我们构建了一个包含偏置项的单隐藏层神经网络，并演示了前向传播和反向传播的过程。我们还回答了练习中的问题，包括计算梯度维度、推导方程、计算内存占用、讨论二阶导数对计算图的影响，以及如何在多GPU环境下进行模型并行。")


# 在以上代码示例中，我们使用了以下主要函数和类：
#
# torch.nn.Module：所有神经网络模块的基类。自定义的神经网络需要继承它。
# 参数：
# self：实例本身。
# 示例：
# class SimpleNeuralNetwork(nn.Module):
#     def __init__(self, ...):
#         super(SimpleNeuralNetwork, self).__init__()
#         ...

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
