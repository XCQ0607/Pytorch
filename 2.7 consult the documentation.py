import torch
import torch.nn as nn
import torch.optim as optim

print("2.7. 查阅文档")

# 2.7.1. 查找模块中的所有函数和类
print("\n2.7.1. 查找模块中的所有函数和类")
print("torch.nn模块中的所有属性:")
nn_attributes = dir(nn)
print(", ".join(attr for attr in nn_attributes if not attr.startswith("_")))
# 在Python中，模块是一个包含Python定义和语句的文件。模块可以定义函数、类和变量。模块也可以包含可执行的代码。
# 当我们提到“模块中的所有属性”时，我们指的是模块中定义的所有名称。这些名称可以是函数、类、变量等。在Python中，我们可以使用dir()函数来获取一个模块、类、实例或其他任何具有__dir__()方法的对象的所有属性列表。
# 这里的“属性”是一个广义的概念，它包括了模块中定义的所有函数、类、变量等。如果你想要了解这些属性的具体类型或用途，你通常需要查阅相应的文档或源代码。
# 另外，需要注意的是，dir()函数返回的列表中的属性名称是按照字母顺序排序的，而不是按照它们在模块中定义的顺序。

# 2.7.2. 查找特定函数和类的用法
print("\n2.7.2. 查找特定函数和类的用法")
print("torch.ones函数的帮助文档:")
help(torch.ones)

# 2.7.3. 创建张量的示例
print("\n2.7.3. 创建张量的示例")

# torch.ones示例
print("torch.ones函数示例:")
ones_tensor = torch.ones(3, 4, dtype=torch.float32)
print(ones_tensor)

# torch.zeros示例
print("\ntorch.zeros函数示例:")
zeros_tensor = torch.zeros(2, 3, 5)
print(zeros_tensor)

# torch.randn示例
print("\ntorch.randn函数示例:")
random_tensor = torch.randn(2, 3)
print(random_tensor)
#randn返回一个张量，包含了从标准正态分布（均值为0，方差为1）中抽取的一组随机数。张量的形状由参数指定。

# 2.7.4. 张量操作示例
print("\n2.7.4. 张量操作示例")

# 张量加法
print("张量加法示例:")
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.add(a, b)
print(f"a + b = {c}")

# 张量乘法
print("\n张量乘法示例:")
matrix1 = torch.randn(2, 3)
matrix2 = torch.randn(3, 2)
result = torch.mm(matrix1, matrix2)
print(f"矩阵乘法结果:\n{result}")

# 2.7.5. 神经网络模块示例
print("\n2.7.5. 神经网络模块示例")

#nn模块是一个包含各种神经网络层和其他模块的模块。它提供了许多常用的神经网络层，如线性层、卷积层、池化层等。
#nn.Linear函数是一个线性层，它接受输入张量并返回一个输出张量。它的参数包括输入特征的数量和输出特征的数量。
#如：nn.Linear(10, 5)表示一个输入维度为10，输出维度为5的线性层。
# 输出为：
class SimpleNet(nn.Module):
    def __init__(self): # 定义网络结构，定义网络的层和参数，这个函数在创建对象时被调用，用于初始化对象的属性。
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5) # 定义一个线性层，输入维度为10，输出维度为5
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''
nn.Linear是PyTorch中torch.nn模块提供的一个类，用于创建一个线性变换层，也常被称为全连接层（Fully Connected Layer）或密集层（Dense Layer）。
线性变换的基本概念
线性变换可以用以下数学公式表示：
[ y = xA^T + b ]
其中：
( y ) 是输出向量。
( x ) 是输入向量。
( A ) 是权重矩阵。
( b ) 是偏置向量。
( A^T ) 表示 ( A ) 的转置。但在某些文献和实现中，也可能直接使用 ( y = Ax + b ) 的形式，这取决于具体的矩阵乘法定义。
在nn.Linear(10, 5)中，这个线性层接受一个维度为10的输入向量，并输出一个维度为5的向量。这意味着权重矩阵 ( A ) 的形状是 ( 5 \times 10 )，偏置向量 ( b ) 的形状是 ( 5 )。

可视化理解
想象你有一个长度为10的输入向量，它可能代表某个数据点的10个特征。这个线性层会拿这个向量，与权重矩阵相乘，并加上偏置向量，最后输出一个长度为5的向量。
这个过程可以看作是输入数据的一个“转换”或“映射”。通过调整权重和偏置，这个线性层可以学习如何最好地将输入数据转换为对后续任务（如分类或回归）有用的输出。
在神经网络中，这样的线性层通常会与其他层（如激活函数层、卷积层等）组合使用，以构建更复杂的模型。
例子：
import torch
import torch.nn as nn
# 创建一个线性层实例，输入维度为10，输出维度为5
linear_layer = nn.Linear(10, 5)
# 创建一个随机输入向量，形状为(1, 10)
input_vector = torch.randn(1, 10)
# 通过线性层传递输入向量，得到输出向量
output_vector = linear_layer(input_vector)
# 输出向量的形状应该是(1, 5)
print(output_vector.shape)  # 输出: torch.Size([1, 5])

在深度学习和PyTorch中，当我们谈论一个层的“输入维度”时，我们通常指的是单个数据样本的特征数量。在这个上下文中，“维度”并不是指数组的维度（比如1D、2D、3D数组等），而是指特征的维度。
这里的input_vector是一个2D张量（在PyTorch中，张量是一个可以包含多维数据的容器），其形状为(1, 10)。这意味着：

第一维（也称为批次维度或batch dimension）是1，表示这个张量包含1个数据样本。

第二维是10，表示每个数据样本有10个特征。

因此，当我们说“输入维度为10”时，我们是指每个输入样本是一个10维向量，即包含10个特征。这个10维向量是线性层nn.Linear(10, 5)所期望的输入，因为它被设计为接受一个10维的输入向量，并输出一个5维的向量。
在神经网络中，我们通常处理的是批次数据，而不是单个样本。批次数据是一个包含多个样本的张量，其中每个样本都有相同的特征维度。在这个例子中，尽管我们只有一个样本（批次大小为1），但PyTorch的处理方式是通用的，可以很容易地扩展到更大的批次。
总结一下，“输入维度为10”意味着每个输入样本是一个包含10个特征的向量，而张量的形状(1, 10)表示这是一个包含1个这样样本的批次。

shape为(2, 3)的张量并不表示一个“输入维度”为6的层。在深度学习和PyTorch中，“输入维度”通常指的是单个数据样本的特征数量，而不是整个张量的元素总数。
对于一个shape为(2, 3)的张量，这表示：
第一维（批次维度）是2，意味着这个张量包含2个数据样本。
第二维是3，表示每个数据样本有3个特征。
所以，这个张量适合作为一个期望“输入维度”为3的层的输入。换句话说，你应该使用nn.Linear(3, ...)（其中...是输出维度的数量）来定义一个能够处理这种输入的线性层。
“输入维度”是特定于每个样本的，而不是整个批次的。在这个例子中，每个样本都是一个3维向量，因此“输入维度”是3，而不是6（6是整个张量的元素总数）。

在nn.Linear(10, 5)中，这个线性层接受一个维度为10的输入向量，并输出一个维度为5的向量。这意味着权重矩阵 ( A ) 的形状是 ( 5 \times 10 )，偏置向量 ( b ) 的形状是 ( 5 )。


nn.Linear执行的操作是：
[ y = xW + b ]
其中：
( y ) 是输出向量。
( x ) 是输入向量。
( W ) 是权重矩阵（在nn.Linear文档中通常表示为weight），其形状为 (out_features, in_features)。
( b ) 是偏置向量（在nn.Linear文档中通常表示为bias），其形状为 (out_features,)。
对于nn.Linear(10, 5)，我们有以下情况：
输入维度（in_features）是10。
输出维度（out_features）是5。
因此：nn.Linear(10, 5)
权重矩阵 ( W ) 的形状是 (5, 10)。
偏置向量 ( b ) 的形状是 (5,)。

假设传入是(1,10),(1,10)*(10,5)才是(1, 5)吧，为什么你的意思是(1,10)*(5,10)的结果是(1,5)?所以W是(10,5)还是(5,10)？
在数学上：
对于矩阵乘法，要求第一个矩阵的列数等于第二个矩阵的行数
结果矩阵的维度将是：(第一个矩阵的行数, 第二个矩阵的列数)
让我们分析两种情况：
如果 W 是 (10,5):
Copy输入 x: (1,10)
权重 W: (10,5)
x * W = (1,5)  ✓ 这样计算是正确的
如果 W 是 (5,10):
Copy输入 x: (1,10)
权重 W: (5,10)
x * W = ??? 这样无法直接计算，因为维度不匹配

从矩阵乘法的角度来看，W 确实应该是 (10,5) 维的矩阵。
让我们用代码验证一下：
========
pythonCopyimport torch
import torch.nn as nn
linear = nn.Linear(10, 5)
# 查看权重W的转置前后的形状
print("Weight shape:", linear.weight.shape)        # 输出: torch.Size([5, 10])
print("Weight.T shape:", linear.weight.T.shape)    # 输出: torch.Size([10, 5])
# 验证前向传播
x = torch.randn(1, 10)
========
这里揭示了一个重要的实现细节：
在PyTorch的nn.Linear中，权重矩阵实际上是以(5,10)的形式存储的
但在进行实际计算时，PyTorch会在内部自动处理矩阵乘法，相当于使用了权重矩阵的转置
所以实际的计算过程等价于 x * W.T
这就是为什么我之前说W的形状是(5,10)，这确实是PyTorch中存储的形式，但从数学运算的角度来说，你说得对，实际的乘法运算需要的是(10,5)的矩阵。
所以总结一下：
从纯数学角度：权重矩阵应该是(10,5)以实现(1,10)->(1,5)的变换
在PyTorch实现中：权重存储为(5,10)，但在计算时会进行适当的转置操作
最终效果是一样的：都是将10维输入转换为5维输出
'''

net = SimpleNet()   # 实例化一个简单的神经网络
print("简单神经网络结构:")
print(net)

# 2.7.6. 优化器示例
print("\n2.7.6. 优化器示例")
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
print("SGD优化器参数:")
print(optimizer.state_dict())     # 输出: {'momentum': 0.9, 'lr': 0.01}
#optimizer.state_dict()返回一个字典，包含了优化器的状态信息，如学习率、动量等。
#同样，它还有其他函数
#optimizer.zero_grad()：将优化器的梯度清零，为下一次迭代做准备。
#optimizer.step()：根据当前梯度更新模型参数。在训练神经网络时，我们需要在每个训练迭代中调用这个函数来更新模型的参数。
#optimizer.param_groups：返回一个包含所有参数组的列表，每个参数组都是一个字典，包含了该组参数的信息，如学习率、动量等。
#optimizer.load_state_dict(state_dict)：从一个字典中加载优化器的状态信息。这在保存和加载模型时非常有用，因为它允许我们在不同的训练过程中恢复优化器的状态。
#optimizer.state_dict()['param_groups'][0]['lr'] = 0.001      # 更改学习率,param_groups是一个列表，[0]是第一个参数组,['lr']是第一个参数组的学习率
'''
optim.SGD是PyTorch中用于优化神经网络权重的一个函数，它实现了随机梯度下降（Stochastic Gradient Descent）算法或其变种。
optim.SGD的主要作用是根据计算出的梯度来更新和调整模型的参数，以最小化模型的损失函数。在训练神经网络的过程中，我们通过反向传播计算出损失函数关于模型参数的梯度，然后使用optim.SGD来根据这些梯度更新模型的参数，从而逐步优化模型的表现。

可以传入的参数
optim.SGD函数可以接收多个参数，其中一些重要的参数包括：
params (iterable)：待优化参数的iterable或者是定义了参数组的dict。通常，我们通过将模型的.parameters()方法传递给这个参数来指定需要优化的参数，如net.parameters()。
lr (float, optional)：学习率（learning rate），默认值为1e-2。学习率决定了参数更新的步长大小，过大可能导致训练不稳定，过小可能导致训练速度过慢。
momentum (float, optional)：动量因子，默认值为0。动量可以帮助加速SGD在相关方向上的收敛，并抑制震荡。它做的是通过加入之前梯度的部分和来更新当前的梯度。
weight_decay (float, optional)：权重衰减（L2惩罚），默认值为0。权重衰减是一种正则化方法，它通过对模型参数施加L2范数的惩罚来防止模型过拟合。
dampening (float, optional)：动量的抑制因子，默认值为0。它用于改变动量的计算方式，但在大多数情况下保持为0即可。
nesterov (bool, optional)：是否使用Nesterov动量，默认值为False。Nesterov动量是传统动量方法的一个变种，它在计算梯度时使用了一个预估的参数位置。
返回的数
optim.SGD函数返回一个优化器对象，该对象包含了进行参数优化所需的所有信息和方法。你可以使用这个优化器对象来执行参数的更新操作，例如在每个训练迭代中调用.step()方法来根据计算出的梯度更新模型的参数，以及调用.zero_grad()方法来清除已计算的梯度信息，为下一次迭代做准备。

import torch
import torch.nn as nn
import torch.optim as optim

# 假设net是一个已经定义好的神经网络模型
net = ...  # 你的模型定义在这里
# 创建优化器对象，指定需要优化的参数、学习率和动量因子
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# 训练循环
for epoch in range(num_epochs):  # num_epochs是总的训练轮数
    for inputs, targets in dataloader:  # dataloader是数据加载器，提供训练数据
        # 前向传播
        outputs = net(inputs)
        loss = criterion(outputs, targets)  # criterion是损失函数，用于计算损失值    #每个数据的targets是一个标量
        # 反向传播计算梯度
        optimizer.zero_grad()  # 清除之前的梯度信息（如果有的话）
        loss.backward()  # 反向传播计算梯度  
        # 使用优化器更新模型参数
        optimizer.step()  # 根据计算出的梯度更新模型参数
'''

# 2.7.7. 损失函数示例
print("\n2.7.7. 损失函数示例")
loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数
input = torch.randn(3, 2)     # 输入张量，形状为(3, 2)
target = torch.empty(3, dtype=torch.long).random_(2)    # torch.empty(3, dtype=torch.long)创建一个形状为(3,)的空张量，dtype=torch.long指定张量的数据类型为长整型。random_(2)表示将张量中的元素随机替换为0或1。
loss = loss_fn(input, target)    # 计算损失值,input是模型的输出，target是目标标签,loss表示模型的预测输出与目标标签之间的差异
print(f"输入: {input}")
print(f"目标: {target}")  #shape为(3,) 表示它是一个一维张量，包含3个元素
print(f"损失: {loss.item()}")

# torch.empty(3, dtype=torch.long) 创建了一个形状为 (3,) 的空张量，并指定了其数据类型为 torch.long（即64位整型）。这个张量初始时包含未初始化的数据，也就是说，它的内容是“空的”或者说是不确定的。紧接着，.random_(2) 方法被调用，这个方法会将张量中的每个元素随机地设置为0或1。这里的 2 表示随机数的范围是从0到2（不包括2），因此可能的值只有0和1。
# target 张量的形状是 (3,)，表示它是一个一维张量，包含3个元素。


"""
本章使用的主要函数和类:
1. dir(): 列出模块的所有属性
2. help(): 显示函数或类的帮助文档
3. torch.ones(): 创建全1张量
4. torch.zeros(): 创建全0张量
5. torch.randn(): 创建随机正态分布张量
6. torch.add(): 张量加法
7. torch.mm(): 矩阵乘法
8. nn.Module: 神经网络模块基类
9. nn.Linear: 线性层
10. optim.SGD: 随机梯度下降优化器
11. nn.CrossEntropyLoss: 交叉熵损失函数
"""
