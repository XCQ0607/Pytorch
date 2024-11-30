# 4.8. 数值稳定性和模型初始化
print("4.8. 数值稳定性和模型初始化")

# 导入必要的库
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# 设置默认张量类型为 float32
torch.set_default_dtype(torch.float32)

# 4.8.1. 梯度消失和梯度爆炸

# 4.8.1.1. 梯度消失
print("\n4.8.1.1. 梯度消失示例")

# 创建从 -8.0 到 8.0，步长为 0.1 的张量 x，并设置需要计算梯度
# torch.arange(start, end, step, requires_grad) 返回一个 1-D 张量，包含从 start 到 end（不包括）的值，步长为 step
# 参数：
# - start (float)：序列的起始值。
# - end (float)：序列的结束值（不包含）。
# - step (float)：相邻值之间的间隔。
# - requires_grad (bool)：是否需要对该张量的操作进行自动求导。
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)    #设置间隔
#x={-8.0, -7.9, -7.8, ..., 7.8, 7.9}
#设置数量
# x = torch.linspace(-8.0, 8.0, 100, requires_grad=True)    #设置数量

# 对 x 应用 sigmoid 函数
# torch.sigmoid(input) 对输入张量的每个元素应用 sigmoid 函数
# 参数：
# - input (Tensor)：输入张量
# sigmoid 函数：y = 1 / (1 + exp(-x))
y = torch.sigmoid(x)    #y=1/(1+exp(-x))

# 计算梯度
# y.backward(gradient) 计算 y 关于 x 的梯度
# 参数：
# - gradient (Tensor)：关于 y 的梯度
# 这里，我们使用 torch.ones_like(x) 指定 y 关于自身的梯度为 1
y.backward(torch.ones_like(x))  #这里如果写成y.backward()，会报错，因为y不是一个标量

# 绘制 sigmoid 函数及其梯度
plt.figure(figsize=(6, 4))    # 设置图像大小为 6x4 英寸
plt.plot(x.detach().numpy(), y.detach().numpy(), label='Sigmoid')   #sigmoid函数
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='Gradient') #梯度
plt.legend()    #显示图例，label='Sigmoid'和label='Gradient'
plt.title('Sigmoid Function and its Gradient')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)    #显示网格
plt.show()

# 解释：
# Sigmoid 函数将输入压缩到 (0, 1) 范围内。
# 它的梯度在 x=0 处达到最大值，当 x 远离 0 时，梯度趋于 0。
# 这导致在深度网络中使用 sigmoid 激活时，梯度会消失。

# 重置梯度
x.grad.zero_()  #..zero_() 是 in-place 操作，将张量的所有元素设置为 0

# 比较 ReLU 激活函数
# 应用 ReLU 函数
# torch.relu(input) 对输入张量的每个元素应用 ReLU 函数
# 参数：
# - input (Tensor)：输入张量
y_relu = torch.relu(x)

# 计算梯度
x.grad.zero_()
y_relu.backward(torch.ones_like(x))

# 绘制 ReLU 函数及其梯度
plt.figure(figsize=(6, 4))
plt.plot(x.detach().numpy(), y_relu.detach().numpy(), label='ReLU')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='Gradient')
plt.legend()
plt.title('ReLU Function and its Gradient')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# 解释：
# ReLU 函数对于负输入输出 0，对于正输入输出输入本身。
# 它的梯度在负输入处为 0，正输入处为 1。
# 这有助于缓解梯度消失问题，因为对于正输入，梯度不会衰减。

# 4.8.1.2. 梯度爆炸
print("\n4.8.1.2. 梯度爆炸示例")

# 初始化一个随机矩阵 M
# torch.randn(*sizes) 返回一个张量，包含从标准正态分布（均值为 0，方差为 1）中抽取的随机数
# 参数：
# - sizes (int...)：定义输出张量形状的整数序列
M = torch.randn(4, 4)
print("一个矩阵 M:\n", M)

# 将 M 乘以 100 个随机矩阵，并观察结果
for i in range(100):
    M = M @ torch.randn(4, 4)   #@ 是矩阵乘法运算符

print("乘以100个矩阵后 M:\n", M)

# 计算 M 的范数
norm_M = torch.norm(M)  #torch.norm(input) 返回输入张量的范数,L2范数
print("矩阵 M 的范数：", norm_M.item())
#计算L2范数就是将norm_M中的所有元素平方和开根号

# 解释：
# 连续乘以多个矩阵可能导致值指数级增长，导致数值不稳定。
# 在深度网络中，如果权重初始化不当，反向传播时梯度可能会爆炸。

# 4.8.1.3. 打破对称性
print("\n4.8.1.3. 打破对称性示例")

# 示例：展示在神经网络中打破对称性的重要性

# 定义一个具有一个隐藏层、两个神经元的简单神经网络
# 如果我们将所有权重初始化为相同的值，网络可能无法学习

# 定义输入和输出维度
input_dim = 2
hidden_dim = 2
output_dim = 1

# 创建使用相同值初始化的网络
net_same = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.Sigmoid(),
    nn.Linear(hidden_dim, output_dim)
)
#
# class SimpleNeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleNeuralNetwork, self).__init__() #调用父类的构造函数，这里显式传入self对象，并传入当前类的名称
#         # 定义隐藏层，使用线性变换，添加偏置项
#         self.hidden = nn.Linear(input_size, hidden_size, bias=True)
#         # 定义激活函数，使用ReLU
#         self.activation = nn.ReLU()
#         # 定义输出层，使用线性变换，添加偏置项
#         self.output = nn.Linear(hidden_size, output_size, bias=True)

# net_same 和 SimpleNeuralNetwork 类在功能上都是定义了简单的神经网络结构，但它们之间有几个关键的区别：
#
# 结构定义方式：
# net_same 使用 nn.Sequential，这是一个有序的容器，其中包含的模块会按照在构造函数中传入的顺序被添加到计算图中。nn.Sequential 使得模型的定义更加简洁和直观。
# SimpleNeuralNetwork 类是继承自 nn.Module 的自定义类。这种方式提供了更大的灵活性，允许你定义更复杂的网络结构和自定义的前向传播逻辑。
# 激活函数：
# 在 net_same 中，使用的是 nn.Sigmoid() 作为激活函数。
# 而在 SimpleNeuralNetwork 类中，使用的是 nn.ReLU() 作为激活函数。
# 这两个激活函数在数学特性和实际应用中有所不同。例如，nn.Sigmoid() 会将输出压缩到 (0, 1) 范围内，而 nn.ReLU() 是一个分段线性函数，其输出为输入值和 0 之间的较大值。这些差异会影响网络的训练动态和性能。
# 灵活性：
# 使用 nn.Sequential 的 net_same 在定义上更加简洁，但如果你需要添加复杂的逻辑（比如条件语句、循环或者自定义层），则可能不够灵活。
# 相比之下，SimpleNeuralNetwork 类允许你在 forward 方法中实现任意复杂的前向传播逻辑（尽管在你提供的代码片段中没有显示 forward 方法）。
# 扩展性：
# 对于 SimpleNeuralNetwork，你可以轻松地添加更多的方法或属性，比如用于模型验证、特定类型的数据预处理或后处理的方法。
# 而对于 net_same，除非你将其封装在一个更大的类中，否则添加这样的功能可能会不太方便。
# 总的来说，net_same 和 SimpleNeuralNetwork 在功能上是相似的，但它们的定义方式、激活函数的选择以及灵活性和扩展性方面存在差异。选择哪种方式取决于你的具体需求和偏好。

# 定义模型：
# SimpleNeuralNetwork是一个自定义类，继承自nn.Module。在这个类中，你通常会定义网络的层（在__init__方法中）和前向传播的逻辑（在forward方法中）。
# nn.Sequential是一个有序的容器，它包含了一系列按顺序执行的模块（层）。你不需要显式地定义forward方法，因为nn.Sequential会自动按照添加的顺序执行其中的模块。
# 实例化模型：
# 对于SimpleNeuralNetwork，你可以通过model = SimpleNeuralNetwork()来实例化模型。
# 对于nn.Sequential，你可以通过net_same = nn.Sequential(...)来实例化一个顺序模型。
# 前向传播：
# 无论是SimpleNeuralNetwork还是nn.Sequential，你都可以通过调用实例并传入输入数据来进行前向传播，例如outputs = model(inputs)或outputs = net_same(inputs)。
# 优化器：
# 对于两种类型的模型，你都可以使用相同的优化器，例如SGD。在创建优化器时，你需要传入模型的参数，这些参数是通过调用.parameters()方法获得的。无论是model.parameters()还是net_same.parameters()，它们都会返回模型中可训练的参数。
# 训练循环：
# 在训练循环中，无论是使用SimpleNeuralNetwork还是nn.Sequential，步骤都是类似的：前向传播、计算损失、反向传播和优化器步骤。


# 将所有权重和偏置初始化为相同的值
def init_weights_same(m):
    if isinstance(m, nn.Linear):    #isinstance() 函数用于检查一个对象是否是指定类的实例
        nn.init.constant_(m.weight, 0.5)    #nn.init.constant_(tensor, val) 用常数 val 填充输入张量
        nn.init.constant_(m.bias, 0.0)  #将偏置初始化为0
# 应用初始化函数
net_same.apply(init_weights_same)   #apply() 方法对网络中的每个模块应用指定的函数

# 打印初始化的权重
print("使用相同值初始化的网络参数：")
for name, param in net_same.named_parameters():    #named_parameters() 返回模型的所有参数（包括权重和偏置），包括权重和偏置，而parameters() 返回模型的所有参数（包括权重和偏置），但不包括权重和偏置
    print(name, param.data) #打印参数名和参数值
#param是权重，param.data是权重值

# #权重
# Parameter containing:
# tensor([0., 0.], requires_grad=True)
# #权重值
# tensor([0., 0.])

# 创建一个简单的数据集
# 尝试学习 XOR 函数
inputs = torch.tensor([[0., 0.],
                       [0., 1.],
                       [1., 0.],
                       [1., 1.]])
labels = torch.tensor([[0.],
                       [1.],
                       [1.],
                       [0.]])

# 定义损失函数和优化器
criterion = nn.MSELoss()    #均方误差损失函数
optimizer_same = torch.optim.SGD(net_same.parameters(), lr=0.1) #optimizer_same = torch.optim.SGD(net_same.parameters(), lr=0.1)：这里创建了一个随机梯度下降（Stochastic Gradient Descent, SGD）优化器，用于更新网络的权重。net_same.parameters() 获取了网络中所有可训练的参数（即权重和偏置），lr=0.1 设置了学习率为0.1。

# 训练网络
epochs = 10
print("\n使用相同初始化值训练网络：")
for epoch in range(epochs):
    optimizer_same.zero_grad()   #梯度清零
    outputs = net_same(inputs)  #前向传播
    loss = criterion(outputs, labels)    #计算损失
    loss.backward()  #反向传播
    optimizer_same.step()    #更新权重
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 解释：
# 由于所有权重初始化为相同的值，且输入具有对称性，
# 隐藏单元计算出相同的输出，梯度也相同。
# 这导致网络无法学习 XOR 函数，因为对称性未被打破。

# 现在，随机初始化权重以打破对称性
net_random = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.Sigmoid(),
    nn.Linear(hidden_dim, output_dim)
)

# 当您在PyTorch中使用nn.Linear或其他层创建神经网络时，如果您没有明确指定权重和偏置的初始化方法，那么PyTorch会使用默认的随机初始化方法来初始化这些参数。
# 这里创建了一个顺序模型net_random，其中包含两个线性层（nn.Linear）和一个Sigmoid激活函数。当这两个nn.Linear层被实例化时，它们的权重（weight）和偏置（bias）参数会被自动初始化。默认情况下，权重通常使用从均匀分布或正态分布中随机抽取的值进行初始化，而偏置通常被初始化为零或接近零的小值。

# 使用默认的随机初始化
print("\n使用随机初始化的网络参数：")
for name, param in net_random.named_parameters():
    print(name, param.data)
#原本应该def一个default_weight_bais(m):函数
# def default_weight_bais(m):
#     if isinstance(m, nn.Linear):
#         nn.init.constant_(m.weight, 0.5)
#         nn.init.constant_(m.bias, 0.0)
# def default_weight_bais(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         nn.init.zeros_(m.bias)
#
# net_random.apply(default_weight_bais)


# 两种不同的初始化方法用于神经网络的权重（weight）和偏置（bias）。这两种方法分别是：
#
# 常数初始化（Constant Initialization）:
# nn.init.constant_(m.weight, 0.5)
# nn.init.constant_(m.bias, 0.0)
# 在这种方法中，权重被初始化为一个固定的常数值，这里是0.5，而偏置被初始化为0.0。常数初始化有时用于特定的场景，但通常不是通用的最佳实践，因为它不考虑网络层的输入或输出节点数量，可能会导致不良的梯度流动。
#
# Xavier均匀初始化（Xavier Uniform Initialization）配合零初始化偏置（Zeros Initialization for Bias）:
# nn.init.xavier_uniform_(m.weight)
# nn.init.zeros_(m.bias)
# Xavier（或称为Glorot）初始化是一种更为复杂且通常更有效的权重初始化方法。它根据输入和输出神经元的数量来自动调整权重的初始值范围，旨在保持各层在初始化时的激活分布和反向传播时的梯度分布大致相同。这有助于减少梯度消失或梯度爆炸的问题，从而加速训练过程。
# 对于偏置，这里使用的是零初始化，即所有偏置项在开始时都被设置为0。零初始化偏置是常见的做法，因为偏置项通常可以在训练过程中通过学习来调整。

# 定义优化器
optimizer_random = torch.optim.SGD(net_random.parameters(), lr=0.1)

# 训练网络
print("\n使用随机初始化训练网络：")
for epoch in range(epochs):
    optimizer_random.zero_grad()
    outputs = net_random(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer_random.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 解释：
# 通过随机初始化，打破了对称性，使网络能够更有效地学习 XOR 函数。
# XOR函数是一个逻辑运算符，它接受两个布尔输入（0或1），并返回一个布尔输出。XOR函数的真值表如下：
# 输入1	输入2	输出
# 0	0	0
# 0	1	1
# 1	0	1
# 1	1	0
# 当输入1和输入2不同时，输出为1，否则输出为0。

# 4.8.2. 参数初始化

# 4.8.2.2. Xavier初始化
print("\n4.8.2.2. Xavier初始化示例")

# 定义一个新的神经网络，并应用 Xavier 初始化

# 定义网络
net_xavier = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)

# 应用 Xavier 初始化
def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(tensor, gain=1.0)
        # 使用均匀分布，根据 Xavier 初始化方法填充输入张量的值
        # 参数：
        # - tensor (Tensor)：n 维张量
        # - gain (float)：可选的缩放因子
        nn.init.xavier_uniform_(m.weight)   #Xavier均匀初始化
        nn.init.zeros_(m.bias)   #零初始化偏置

net_xavier.apply(init_weights_xavier)    #应用初始化函数

# 打印初始化的权重
print("\n使用Xavier初始化的网络参数：")
for name, param in net_xavier.named_parameters():
    print(name, param.data)

# 现在，比较使用 Xavier 初始化和默认初始化的输出方差

# 生成随机输入
inputs = torch.randn(1000, input_dim)

# 通过网络前向传播
outputs = net_xavier(inputs)

# 计算输出的均值和方差
mean = outputs.mean().item()    #.mean()计算张量的均值
var = outputs.var().item()  #.var()计算张量的方差
#还有.std()计算张量的标准差,.max()计算张量的最大值,.min()计算张量的最小值,.sum()计算张量的和,.prod()计算张量的积,.abs()计算张量的绝对值,.exp()计算张量的指数,.log()计算张量的对数,.sqrt()计算张量的平方根,.pow()计算张量的幂,.sin()计算张量的正弦,.cos()计算张量的余弦,.tan()计算张量的正切,.asin()计算张量的反正弦,.acos()计算张量的反余弦,.atan()计算张量的反正切,.sinh()计算张量的双曲正弦,.cosh()计算张量的双曲余弦,.tanh()计算张量的双曲正切,.asinh()计算张量的反双曲正弦,.acosh()计算张量的反双曲余弦,.atanh()计算张量的反双曲正切,.round()计算张量的四舍五入,.floor()计算张量的向下取整,.ceil()计算张量的向上取整,.trunc()计算张量的截断,.frac()计算张量的分数部分,.sign()计算张量的符号,.clamp()计算张量的裁剪,.clamp_min()计算张量的最小值裁剪,.clamp_max()计算张量的最大值裁剪,.clamp_()计算张量的裁剪（in-place）,.clamp_min_()计算张量的最小值裁剪（in-place）,.clamp_max_()计算张量的最大值裁剪（in-place）,.clamp_()计算张量的裁剪（in-place）,.clamp
print(f"\n输出的均值：{mean}, 方差：{var}")

# 与默认初始化比较
net_default = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)

# 前向传播
outputs_default = net_default(inputs)
mean_default = outputs_default.mean().item()    #.mean()计算张量的均值
var_default = outputs_default.var().item()  #.var()计算张量的方差
print(f"默认初始化的输出均值：{mean_default}, 方差：{var_default}")

# 解释：
# Xavier 初始化有助于将输出的方差保持在合理的范围内，
# 这有助于在反向传播过程中保持梯度的稳定。

# 4.8.3. 小结
print("\n4.8.3. 小结")

# 梯度消失和梯度爆炸是深度网络中常见的问题。
# 需要用启发式的初始化方法来确保初始梯度既不太大也不太小。

# 4.8.4. 练习

print("\n4.8.4. 练习")

# 问题1: 除了多层感知机的排列对称性之外，还能设计出其他神经网络可能会表现出对称性且需要被打破的情况吗？

print("\n问题1：")
print("在卷积神经网络中，如果所有的卷积核（滤波器）被初始化为相同的值，"
      "那么所有的特征映射将是相同的，这会导致模型无法学习多样的特征。"
      "因此，需要随机初始化卷积核权重以打破对称性。")

# 问题2: 我们是否可以将线性回归或softmax回归中的所有权重参数初始化为相同的值？

print("\n问题2：")
print("在线性回归中，初始化所有权重为相同的值（例如0）是可以的，因为模型是凸的，"
      "并且没有隐藏层，不会出现对称性破坏的问题。"
      "但在softmax回归中，初始化所有权重为相同的值会导致对称性问题，"
      "因为所有类别的参数都相同，模型无法学习到类别之间的差异。"
      "因此，softmax回归中的权重应该随机初始化。")

# 问题3: 在相关资料中查找两个矩阵乘积特征值的解析界。这对确保梯度条件合适有什么启示？

print("\n问题3：")
print("对于矩阵A和B，其乘积AB的特征值λ满足以下关系："
      "λ(AB) ≤ λ_max(A) * λ_max(B)，其中λ_max表示矩阵的最大特征值。"
      "这意味着，如果矩阵的特征值大于1，乘积的特征值会指数增长，导致梯度爆炸；"
      "如果特征值小于1，乘积的特征值会指数减小，导致梯度消失。"
      "因此，为了确保梯度的数值稳定性，需要初始化权重矩阵，使其特征值接近于1。")

# 问题4: 如果我们知道某些项是发散的，我们能在事后修正吗？看看关于按层自适应速率缩放的论文。

print("\n问题4：")
print("是的，如果我们在训练过程中发现梯度或激活值发散，可以采用一些方法进行修正。"
      "例如，使用Layer-wise Adaptive Rate Scaling (LARS)算法，它为每一层适应性地调整学习率，"
      "使得每一层的权重更新在合适的范围内，防止发散。"
      "此外，还可以使用梯度裁剪、批归一化等方法来控制梯度的范围。")

# 总结:
# 在本代码示例中，我们演示了梯度消失和梯度爆炸的问题，说明了初始化方法对神经网络训练的影响。
# 通过使用Xavier初始化和随机初始化，我们可以有效地缓解梯度消失或爆炸的问题，保持训练过程的数值稳定性。
# 同时，我们强调了在神经网络中打破对称性的重要性，确保模型能够学习到丰富的特征。


# 简单的可视化例子，展示隐藏层单元之间的对称性如何影响神经网络的训练。这个例子会说明在不同情况下隐藏层单元的行为，以及如何通过打破对称性来改善网络的学习能力。
#
# 示例 : 隐藏层单元之间的对称性
# 假设我们有一个简单的多层感知机（MLP）网络，输入层有2个特征，隐藏层有2个神经元，输出层有1个神经元。初始时，假设隐藏层的两个神经元（单元）的权重和偏置完全相同。
#
# 网络结构：
#
# 输入层：2个特征
# 隐藏层：2个神经元（权重和偏置初始化相同）
# 输出层：1个神经元
# 前向传播：假设输入是 x=[x1,x2]，对于两个隐藏单元，它们的输入和权重相同，因此它们的激活值也是相同的。
# 激活函数（如ReLU）会使两个单元的输出一样。
# 反向传播：
#
# 在反向传播中，由于梯度也是相同的，这导致两个隐藏单元的权重更新完全一致。
# 示意图：
#  输入层:  x1, x2
#    |       |
#    V       V
#  隐藏层:  h1, h2 (相同的权重和偏置)
#    |       |
#    V       V
#  输出层:  y (1个输出)
# 可视化效果：
# 前向传播时，两个隐藏单元的输出完全相同，因此它们的梯度也是相同的，权重更新也完全一致。
# 训练过程中，尽管有两个隐藏单元，但它们的行为无法区分。它们在网络中发挥相同的作用，网络的能力与仅使用一个隐藏单元几乎没有区别。



'''
在神经网络中，隐藏层单元之间的对称性指的是当多个隐藏单元在某些条件下（例如，权重初始化相同或具有相同的输入）时，它们的行为、输出和梯度更新在训练过程中是完全一致的。简单来说，如果隐藏层中的多个神经元在前向传播和反向传播中表现出完全相同的模式，它们就具有对称性。

隐藏层单元之间对称性的体现：
    权重初始化相同：
        如果所有隐藏单元的权重在训练开始时初始化为相同的值（例如都初始化为相同的常数），那么在前向传播时，每个隐藏单元对输入的处理方式就完全相同。
        假设每个隐藏单元都有相同的权重和偏置，并且接收到相同的输入，它们的输出就会完全相同，即每个隐藏单元的激活值会相同。
    相同的激活函数：
        即使隐藏单元的输入和权重相同，它们经过激活函数（如ReLU、Sigmoid、tanh等）处理后，输出也将是相同的。比如，若每个单元的输入加权和相同，激活函数对这些输入的输出也相同。
    反向传播中的梯度更新：
        在反向传播过程中，如果隐藏层单元的输出相同，那么它们对误差的贡献也会相同，导致计算出的梯度也会相同。
        由于梯度更新的公式中会涉及到梯度与学习率的乘积，如果所有的隐藏单元的梯度都相同，那么每个单元的权重更新步长也将相同。
        
为什么对称性会带来问题？
  当隐藏层的多个单元之间具有对称性时，它们的行为会高度相似，甚至完全一致。具体来说，假设网络有多个隐藏单元，但它们的输入、权重、偏置都相同，前向传播过程中它们的激活值也相同。这种相同的激活会导致反向传播时，梯度也相同，进而导致在训练过程中它们的权重更新也是完全一致的。
这种情况会带来以下问题：
    缺乏多样性：如果多个隐藏单元的行为和梯度更新完全一致，那么这些单元就无法学习到不同的特征。换句话说，网络会退化为“冗余的”神经元，这些神经元实际上是在学习相同的功能，而不是发挥它们各自独特的作用。
    表达能力受限：多个隐藏单元之间的对称性会导致网络的表达能力下降，因为它们并没有各自独立地学习不同的数据特征或模式。网络的实际能力就会变得非常有限。
    
举个简单的例子：
  假设你有一个包含两个隐藏单元的隐藏层，并且这两个单元的权重和偏置都初始化为相同的值。如果在前向传播时，两个隐藏单元接收到相同的输入，且使用相同的激活函数，那么它们的激活值会完全相同。接着，在反向传播时，两个隐藏单元的梯度也会相同，导致这两个单元的权重更新完全一致。
  最终，这两个隐藏单元的学习过程就无法区分彼此，它们的权重和行为几乎不会变化，网络相当于只用一个隐藏单元来表示两者的组合，从而丧失了使用两个单元来表达更多特征的能力。

如何打破对称性？
  为了避免隐藏层单元之间的对称性，通常采取以下措施：
    随机初始化权重：为了打破对称性，通常会对每个隐藏单元的权重进行随机初始化。这样，即使它们接收到相同的输入，因权重不同，它们的输出也会不同，从而可以学习到不同的特征。
    使用激活函数的非线性特性：激活函数（如ReLU、Sigmoid等）本身具有非线性特性，不同的输入会产生不同的输出，即使在权重相同的情况下，也可能由于输入数据不同，激活值不同。
    正则化方法：如暂退法（Dropout）、**批量归一化（Batch Normalization）**等方法，在训练过程中可以随机屏蔽神经元或调整激活值，进一步打破对称性，避免所有隐藏单元学习到相同的特征。

数据增强：通过增加训练数据的多样性，可以有效地减少模型学习到冗余特征的风险。
'''