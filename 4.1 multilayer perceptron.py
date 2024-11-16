print("4.1. 多层感知机")
# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
# 配置中文字体    微软雅黑
rcParams['font.family'] = 'Microsoft YaHei'
# -------------------------------
print("ReLU激活函数示例")
# -------------------------------

# 定义输入范围，设置requires_grad=True以便计算梯度
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

# 使用torch.nn.functional中的relu函数计算ReLU激活函数
# relu函数的定义为：relu(input, inplace=False)
# 参数：
# - input：输入张量
# - inplace：是否进行原地操作，默认为False
y = F.relu(x)     # 计算ReLU激活函数

# 绘制ReLU激活函数的图像
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach().numpy(), y.detach().numpy())    #detach()：返回一个新的张量，与当前张量共享数据，但不具有梯度信息
plt.title('ReLU激活函数')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.show()

# 计算ReLU激活函数的导数
y.backward(torch.ones_like(x))
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach().numpy(), x.grad.detach().numpy())
plt.title('ReLU激活函数的导数')
plt.xlabel('x')
plt.ylabel('grad of ReLU(x)')
plt.show()

# 清除梯度
x.grad.zero_()

# -------------------------------
print("Sigmoid激活函数示例")
# -------------------------------

# 计算Sigmoid激活函数的输出
# torch.sigmoid函数的定义为：sigmoid(input)
# 参数：
# - input：输入张量
y = torch.sigmoid(x)

# 绘制Sigmoid激活函数的图像
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.title('Sigmoid激活函数')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.show()

# 计算Sigmoid激活函数的导数
x.grad.zero_()  # 清除之前的梯度
y.backward(torch.ones_like(x))
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach().numpy(), x.grad.detach().numpy())
plt.title('Sigmoid激活函数的导数')
plt.xlabel('x')
plt.ylabel('grad of Sigmoid(x)')
plt.show()

# -------------------------------
print("Tanh激活函数示例")
# -------------------------------

# 计算Tanh激活函数的输出
# torch.tanh函数的定义为：tanh(input)
# 参数：
# - input：输入张量
y = torch.tanh(x)

# 绘制Tanh激活函数的图像
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.title('Tanh激活函数')
plt.xlabel('x')
plt.ylabel('Tanh(x)')
plt.show()

# 计算Tanh激活函数的导数
x.grad.zero_()  # 清除之前的梯度
y.backward(torch.ones_like(x))
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach().numpy(), x.grad.detach().numpy())
plt.title('Tanh激活函数的导数')
plt.xlabel('x')
plt.ylabel('grad of Tanh(x)')
plt.show()

# -------------------------------
print("PReLU激活函数示例")
# -------------------------------

# 定义PReLU激活函数
# 使用nn.PReLU模块
# PReLU(num_parameters=1, init=0.25)
# 参数：
# - num_parameters：α的数量，可以是1（所有通道共享一个α）或者输入通道数
# - init：α的初始值，默认为0.25

prelu = nn.PReLU(num_parameters=1, init=0.25)

# 计算PReLU激活函数的输出
y = prelu(x)

# 绘制PReLU激活函数的图像
plt.figure(figsize=(5, 2.5))    # 绘制PReLU激活函数的图像
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.title('PReLU激活函数')
plt.xlabel('x')
plt.ylabel('PReLU(x)')
plt.show()

# 计算PReLU激活函数的导数
x.grad.zero_()  # 清除之前的梯度
y.backward(torch.ones_like(x))
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach().numpy(), x.grad.detach().numpy())
plt.title('PReLU激活函数的导数')
plt.xlabel('x')
plt.ylabel('grad of PReLU(x)')
plt.show()

# x.grad.zero_()：这是清空 x 的梯度，防止上次计算的梯度影响到这次的计算。
# y.backward(torch.ones_like(x))：执行反向传播，计算 y 对 x 的梯度。torch.ones_like(x) 是反向传播时使用的梯度（通常是损失函数的梯度）。在这里，我们用一个与 x 相同形状的全1张量来进行反向传播，目的是计算每个 x 对应的 PReLU 激活函数的导数。
# plt.plot(x.detach().numpy(), x.grad.detach().numpy())：绘制 x 和其梯度之间的关系，即 PReLU 激活函数的导数。
#
# 当 𝑥>0 时，PReLU(x)=x
#
# 当 x≤0 时，PReLU(x)=αx，其中 𝛼 是一个学习的参数，通常是一个小的正值。
#
# 导数（梯度）为：x>0 时，导数是 1
# 当 x≤0 时，导数是 𝛼
#反向传播计算梯度：y.backward(torch.ones_like(x)) 会计算 y 对 x 的梯度。由于 y 是通过 PReLU 激活函数计算得到的，所以反向传播后，x.grad 将包含 PReLU 激活函数的导数。
# 绘制梯度：通过 x.grad.detach().numpy() 获取计算出的梯度，并与 x.detach().numpy() 一起绘制，这样能展示出 PReLU 激活函数在不同输入值下的导数。
# 打印α的值
print("PReLU的参数α：", prelu.weight.item())

# -------------------------------
print("多层感知机示例")


# -------------------------------

# 定义一个多层感知机模型，包含多个隐藏层和可选的激活函数

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_function=nn.ReLU()):
        """
        初始化多层感知机模型
        参数：
        - input_size：输入层大小（特征数）
        - hidden_sizes：隐藏层大小列表，例如[64, 32]表示两个隐藏层，大小分别为64和32
        - output_size：输出层大小（类别数）
        - activation_function：激活函数，默认为ReLU
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        in_size = input_size

        # 构建隐藏层
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_size, hidden_size))
            in_size = hidden_size
        # 构建输出层
        self.out_layer = nn.Linear(in_size, output_size)
        # 激活函数
        self.activation = activation_function

    def forward(self, x):
        """
        前向传播函数
        参数：
        - x：输入张量
        返回：
        - 输出张量
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.out_layer(x)
        return x


# nn.Linear模块用于实现一个线性变换，即全连接层
# 定义为：nn.Linear(in_features, out_features, bias=True)
# 参数：
# - in_features：输入的特征数
# - out_features：输出的特征数
# - bias：是否包含偏置项，默认为True

# 激活函数可以是nn模块中的各种激活函数，例如：
# - nn.ReLU()
# - nn.Sigmoid()
# - nn.Tanh()
# - nn.PReLU()

# 在forward函数中，我们将输入依次通过隐藏层和激活函数，然后通过输出层

# 创建一个多层感知机实例
input_size = 784  # 输入层大小，例如MNIST数据集的28x28像素展开为784维
hidden_sizes = [256, 128]  # 两个隐藏层，大小分别为256和128
output_size = 10  # 输出层大小，例如10分类

# 使用ReLU激活函数
mlp_model = MLP(input_size, hidden_sizes, output_size, activation_function=nn.ReLU())

print("多层感知机模型结构：")
print(mlp_model)

# 生成随机输入数据，批量大小为64
batch_size = 64
x = torch.randn(batch_size, input_size)

# 前向传播
output = mlp_model(x)

print("模型输出的形状：", output.shape)

# 使用Sigmoid激活函数
mlp_model_sigmoid = MLP(input_size, hidden_sizes, output_size, activation_function=nn.Sigmoid())

print("使用Sigmoid激活函数的多层感知机模型结构：")
print(mlp_model_sigmoid)

# 前向传播
output_sigmoid = mlp_model_sigmoid(x)

print("使用Sigmoid激活函数的模型输出形状：", output_sigmoid.shape)

# -------------------------------
print("练习1：计算PReLU激活函数的导数")
# -------------------------------

# PReLU激活函数定义为：
# PReLU(x) = max(0, x) + α * min(0, x)

# 它的导数为：
# 当x > 0时，导数为1
# 当x < 0时，导数为α
# 当x = 0时，导数未定义，通常取1或α

# 我们可以通过绘制导数来验证

# 获取α的值
alpha = prelu.weight.item()

# 计算导数
x_np = x.detach().numpy()
grad_prelu = np.ones_like(x_np)
grad_prelu[x_np < 0] = alpha

# 绘制PReLU激活函数的导数
plt.figure(figsize=(5, 2.5))
plt.plot(x_np.flatten(), grad_prelu.flatten())
plt.title('PReLU激活函数的导数 (手动计算)')
plt.xlabel('x')
plt.ylabel('grad of PReLU(x)')
plt.show()

# -------------------------------
print("练习2：证明使用ReLU的多层感知机构造了一个连续的分段线性函数")


# -------------------------------

# 定义一个简单的MLP，输入为1维，输出为1维
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


# 实例化模型
simple_mlp = SimpleMLP()

# 生成输入范围
x_input = torch.unsqueeze(torch.linspace(-5, 5, 200), dim=1)

# 前向传播
y_output = simple_mlp(x_input)

# 绘制输出函数
plt.figure(figsize=(5, 2.5))
plt.plot(x_input.detach().numpy(), y_output.detach().numpy())
plt.title('使用ReLU的MLP输出')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 由于ReLU和线性层的组合，输出函数是连续的分段线性函数

# -------------------------------
print("练习3：证明 tanh(x) + 1 = 2 * sigmoid(2x)")
# -------------------------------

x_values = torch.linspace(-5, 5, 100)

lhs = torch.tanh(x_values) + 1
rhs = 2 * torch.sigmoid(2 * x_values)

# 绘制比较图
plt.figure(figsize=(5, 2.5))
plt.plot(x_values.detach().numpy(), lhs.detach().numpy(), label='tanh(x) + 1')
plt.plot(x_values.detach().numpy(), rhs.detach().numpy(), label='2 * sigmoid(2x)', linestyle='--')
plt.title('tanh(x) + 1 与 2 * sigmoid(2x) 比较')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 验证两者的差异
difference = torch.abs(lhs - rhs)
print("最大差异：", torch.max(difference).item())

# -------------------------------
print("练习4：非线性单元一次应用于一个小批量数据可能导致的问题")


# -------------------------------

# 定义一个非线性函数，错误地作用于整个批量数据
def nonlinear_batch_function(x):
    # 例如，对整个批量计算softmax
    return torch.softmax(x, dim=0)


# 生成一个小批量数据，形状为(batch_size, features)
batch_size = 4
features = 2
x_batch = torch.randn(batch_size, features)

# 应用非线性函数
y_batch = nonlinear_batch_function(x_batch)

print("输入x_batch：")
print(x_batch)
print("输出y_batch：")
print(y_batch)

# 问题在于，非线性函数作用在整个批量上，会导致样本间的信息混合，破坏了样本的独立性

# 总结：
# 本代码示例中使用了以下函数和模块：

# 1. torch.nn.functional.relu(input, inplace=False)
#    - 计算ReLU激活函数
#    - 参数：
#      - input：输入张量
#      - inplace：是否进行原地操作，默认为False
#    - 示例：
#      y = F.relu(x)

# 2. torch.sigmoid(input)
#    - 计算Sigmoid激活函数
#    - 参数：
#      - input：输入张量
#    - 示例：
#      y = torch.sigmoid(x)

# 3. torch.tanh(input)
#    - 计算Tanh激活函数
#    - 参数：
#      - input：输入张量
#    - 示例：
#      y = torch.tanh(x)

# 4. torch.nn.PReLU(num_parameters=1, init=0.25)
#    - 定义PReLU激活函数
#    - 参数：
#      - num_parameters：α参数的数量，可以是1或输入通道数
#      - init：α的初始值，默认为0.25
#    - 示例：
#      prelu = nn.PReLU(num_parameters=1, init=0.25)
#      y = prelu(x)

# 5. nn.Linear(in_features, out_features, bias=True)
#    - 定义全连接层（线性变换）
#    - 参数：
#      - in_features：输入的特征数
#      - out_features：输出的特征数
#      - bias：是否包含偏置项，默认为True
#    - 示例：
#      linear = nn.Linear(10, 5)
#      y = linear(x)

# 6. torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False)
#    - 计算梯度
#    - 参数：
#      - tensors：需要计算梯度的张量
#      - grad_tensors：关于每个元素的梯度权重，默认为1
#    - 示例：
#      y.backward(torch.ones_like(x))

# 7. torch.softmax(input, dim=None)
#    - 计算Softmax函数
#    - 参数：
#      - input：输入张量
#      - dim：计算Softmax的维度
#    - 示例：
#      y = torch.softmax(x, dim=1)

# 本代码示例展示了多种激活函数的使用方法、其导数的计算以及在多层感知机中的应用。
# 通过练习，深入理解了PReLU激活函数的导数计算，证明了使用ReLU的多层感知机构造了一个连续的分段线性函数，
# 验证了tanh和sigmoid函数之间的关系，并说明了非线性函数错误地作用于整个批量数据可能导致的问题。
