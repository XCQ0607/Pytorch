print("5.4. 自定义层")

# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 定义一个不带参数的自定义层，例如一个标准化层，将输入除以其标准差

class StandardizeLayer(nn.Module):
    """
    自定义层：标准化层，将输入除以其标准差
    """
    def __init__(self):
        super(StandardizeLayer, self).__init__()

    def forward(self, X):
        return X / X.std()

# 测试标准化层
print("\nStandardizeLayer 示例:")
layer = StandardizeLayer()
X = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print("原始输入 X:", X)
print("标准化后的输出:", layer(X))

# -----------------------------------------------------------------------------
# 定义一个带参数的自定义层，例如一个自定义的线性层

class MyLinear(nn.Module):
    """
    自定义线性层，包含权重和偏置参数，可以指定激活函数
    """
    def __init__(self, in_features, out_features, bias=True, activation=None):
        """
        参数：
        - in_features: 输入特征数
        - out_features: 输出特征数
        - bias: 是否使用偏置，默认为 True
        - activation: 激活函数，默认为 None
        """
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.activation = activation

    def forward(self, X):
        X = torch.matmul(X, self.weight)
        if self.bias is not None:
            X += self.bias
        if self.activation:
            X = self.activation(X)
        return X

# 测试自定义线性层
print("\nMyLinear 示例:")
input_dim = 4
output_dim = 3
linear_layer = MyLinear(input_dim, output_dim, activation=F.relu)
X = torch.randn(2, input_dim)
print("输入 X:", X)
print("线性层输出:", linear_layer(X))

# -----------------------------------------------------------------------------
# 定义一个更复杂的层，例如实现练习中的层，计算 y_k = sum_{i,j} W_{ijk} x_i x_j

class BilinearLayer(nn.Module):
    """
    自定义层：计算 y_k = sum_{i,j} W_{ijk} x_i x_j
    """
    def __init__(self, input_dim, output_dim):
        """
        参数：
        - input_dim: 输入特征数
        - output_dim: 输出特征数
        """
        super(BilinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim, output_dim))

    def forward(self, X):
        # X 的形状：(batch_size, input_dim)
        batch_size = X.size(0)
        # 计算 x_i * x_j 的组合
        X_expanded = X.unsqueeze(2)  # (batch_size, input_dim, 1)
        X_expanded_t = X.unsqueeze(1)  # (batch_size, 1, input_dim)
        X_product = X_expanded * X_expanded_t  # (batch_size, input_dim, input_dim)
        # 将 X_product 与权重张量相乘并对 i, j 求和
        Y = torch.einsum('bij,ijk->bk', X_product, self.weight)
        return Y

# 测试 BilinearLayer
print("\nBilinearLayer 示例:")
input_dim = 3
output_dim = 2
bilinear_layer = BilinearLayer(input_dim, output_dim)
X = torch.randn(4, input_dim)
print("输入 X:", X)
print("双线性层输出 Y:", bilinear_layer(X))

# -----------------------------------------------------------------------------
# 定义一个返回输入数据傅立叶系数前半部分的层

class FourierLayer(nn.Module):
    """
    自定义层：返回输入数据傅立叶变换的前半部分系数
    """
    def __init__(self):
        super(FourierLayer, self).__init__()

    def forward(self, X):
        # 对输入进行傅立叶变换
        X_fft = torch.fft.fft(X)
        # 只取前一半的系数
        N = X_fft.size(-1)
        X_fft_half = X_fft[..., :N//2]
        return X_fft_half

# 测试 FourierLayer
print("\nFourierLayer 示例:")
X = torch.randn(2, 8)
fourier_layer = FourierLayer()
print("输入 X:", X)
print("傅立叶层输出:", fourier_layer(X))

# -----------------------------------------------------------------------------
# 在模型中使用自定义层

# 定义一个包含多个自定义层的模型
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = MyLinear(8, 16, activation=F.relu)
        self.layer2 = StandardizeLayer()
        self.layer3 = BilinearLayer(16, 4)
        self.layer4 = FourierLayer()

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        return X

# 测试模型
print("\nCustomModel 示例:")
model = CustomModel()
X = torch.randn(2, 8)
print("模型输入 X:", X)
print("模型输出:", model(X))

# -----------------------------------------------------------------------------
# 总结：
# 在上述代码示例中，我们定义了多个自定义层，并将其组合到一个模型中。

# 使用到的函数和方法：
# 1. torch.nn.Module: 所有神经网络模块的基类，用户自定义的层需要继承它。
# 2. torch.nn.Parameter: 参数类，封装张量，使其在模型训练中被视为参数。
#    - 传入参数为张量，用于初始化参数的值。
# 3. torch.matmul(input, other): 矩阵乘法，计算两个张量的矩阵乘法。
#    - input: 第一个张量。
#    - other: 第二个张量。
# 4. torch.einsum(equation, operands): 爱因斯坦求和约定，可以方便地进行张量的各种操作。
#    - equation: 描述操作的字符串，例如 'bij,ijk->bk'。
#    - operands: 参与计算的张量列表。
# 5. torch.fft.fft(input): 对输入张量进行一维快速傅立叶变换。
#    - input: 输入张量。

# 调用示例：
# - 创建自定义线性层：
#   linear_layer = MyLinear(in_features=4, out_features=3, bias=True, activation=F.relu)
# - 前向传播：
#   output = linear_layer(input_tensor)

# 通过这些示例，我们展示了如何定义不带参数和带参数的自定义层，以及如何在模型中使用它们。
