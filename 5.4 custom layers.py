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
        return X / X.std()    #将输入除以其标准差
""""
将输入除以其标准差是一种常见的标准化操作，这种操作在数据预处理和神经网络设计中非常重要，具有以下几个含义和用途：
数据标准化：通过将数据除以其标准差，可以使数据具有单位方差（方差为1），这有助于消除不同维度之间的量纲差异，使得每个特征在模型训练过程中具有相似的权重影响。这种处理对于很多机器学习算法来说是非常有益的，因为它可以加快收敛速度，提高模型的泛化能力。
稳定性：在深度学习中，输入数据的分布变化（即所谓的内部协变量偏移）会对模型的训练过程产生不利影响。通过标准化处理，可以减少这种分布变化，使得模型的训练过程更加稳定。
加速训练：在训练神经网络时，如果输入数据的分布变化较大，那么模型需要不断地适应这种变化，这会导致训练速度变慢。通过标准化处理，可以使得输入数据的分布相对稳定，从而加速模型的训练过程。
避免梯度消失或爆炸：在深度神经网络中，如果输入数据的范围差异很大，那么在反向传播过程中，可能会出现梯度消失或梯度爆炸的问题。通过标准化处理，可以将输入数据缩放到一个相对较小的范围内，有助于缓解这个问题。
""""


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
        self.weight = nn.Parameter(torch.randn(in_features, out_features))  #Parameter 是一个特殊的张量，它默认要求梯度
        #正常的张量为nn.Tensor，不需要梯度,如nn.Tensor(torch.randn(in_features, out_features)),但也可以指定需要梯度，则需要nn.Tensor(torch.randn(in_features, out_features), requires_grad=True)
        '''
        nn.Parameter 是 PyTorch 中用于表示模型参数的类。
        当创建一个 nn.Parameter 对象时，它默认是需要计算梯度的（即 requires_grad=True）。这是因为在训练神经网络时，我们需要对这些参数进行梯度下降等优化操作。
        '''
        if bias:    #如果需要偏置
            self.bias = nn.Parameter(torch.randn(out_features))    #如果需要偏置，则创建一个偏置参数
        else:    #如果不需要偏置
            self.register_parameter('bias', None)    #如果不需要偏置，则将bias设置为None,register_parameter()是一个方法，用于将一个参数添加到模型中，使其可以被训练，也可以使用其他优化器进行优化。
        self.activation = activation    #这个要在forward()中调用activation的值以确定是否使用激活函数，所以把这个赋值给self.activation，在forward()中调用self.activation(X)即可使用激活函数

    def forward(self, X):
        X = torch.matmul(X, self.weight)    #X是输入，self.weight是权重，torch.matmul()是矩阵乘法，torch.matmul(X, self.weight)是X乘以self.weight
        if self.bias is not None:    #如果需要偏置
            X += self.bias    #如果需要偏置，则将偏置加到X上
        if self.activation:    #如果需要激活函数
            X = self.activation(X)    #如果需要激活函数，则使用激活函数,如F.relu(X)是ReLU激活函数,F是torch.nn.functional中的函数
        return X    #返回X

# 测试自定义线性层
print("\nMyLinear 示例:")
input_dim = 4    #输入特征数
output_dim = 3    #输出特征数
linear_layer = MyLinear(input_dim, output_dim, activation=F.relu)    #创建一个自定义线性层，输入特征数为4，输出特征数为3，激活函数为ReLU
X = torch.randn(2, input_dim)    #创建一个随机输入，形状为(2, 4)
print("输入 X:", X)    #创建一个随机输入，形状为(2, 4)
print("线性层输出:", linear_layer(X))    #输出线性层的输出

# -----------------------------------------------------------------------------
# 定义一个更复杂的层，例如实现练习中的层，计算 y_k = sum_{i,j}(W_{ijk}·x_i·x_j)   # 其中 i, j 是输入特征的索引，k 是输出特征的索引。W_{ijk} 是权重张量，x_i 是输入特征，x_j 是输入特征，y_k 是输出特征

#实现一个双线性层（Bilinear Layer），该层根据给定的公式 y_k = sum_{i,j}(W_{ijk}·x_i·x_j) 计算输出特征 y_k。
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
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim, output_dim))   #创建一个权重张量，形状为(input_dim, input_dim, output_dim)

    def forward(self, X):
        # X 的形状：(batch_size, input_dim)
        batch_size = X.size(0)  # 获取批处理大小，X.size返回一个元组
        #size() 方法返回的元组将包含两个元素：第一个元素是矩阵的行数（对应于批处理大小 batch_size），第二个元素是矩阵的列数（对应于特征维度 input_dim）。
        # 计算 x_i * x_j 的组合
        X_expanded = X.unsqueeze(2)  # (batch_size, input_dim, 1)   #unsqueeze() 方法在指定维度上增加一个维度，unsqueeze(2) 在第3个维度上增加一个维度，这样就可以将 X 扩展为 (batch_size, input_dim, 1) 的形状，以便进行矩阵乘法。
        #这里的“第2维度”是一个容易让人混淆的说法，因为实际上你是在第3个位置（从0开始计数）插入了一个新的维度。更准确的描述应该是“在第3个位置（索引为2）插入了一个新的维度”。
        X_expanded_t = X.unsqueeze(1)  # (batch_size, 1, input_dim)
        X_product = X_expanded * X_expanded_t  # (batch_size, input_dim, input_dim)
        #矩阵乘法的输出矩阵shape决定:A*B矩阵。
        # 将 X_product 与权重张量相乘并对 i, j 求和
        Y = torch.einsum('bij,ijk->bk', X_product, self.weight) #(batch_size, input_dim, input_dim)-einsum-(input_dim, input_dim, output_dim)->(batch_size, output_dim)
        #这里的字符串意思是：bij,ijk->bk，表示将 X_product 的形状 (batch_size, input_dim, input_dim) 与 self.weight 的形状 (input_dim, input_dim, output_dim) 进行矩阵乘法，得到 Y 的形状 (batch_size, output_dim)。
        #bij指的是X_product的形状，ijk指的是self.weight的形状，bk指的是Y的形状。
        return Y



# 测试 BilinearLayer(双线性层)
print("\nBilinearLayer 示例:")
input_dim = 3
output_dim = 2
bilinear_layer = BilinearLayer(input_dim, output_dim)
X = torch.randn(4, input_dim)   #一次4个样本，每个样本有3个特征
print("输入 X:", X)
print("双线性层输出 Y:", bilinear_layer(X))

# -----------------------------------------------------------------------------
# 定义一个返回输入数据傅立叶系数前半部分的层

class FourierLayer(nn.Module):  #傅立叶变换层
    """
    自定义层：返回输入数据傅立叶变换的前半部分系数
    """
    def __init__(self):
        super(FourierLayer, self).__init__()

    def forward(self, X):
        # 对输入进行傅立叶变换
        X_fft = torch.fft.fft(X)    #对输入进行傅立叶变换
        # 只取前一半的系数
        N = X_fft.size(-1)  #size(-1)返回最后一个维度的大小，即输入数据的特征数（列数）
        X_fft_half = X_fft[..., :N//2]   #只取前一半的系数
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
        self.layer1 = MyLinear(8, 16, activation=F.relu)    #创建一个自定义线性层，输入特征数为8，输出特征数为16，激活函数为ReLU
        self.layer2 = StandardizeLayer()    #创建一个标准化层
        self.layer3 = BilinearLayer(16, 4)    #创建一个双线性层，输入特征数为16，输出特征数为4
        self.layer4 = FourierLayer()    #创建一个傅立叶层

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
