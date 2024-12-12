import torch
from torch import nn

#============================================================
# 打印章节名称及说明
print("6.2. 图像卷积")
print("本示例代码将演示如何在PyTorch中实现二维卷积层的各种功能，")
print("包括：")
print("1. 手动实现二维互相关运算函数。")
print("2. 基于该函数构建简单的卷积层。")
print("3. 利用卷积核进行边缘检测。")
print("4. 学习卷积核参数，从输入映射到输出。")
print("5. 展示卷积层中参数(卷积核)的训练过程。")
print("6. 展示如何将二维卷积运算表示为矩阵乘法。")
print("7. 回答并展示文档中的问题与思考。")

print("\n" + "="*60)
print("【函数定义区】")
print("在这里我们将定义用于2D互相关操作的函数corr2d，并构建我们的Conv2D类。")
print("corr2d函数用于对输入张量X和卷积核K进行2D互相关运算。")
print("Conv2D类则模拟PyTorch中的卷积层，可自定义卷积核大小，并在forward中调用corr2d。")

def corr2d(X, K):
    """
    二维互相关运算函数
    参数:
        X (torch.Tensor): 输入的二维张量，形状为(H_in, W_in)
        K (torch.Tensor): 卷积核张量，形状为(H_k, W_k)
    返回:
        Y (torch.Tensor): 输出的二维张量，形状为(H_out, W_out)
                          其中 H_out = H_in - H_k + 1
                               W_out = W_in - W_k + 1
    """
    h, w = K.shape  # 获取卷积核的高度和宽度
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))   # 初始化输出张量
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()    # 逐元素相乘并求和
    return Y

"""
二维互相关运算（或称卷积运算），它在计算机视觉领域中广泛用于处理图像数据。在这个函数中，X 是输入的二维张量（例如一张图像），K 是卷积核（滤波器），而输出是一个新的二维张量 Y，通常称为特征图或输出图像。

互相关运算流程：
输入和卷积核： 输入张量 X 和卷积核 K 都是二维的。卷积核通常比输入张量小，且我们将卷积核移动到输入图像的每个位置，并计算它们之间的逐元素乘积求和。
计算输出： 输出张量的每个元素是卷积核与输入张量相应区域逐元素相乘后求和的结果。
示例：
我们可以通过一个简单的例子来演示这个过程。
假设我们有以下输入张量 X 和卷积核 K：
import torch
X = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=torch.float32)
K = torch.tensor([[1, 0],
                  [0, -1]], dtype=torch.float32)
输入张量 X 是一个 3x3 的矩阵（类似一个小图像）。
卷积核 K 是一个 2x2 的矩阵，它将被滑动并与 X 进行逐元素相乘。
计算过程：
假设我们从输入张量的左上角开始，卷积核的左上角与输入的对应区域对齐，进行逐元素相乘并求和：
对于 Y[0,0]，卷积核覆盖的区域是 X[0:2, 0:2]：
[[1, 2],
 [4, 5]]
对应的逐元素相乘：
(1 * 1) + (2 * 0) + (4 * 0) + (5 * -1) = 1 + 0 + 0 - 5 = -4
所以，Y[0, 0] 的值为 -4。
对于 Y[0,1]，卷积核覆盖的区域是 X[0:2, 1:3]：
[[2, 3],
 [5, 6]]
逐元素相乘：
(2 * 1) + (3 * 0) + (5 * 0) + (6 * -1) = 2 + 0 + 0 - 6 = -4
所以，Y[0, 1] 的值为 -4。

以此类推，整个输出 Y 的计算过程如下：
输出 Y：
Y =
[[-4, -4],
 [ 4,  4]]
"""

class Conv2D(nn.Module):
    """
    自定义二维卷积层类（不使用偏置或可选择使用偏置）
    参数:
        kernel_size (tuple): 卷积核的大小，例如(2,3)表示2x3的卷积核
        use_bias (bool): 是否使用偏置项, 默认为True
    """
    def __init__(self, kernel_size, use_bias=True):
        super().__init__()  # 调用父类的初始化方法，会初始化父类nn.Module的参数--weight,bias  etc.
        self.weight = nn.Parameter(torch.randn(kernel_size))    # 初始化卷积核为随机值
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))     # 初始化偏置项为0,nn.Parameter将其转换为可学习的参数
        else:
            self.bias = None

    def forward(self, x):
        """
        前向计算，x为输入张量:
        x的shape: (H_in, W_in)，单通道情况
        返回: (H_out, W_out)
        """
        Y = corr2d(x, self.weight)  # 调用corr2d函数进行卷积运算
        if self.bias is not None:
            Y = Y + self.bias
        return Y

#============================================================
print("\n" + "="*60)
print("【基础示例】")
print("使用给定输入X和卷积核K进行2D互相关计算:")

X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0],
                  [2.0, 3.0]])

Y = corr2d(X, K)
print("输入X:\n", X)
print("卷积核K:\n", K)
print("输出Y:\n", Y)

#============================================================
print("\n" + "="*60)
print("【构建简单卷积层示例】")
print("使用Conv2D类进行演示:")

conv2d_layer = Conv2D(kernel_size=(2,2))    # 初始化卷积层
print("随机初始化的卷积核参数:\n", conv2d_layer.weight.data)
if conv2d_layer.bias is not None:
    print("随机初始化的偏置参数:\n", conv2d_layer.bias.data)

# 前向传播
Y_hat = conv2d_layer(X)
print("Conv2D层输出:\n", Y_hat)

#============================================================
print("\n" + "="*60)
print("【边缘检测示例】")
print("构造一个6x8的黑白图像，中间4列为0（黑色），其余为1（白色）。")

X_edge = torch.ones((6,8))
X_edge[:, 2:6] = 0
print("输入图像X_edge:\n", X_edge)

print("构造水平检测卷积核K_edge = [[1, -1]]，用于检测垂直边缘。")
K_edge = torch.tensor([[1.0, -1.0]])

Y_edge = corr2d(X_edge, K_edge)
print("使用K_edge对X_edge进行互相关运算:")
print("输出Y_edge:\n", Y_edge)

print("现在将输入X_edge转置后再次做同样的操作:")
Y_edge_transpose = corr2d(X_edge.t(), K_edge)
print("X_edge转置后的输出:\n", Y_edge_transpose)

print("结果表明K_edge只能检测垂直方向变化，不检测水平方向变化。")

#============================================================
print("\n" + "="*60)
print("【学习卷积核示例】")
print("我们希望通过随机初始化的卷积核来学习使得X_edge -> Y_edge的映射。")

# 使用nn.Conv2d来学习，此时需要4维数据(Batch, Channel, H, W)
# 批量大小=1, 通道数=1
X_edge_4d = X_edge.unsqueeze(0).unsqueeze(0)  # (1,1,6,8)
Y_edge_4d = Y_edge.unsqueeze(0).unsqueeze(0)  # (1,1,6,7)

conv2d_learn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,2), bias=False)  # 卷积核大小为(1,2)

# 学习率
lr = 3e-2
for i in range(10):
    Y_hat = conv2d_learn(X_edge_4d)
    l = (Y_hat - Y_edge_4d)**2  # 均方误差
    conv2d_learn.zero_grad()    # 梯度清零
    l.sum().backward()  # 反向传播
    # 更新参数
    conv2d_learn.weight.data -= lr * conv2d_learn.weight.grad    # 更新卷积核参数
    if (i+1) % 2 == 0:
        print(f"epoch {i+1}, loss {l.sum().item():.3f}")

print("学习后的卷积核参数:\n", conv2d_learn.weight.data.reshape((1,2)))

#============================================================
print("\n" + "="*60)
print("【将互相关运算表示为矩阵乘法示例】")
print("在某些场景下，我们可以将卷积操作展平为矩阵乘法。")
print("下面以一个简单示例演示。")

X_small = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])
K_small = torch.tensor([[1.0, 0.0],
                        [0.0, -1.0]])

# 常规互相关运算
Y_small = corr2d(X_small, K_small)
print("X_small:\n", X_small)
print("K_small:\n", K_small)
print("Y_small:\n", Y_small)

# 将X_small展开为能够与K_small(展平)相乘的矩阵形式
# K_small展平为向量: [1.0,0.0,0.0,-1.0]
# 对X_small进行"im2col"式展开:
# 卷积核为2x2，对X_small的卷积核滑动窗口有:
# (上左)  (上右)  (下左)  (下右)
# 次序为从左到右，从上到下:
X_cols = []
h_k, w_k = K_small.shape
for i in range(X_small.shape[0] - h_k + 1):
    for j in range(X_small.shape[1] - w_k + 1):
        patch = X_small[i:i+h_k, j:j+w_k].reshape(-1)
        X_cols.append(patch)
X_mat = torch.stack(X_cols, dim=0) # 展成一个矩阵 (H_out*W_out, h_k*w_k)

K_flat = K_small.reshape(-1)
Y_mat_mul = X_mat @ K_flat # 矩阵乘法
Y_mat_mul = Y_mat_mul.reshape(Y_small.shape)

print("X_mat(展开后的X):\n", X_mat)
print("K_flat(展开后的K):\n", K_flat)
print("通过矩阵乘法得到的Y:\n", Y_mat_mul)
print("与直接corr2d计算的Y_small是否相同?", torch.allclose(Y_mat_mul, Y_small))

#============================================================
print("\n" + "="*60)
print("【回答并演示文档中的问题】")

print("问题1: 构建一个具有对角线边缘的图像X_diagonal。")
# 构建6x8的图像，对角线位置为0，其余为1
X_diagonal = torch.ones((6,8))
for idx in range(min(X_diagonal.shape)):
    X_diagonal[idx, idx] = 0
print("X_diagonal:\n", X_diagonal)

print("问题2: 如果将K_edge应用于X_diagonal会发生什么情况？")
Y_diagonal = corr2d(X_diagonal, K_edge)
print("Y_diagonal:\n", Y_diagonal)
print("K_edge = [[1,-1]]检测水平相邻像素的变化，")
print("X_diagonal中，只有当相邻列像素不同时才会有非零输出，")
print("可以看到输出中非零元素表示在水平方向上遇到0->1或者1->0的跳变处。")

print("问题3: 如果转置X_diagonal会发生什么？")
X_diagonal_T = X_diagonal.t()
Y_diagonal_T = corr2d(X_diagonal_T, K_edge)
print("X_diagonal^T:\n", X_diagonal_T)
print("Y_diagonal_T:\n", Y_diagonal_T)
print("由于转置后对角线变化方向不同，K_edge能检测到的水平变化位置也发生变化。")

print("问题4: 如果转置K_edge会发生什么？")
K_edge_T = K_edge.t()  # 转置K_edge,变成[[1],[ -1]]
Y_diagonal_KT = corr2d(X_diagonal, K_edge_T)
print("K_edge^T:\n", K_edge_T)
print("Y_diagonal with K_edge^T:\n", Y_diagonal_KT)
print("转置后的K_edge_T=[ [1],[ -1] ]将检测垂直相邻像素的变化，而非水平变化。")

print("问题5: 在创建Conv2D自动求导时有哪些注意？")
print("在本例中，我们没有d2l包，也没有报错，主要注意数据形状匹配：")
print("    Conv2D要求输入为4维张量(B, C, H, W)。")
print("    若不匹配会报错，例如传入二维张量会报shape不匹配的错误。")

print("问题6: 如何通过改变输入张量和卷积核张量，将互相关运算表示为矩阵乘法？")
print("如上已示例，我们使用im2col的方式将输入转换为二维矩阵，将卷积核展平为向量，")
print("然后通过矩阵相乘的方式实现卷积运算。")

print("问题7: 手工设计一些卷积核。")
print("例如：检测水平边缘的核K_horizontal = [[1, -1]]")
print("检测垂直边缘的核K_vertical = [[1],[ -1]]")
print("检测对角线的核K_diag = [[1,0],[0,-1]]等。")

print("问题8: 二阶导数的核的形式是什么？")
print("一阶导数如K_edge = [[1,-1]]，二阶导数可类比为[[1,-2,1]]用于离散二阶导差，")
print("或2D情况[[0,1,0],[1,-4,1],[0,1,0]]等表示拉普拉斯算子(Laplacian)。")

print("问题9: 积分的核的形式是什么？")
print("积分相当于求和，可用核如[[1,1]]或更大的一致核，比如[[1,1],[1,1]]，进行平均即为积分平滑。")

print("问题10: 得到d次导数的最小核的大小是多少？")
print("一阶导数最小核大小为2(如[1,-1])，二阶导数为3(如[1,-2,1])。")
print("对于d阶导数，至少需要(d+1)个点进行离散近似。因此最小核的大小至少为d+1。")

#============================================================
print("\n" + "="*60)
print("【总结】")
print("在本示例中，我们实现了以下内容：")
print("1. corr2d函数：用于对输入和核进行二维互相关计算。")
print("2. Conv2D类：模拟卷积层的前向计算过程。可接受参数:")
print("   - kernel_size: tuple,指定卷积核大小，如(2,2)")
print("   - use_bias: bool,是否使用偏置项，默认True。")
print("3. 利用Conv2D类和PyTorch自带的nn.Conv2d对输入X和卷积核K进行卷积，并展示输出。")
print("4. 展示边缘检测核的使用。")
print("5. 展示通过梯度下降来学习卷积核，从而将输入X映射到目标Y。")
print("6. 展示如何将卷积运算展开为矩阵乘法(im2col)的思路。")
print("7. 回答文档中的思考问题并进行相应的代码示例。")

print("\n所有函数参数介绍与调用示例已在注释中给出。")

#============================================================
