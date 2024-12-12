# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F

# 打印目录名称
print("6.4. 多输入多输出通道")

# ---------------------------------------------------
# 6.4.1. 多输入通道
print("\n6.4.1. 多输入通道")


def corr2d_multi_in(X, K):
    """
    实现多输入通道的二维互相关运算。

    参数:
    X (Tensor): 输入张量，形状为 (c_i, h, w)，其中 c_i 是输入通道数。
    K (Tensor): 卷积核张量，形状为 (c_i, k_h, k_w)。

    返回:
    Tensor: 互相关运算的结果，形状为 (h_out, w_out)。
    """
    # 对每个输入通道执行互相关操作，并将结果相加
    return sum(F.conv2d(x.unsqueeze(0).unsqueeze(0), k.unsqueeze(0).unsqueeze(0))
               for x, k in zip(X, K)).squeeze()


# 构造示例输入张量 X 和卷积核张量 K
X = torch.tensor([
    [[0.0, 1.0, 2.0],
     [3.0, 4.0, 5.0],
     [6.0, 7.0, 8.0]],

    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]]
])

K = torch.tensor([
    [[0.0, 1.0],
     [2.0, 3.0]],

    [[1.0, 2.0],
     [3.0, 4.0]]
])

# 执行多输入通道的互相关运算
Y = corr2d_multi_in(X, K)
print("多输入通道互相关运算的输出:")
print(Y)

# ---------------------------------------------------
# 6.4.2. 多输出通道
print("\n6.4.2. 多输出通道")


def corr2d_multi_in_out(X, K):
    """
    实现多输入多输出通道的二维互相关运算。

    参数:
    X (Tensor): 输入张量，形状为 (c_i, h, w)。
    K (Tensor): 卷积核张量，形状为 (c_o, c_i, k_h, k_w)。

    返回:
    Tensor: 互相关运算的结果，形状为 (c_o, h_out, w_out)。
    """
    # 对每个输出通道执行多输入通道的互相关运算，并堆叠结果
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


# 构造具有3个输出通道的卷积核 K
K_multi_out = torch.stack((K, K + 1, K + 2), 0)
print("多输出通道卷积核 K 的形状:", K_multi_out.shape)

# 执行多输入多输出通道的互相关运算
Y_multi_out = corr2d_multi_in_out(X, K_multi_out)
print("多输入多输出通道互相关运算的输出:")
print(Y_multi_out)

# ---------------------------------------------------
# 6.4.3. 1x1 卷积层
print("\n6.4.3. 1x1 卷积层")


def corr2d_multi_in_out_1x1(X, K):
    """
    实现1x1卷积的多输入多输出通道互相关运算。

    参数:
    X (Tensor): 输入张量，形状为 (c_i, h, w)。
    K (Tensor): 卷积核张量，形状为 (c_o, c_i, 1, 1)。

    返回:
    Tensor: 互相关运算的结果，形状为 (c_o, h, w)。
    """
    c_i, h, w = X.shape
    c_o = K.shape[0]
    # 将输入张量重塑为 (c_i, h * w)
    X_reshaped = X.view(c_i, h * w)
    # 将卷积核重塑为 (c_o, c_i)
    K_reshaped = K.view(c_o, c_i)
    # 执行矩阵乘法
    Y = torch.matmul(K_reshaped, X_reshaped)
    # 重塑输出为 (c_o, h, w)
    return Y.view(c_o, h, w)


# 构造随机输入张量 X 和1x1卷积核 K_1x1
torch.manual_seed(0)  # 设置随机种子以保证结果可复现
X_random = torch.normal(0, 1, (3, 3, 3))
K_1x1 = torch.normal(0, 1, (2, 3, 1, 1))

print("输入张量 X_random:")
print(X_random)
print("1x1 卷积核 K_1x1:")
print(K_1x1)

# 执行1x1卷积运算
Y1 = corr2d_multi_in_out_1x1(X_random, K_1x1)
print("1x1 卷积运算的输出 Y1:")
print(Y1)

# 对比常规多输入多输出卷积的输出
Y2 = corr2d_multi_in_out(X_random, K_1x1)
print("多输入多输出通道互相关运算的输出 Y2:")
print(Y2)

# 验证 Y1 和 Y2 是否相同
difference = torch.abs(Y1 - Y2).sum()
print("Y1 和 Y2 的差异:", difference.item())
assert difference < 1e-6, "Y1 和 Y2 不相同！"

# ---------------------------------------------------
# 6.4.4. 小结
print("\n6.4.4. 小结")
print("""
多输入多输出通道可以用来扩展卷积层的模型。
当以每像素为基础应用时，1x1卷积层相当于全连接层。
1x1卷积层通常用于调整网络层的通道数量和控制模型复杂性。
""")

# ---------------------------------------------------
# 6.4.5. 练习
print("\n6.4.5. 练习")

# 练习1: 假设我们有两个卷积核，大小分别为k1和k2（中间没有非线性激活函数）。
# 证明运算可以用单次卷积来表示。
# 这个等效的单个卷积核的维数是多少呢？
# 反之亦然吗？

print("\n练习1: 两个卷积核合并为一个卷积核")
# 假设 k1 和 k2 是两个卷积核，形状均为 (c_i, k_h, k_w)
k1 = torch.tensor([
    [[1.0, 0.0],
     [0.0, -1.0]],

    [[0.5, 0.5],
     [0.5, 0.5]]
])

k2 = torch.tensor([
    [[-1.0, 1.0],
     [1.0, -1.0]],

    [[1.0, -1.0],
     [-1.0, 1.0]]
])

# 合并两个卷积核，形成一个新的卷积核，形状为 (2, c_i, k_h, k_w)
K_combined = torch.stack((k1, k2), 0)
print("合并后的卷积核 K_combined 的形状:", K_combined.shape)

# 输入张量 X
X_ex = torch.tensor([
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]],

    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]]
])

# 执行单次卷积运算
Y_combined = corr2d_multi_in_out(X_ex, K_combined)
print("合并后卷积运算的输出 Y_combined:")
print(Y_combined)

# 反之亦然，无法通过单次卷积分解回两个独立的卷积运算
print("反向操作无法直接从合并的卷积核中分离出原始的两个卷积核。")

# 练习2: 假设输入为c_i×h×w，卷积核大小为c_o×c_i×k_h×k_w，填充为(p_h, p_w)，步幅为(s_h, s_w)。
# 前向传播的计算成本（乘法和加法）是多少？

print("\n练习2: 计算前向传播的计算成本")
c_i, c_o, k_h, k_w = 3, 4, 5, 5
h, w = 32, 32
p_h, p_w = 2, 2
s_h, s_w = 1, 1

# 输出的高度和宽度计算
h_out = (h + 2 * p_h - k_h) // s_h + 1
w_out = (w + 2 * p_w - k_w) // s_w + 1

# 每个输出元素的计算成本
cost_per_output = c_i * k_h * k_w * 2  # 乘法和加法

# 总的输出元素数量
num_outputs = c_o * h_out * w_out

# 总的计算成本
total_cost = num_outputs * c_i * k_h * k_w * 2

print(f"前向传播的计算成本: {total_cost} 次乘加运算")

# 练习3: 内存占用是多少？
print("\n练习3: 内存占用计算")
# 卷积核参数数量
num_params = c_o * c_i * k_h * k_w
# 假设每个参数为32位浮点数，即4字节
memory_params = num_params * 4  # 字节

# 输入和输出的内存占用
memory_input = c_i * h * w * 4
memory_output = c_o * h_out * w_out * 4

print(f"卷积核参数内存占用: {memory_params} 字节")
print(f"输入内存占用: {memory_input} 字节")
print(f"输出内存占用: {memory_output} 字节")

# 练习4: 反向传播的内存占用和计算成本
print("\n练习4: 反向传播的内存占用和计算成本")
# 反向传播需要存储输入、输出和中间梯度
memory_backward = memory_input + memory_output + memory_params
print(f"反向传播的内存占用: {memory_backward} 字节")

# 反向传播的计算成本大约是前向传播的两倍
total_cost_backward = total_cost * 2
print(f"反向传播的计算成本: {total_cost_backward} 次乘加运算")

# 练习5: 加倍输入和输出通道数量
print("\n练习5: 加倍输入和输出通道数量的计算数量变化")
c_i_new, c_o_new = c_i * 2, c_o * 2
total_cost_new = (c_i_new * k_h * k_w * 2) * (c_o_new * h_out * w_out)
print(f"加倍后的前向传播计算成本: {total_cost_new} 次乘加运算")
print("计算数量将增加四倍。")

# 练习6: 加倍填充数量
print("\n练习6: 加倍填充数量的影响")
p_h_new, p_w_new = p_h * 2, p_w * 2
h_out_new = (h + 2 * p_h_new - k_h) // s_h + 1
w_out_new = (w + 2 * p_w_new - k_w) // s_w + 1
num_outputs_new = c_o * h_out_new * w_out_new
total_cost_padded = num_outputs_new * c_i * k_h * k_w * 2
print(f"加倍填充后的前向传播计算成本: {total_cost_padded} 次乘加运算")

# 练习7: 1x1 卷积的计算复杂度
print("\n练习7: 1x1 卷积的前向传播计算复杂度")
k_h_1x1, k_w_1x1 = 1, 1
h_out_1x1 = (h + 2 * p_h - k_h_1x1) // s_h + 1
w_out_1x1 = (w + 2 * p_w - k_w_1x1) // s_w + 1
total_cost_1x1 = c_o * c_i * h_out_1x1 * w_out_1x1 * k_h_1x1 * k_w_1x1 * 2
print(f"1x1 卷积的前向传播计算复杂度: {total_cost_1x1} 次乘加运算")

# 练习8: Y1 和 Y2 是否完全相同？为什么？
print("\n练习8: Y1 和 Y2 是否完全相同？")
print("是的，Y1 和 Y2 完全相同，因为 corr2d_multi_in_out_1x1 和 corr2d_multi_in_out 实现了相同的1x1卷积运算。")
print("这是因为1x1卷积可以通过矩阵乘法来实现，两个方法最终得到的结果一致。")

# 练习9: 当卷积窗口不是1x1时，如何使用矩阵乘法实现卷积？
print("\n练习9: 使用矩阵乘法实现非1x1卷积")
print("""
当卷积窗口不是1x1时，可以通过将输入张量展开（im2col方法）为二维矩阵，
然后将卷积核展开为二维矩阵，执行矩阵乘法，最后将结果重新形状为输出张量。
这种方法可以高效地利用矩阵乘法的优化。
""")

# ---------------------------------------------------
# 代码示例总结
print("\n代码示例总结")
print("""
本代码示例演示了如何在PyTorch中实现多输入多输出通道的二维互相关运算，包括1x1卷积层的实现。
主要使用的函数和方法包括：

1. torch.tensor: 创建张量。
2. torch.stack: 将多个张量沿新维度堆叠。
3. F.conv2d: 执行二维卷积运算。
4. torch.matmul: 执行矩阵乘法。
5. tensor.view: 重塑张量形状。
6. tensor.unsqueeze: 增加维度以适应卷积运算的输入要求。

各函数的参数介绍：
- torch.tensor(data): 创建一个新的张量。
- torch.stack(tensors, dim=0): 将给定序列的张量沿指定维度连接。
- F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1): 执行二维卷积。
  - input: 输入张量，形状为 (batch_size, channels, height, width)。
  - weight: 卷积核张量，形状为 (out_channels, in_channels, kernel_height, kernel_width)。
  - stride: 卷积的步幅。
  - padding: 输入张量的每一条边补充0的层数。
- torch.matmul(a, b): 矩阵乘法。
- tensor.view(shape): 重塑张量的形状。
- tensor.unsqueeze(dim): 在指定位置增加一个维度。

调用示例：
- corr2d_multi_in(X, K): 对多输入通道进行互相关运算。
- corr2d_multi_in_out(X, K): 对多输入多输出通道进行互相关运算。
- corr2d_multi_in_out_1x1(X, K): 实现1x1卷积的多输入多输出通道互相关运算。

整个代码示例通过逐步构建和验证多输入多输出通道的卷积运算，加深了对卷积层参数和计算过程的理解。
""")
