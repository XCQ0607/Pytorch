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
    # return sum(F.conv2d(x.unsqueeze(0).unsqueeze(0), k.unsqueeze(0).unsqueeze(0))   #两个.unsqueeze(0)目的是什么？
    #            for x, k in zip(X, K)).squeeze() #zip()函数用于将两个可迭代对象的对应元素打包成元组,.squeeze()函数用于移除张量中维度为1的维度
# --------------------
    # 1.
    # return sum(
    #     F.conv2d(
    #         x.unsqueeze(0).unsqueeze(0),  # 添加批次和通道维度，形状变为 (1, 1, h, w)
    #         k.unsqueeze(0).unsqueeze(0)   # 添加输出和输入通道维度，形状变为 (1, 1, k_h, k_w)
    #     )
    #     for x, k in zip(X, K)
    # ).squeeze()  # 移除批次和通道维度，返回二维张量 (h_out, w_out)
    # 2.
    # return F.conv2d(X.unsqueeze(0), K.unsqueeze(0)).squeeze(0)
    # 3.
    # 添加批次维度
    X = X.unsqueeze(0)  # (1, c_i, h, w)
    # 添加输出通道维度
    K = K.unsqueeze(0)  # (1, c_i, k_h, k_w)
    # 执行卷积
    Y = F.conv2d(X, K)  # (1, 1, h_out, w_out)  ,参数X是conv2d的输入，参数K是卷积核,
    # 移除批次和通道维度
    return Y.squeeze(0).squeeze(0)  # (h_out, w_out)
# ----------------------
#torch.nn.functional.conv2d 函数期望的输入张量格式如下：
# 输入张量 input: 形状为 (batch_size, in_channels, height, width)
# 卷积核张量 weight: 形状为 (out_channels, in_channels, kernel_height, kernel_width)
# 这意味着，conv2d 需要知道每个输入样本的批次大小（batch_size）、输入通道数（in_channels）以及输出通道数（out_channels）。
# 2. 当前张量的维度
# 在您的代码中：
# 输入张量 X 的形状为 (c_i, h, w)，其中 c_i 是输入通道数。
# 卷积核张量 K 的形状为 (c_i, k_h, k_w)，对应每个输入通道的卷积核。
# 3. 为什么需要 unsqueeze(0) 两次？
# x 和 k 在 zip(X, K) 中分别对应单个输入通道的数据和单个通道的卷积核。具体来说：
# x 的形状: (h, w)
# k 的形状: (k_h, k_w)
# 为了使用 F.conv2d，我们需要将它们调整为包含批次和通道维度的四维张量：
# 添加批次维度: unsqueeze(0) 将 x 从 (h, w) 变为 (1, h, w)，表示批次大小为1。
# 添加通道维度: 再次 unsqueeze(0) 将 (1, h, w) 变为 (1, 1, h, w)，表示输入通道数为1。
# 对于卷积核 k：
# 添加输出通道维度: unsqueeze(0) 将 k 从 (k_h, k_w) 变为 (1, k_h, k_w)，表示输出通道数为1。
# 添加输入通道维度: 再次 unsqueeze(0) 将 (1, k_h, k_w) 变为 (1, 1, k_h, k_w)，表示输入通道数为1。


# 构造示例输入张量 X 和卷积核张量 K
X = torch.tensor([
    [[0.0, 1.0, 2.0],
     [3.0, 4.0, 5.0],
     [6.0, 7.0, 8.0]],

    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]]
])  #shape: (2, 3, 3)

K = torch.tensor([
    [[0.0, 1.0],
     [2.0, 3.0]],

    [[1.0, 2.0],
     [3.0, 4.0]]
])  #shape: (2, 2, 2)

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
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)   #stack()函数用于将多个张量沿指定维度堆叠


# 构造具有3个输出通道的卷积核 K
K_multi_out = torch.stack((K, K + 1, K + 2), 0) #shape: (3, 2, 2, 2),(k,k+1,k+2)指的是stack的三个张量，0指的是在第0维度上堆叠
print("多输出通道卷积核 K 的形状:", K_multi_out.shape) #原本K的shape是(2,2,2)，现在变成了(3,2,2,2)

# 执行多输入多输出通道的互相关运算
Y_multi_out = corr2d_multi_in_out(X, K_multi_out)   #X与K_multi_out中的每个卷积核进行互相关运算
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
    K (Tensor): 卷积核张量，形状为 (c_o, c_i, 1, 1)。 1x1卷积核用于将每个通道的特征图进行线性变换。
    c_0: 输出通道数。
    c_i: 输入通道数。

    返回:
    Tensor: 互相关运算的结果，形状为 (c_o, h, w)。
    """
    c_i, h, w = X.shape    #输入通道数，高，宽
    c_o = K.shape[0]    #输出通道数
    # 将输入张量重塑为 (c_i, h * w)
    X_reshaped = X.view(c_i, h * w) #.view()函数用于重塑张量的形状，变为2维张量，第一维为通道数，第二维为像素点个数
    # 将卷积核重塑为 (c_o, c_i)
    K_reshaped = K.view(c_o, c_i)   #变为2维张量，第一维为通道数，第二维为像素点个数
    # 执行矩阵乘法
    Y = torch.matmul(K_reshaped, X_reshaped)
    # 重塑输出为 (c_o, h, w)
    return Y.view(c_o, h, w)       #.view()函数用于重塑张量的形状，变为2维张量，第一维为通道数，第二维为像素点个数


# 构造随机输入张量 X 和1x1卷积核 K_1x1
torch.manual_seed(0)  # 设置随机种子以保证结果可复现
X_random = torch.normal(0, 1, (3, 3, 3))    #(3,3,3)指的是生成3个3x3的输入张量 #3个输入通道
K_1x1 = torch.normal(0, 1, (2, 3, 1, 1))    #normal()函数用于生成服从正态分布的随机数,0为均值,1为标准差,(2,3,1,1)指的是生成2个3通道的1x1卷积核 #2个输出通道

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
p_h, p_w = 2, 2  # 填充
s_h, s_w = 1, 1  # 步幅

# 输出的高度和宽度计算
h_out = (h + 2 * p_h - k_h) // s_h + 1
w_out = (w + 2 * p_w - k_w) // s_w + 1

# 每个输出元素的计算成本
cost_per_output = c_i * k_h * k_w * 2  #*2是因为有两个输入通道

# 总的输出元素数量
num_outputs = c_o * h_out * w_out

# 总的计算成本
total_cost = num_outputs * cost_per_output

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
total_cost_backward = total_cost * 2    # 反向传播的计算成本是前向传播的两倍
print(f"反向传播的计算成本: {total_cost_backward} 次乘加运算")

# 练习5: 加倍输入和输出通道数量
print("\n练习5: 加倍输入和输出通道数量的计算数量变化")
c_i_new, c_o_new = c_i * 2, c_o * 2
total_cost_new = (c_i_new * k_h * k_w * 2) * (c_o_new * h_out * w_out)
print(f"加倍后的前向传播计算成本: {total_cost_new} 次乘加运算")
print("计算数量将增加四倍。")

# 练习6: 加倍填充数量
print("\n练习6: 加倍填充数量的影响")
p_h_new, p_w_new = p_h * 2, p_w * 2 # 加倍填充
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
