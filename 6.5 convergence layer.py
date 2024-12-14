# 6.5. 汇聚层
print("6.5. 汇聚层\n")

import torch
from torch import nn
import torch.nn.functional as F

# 6.5.1. 最大汇聚层和平均汇聚层
print("6.5.1. 最大汇聚层和平均汇聚层\n")

# 定义一个2D张量X
X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])

print("输入张量X:")
print(X)

# 实现pool2d函数，用于前向传播
def pool2d(X, pool_size, mode='max'):
    """
    对输入张量X应用2D汇聚操作。

    参数：
    - X (torch.Tensor): 输入张量，形状为 (高度, 宽度)。
    - pool_size (tuple): 汇聚窗口的大小 (p_h, p_w)。
    - mode (str): 汇聚模式，'max' 或 'avg'。

    返回：
    - Y (torch.Tensor): 汇聚后的输出张量。
    """
    p_h, p_w = pool_size
    out_h = (X.shape[0] - p_h) // 1 + 1
    out_w = (X.shape[1] - p_w) // 1 + 1
    Y = torch.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = X[i: i + p_h, j: j + p_w]
            if mode == 'max':
                Y[i, j] = window.max()
            elif mode == 'avg':
                Y[i, j] = window.mean()
    return Y

# 使用2x2的最大汇聚
print("2x2 最大汇聚层的输出:")
Y_max = pool2d(X, (2, 2), mode='max')
print(Y_max)

# 使用2x2的平均汇聚
print("\n2x2 平均汇聚层的输出:")
Y_avg = pool2d(X, (2, 2), mode='avg')
print(Y_avg)

print("\n" + "-"*50 + "\n")

# 使用PyTorch内置的最大汇聚层和平均汇聚层
print("使用PyTorch内置的汇聚层\n")

# 构建输入张量，添加批量大小和通道维度
X_pytorch = torch.tensor([[[[0., 1., 2.],
                            [3., 4., 5.],
                            [6., 7., 8.]]]])

print("输入张量X_pytorch:")
print(X_pytorch)

# 定义最大汇聚层和平均汇聚层
max_pool = nn.MaxPool2d(kernel_size=2, stride=1)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=1)

# 应用最大汇聚层
Y_max_pytorch = max_pool(X_pytorch)
print("\nPyTorch 最大汇聚层的输出:")
print(Y_max_pytorch)

# 应用平均汇聚层
Y_avg_pytorch = avg_pool(X_pytorch)
print("\nPyTorch 平均汇聚层的输出:")
print(Y_avg_pytorch)

print("\n" + "-"*50 + "\n")

# 6.5.2. 填充和步幅
print("6.5.2. 填充和步幅\n")

# 定义一个4x4的输入张量，带有批量大小和通道数
X_pad_stride = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print("输入张量X_pad_stride:")
print(X_pad_stride)

# 定义默认步幅的3x3最大汇聚层
pool_default = nn.MaxPool2d(kernel_size=3)
Y_default = pool_default(X_pad_stride)
print("\n默认步幅的3x3 最大汇聚层的输出:")
print(Y_default)

# 定义带有填充和步幅的3x3最大汇聚层
pool_custom = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
Y_custom = pool_custom(X_pad_stride)
print("\n带填充=1和步幅=2的3x3 最大汇聚层的输出:")
print(Y_custom)

# 定义一个任意大小的矩形汇聚窗口，并分别设定填充和步幅
pool_rect = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3), padding=(0, 1))
Y_rect = pool_rect(X_pad_stride)
print("\n矩形窗口 (2x3)，填充=(0,1)，步幅=(2,3) 的最大汇聚层的输出:")
print(Y_rect)

print("\n" + "-"*50 + "\n")

# 6.5.3. 多个通道
print("6.5.3. 多个通道\n")

# 构建具有2个通道的输入张量
X_multi = torch.cat((X_pad_stride, X_pad_stride + 1), 1)
print("多通道输入张量X_multi:")
print(X_multi)

# 应用带填充和步幅的最大汇聚层
pool_multi = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
Y_multi = pool_multi(X_multi)
print("\n多通道最大汇聚层的输出:")
print(Y_multi)

print("\n" + "-"*50 + "\n")

# 6.5.4. 小结
print("6.5.4. 小结\n")
print("""
- 最大汇聚层会输出汇聚窗口内的最大值，平均汇聚层会输出汇聚窗口内的平均值。
- 汇聚层减轻了卷积层对位置的过度敏感。
- 可以指定汇聚层的填充和步幅。
- 使用最大汇聚层以及大于1的步幅，可减少空间维度（高度和宽度）。
- 汇聚层的输出通道数与输入通道数相同。
""")

print("\n" + "-"*50 + "\n")

# 6.5.5. 练习
print("6.5.5. 练习\n")

# 练习1: 尝试将平均汇聚层作为卷积层的特殊情况实现。
print("练习1: 将平均汇聚层作为卷积层的特殊情况实现。\n")

class AvgPoolConv(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(AvgPoolConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        # 初始化卷积核权重为全1，并除以窗口大小以实现平均
        self.conv.weight.data.fill_(1.0)
        self.conv.weight.data /= (kernel_size[0] * kernel_size[1])

    def forward(self, x):
        return self.conv(x)

# 定义平均汇聚层的参数
avg_pool_as_conv = AvgPoolConv(kernel_size=(2, 2), stride=2)

# 定义一个输入张量
X_conv = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0, 12.0],
                         [13.0, 14.0, 15.0, 16.0]]]])

print("输入张量X_conv:")
print(X_conv)

# 应用平均汇聚层
Y_avg_conv = avg_pool_as_conv(X_conv)
print("\n通过卷积实现的平均汇聚层输出:")
print(Y_avg_conv)

print("\n" + "-"*50 + "\n")

# 练习2: 尝试将最大汇聚层作为卷积层的特殊情况实现。
print("练习2: 将最大汇聚层作为卷积层的特殊情况实现。\n")

class MaxPoolConv(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPoolConv, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)

# 定义最大汇聚层的参数
max_pool_as_conv = MaxPoolConv(kernel_size=(2, 2), stride=2)

# 应用最大汇聚层
Y_max_conv = max_pool_as_conv(X_conv)
print("通过卷积实现的最大汇聚层输出:")
print(Y_max_conv)

print("\n" + "-"*50 + "\n")

# 练习3: 计算汇聚层的计算成本
print("练习3: 计算汇聚层的计算成本\n")

def pooling_cost(c, h, w, p_h, p_w, s_h, s_w):
    """
    计算汇聚层的计算成本。

    参数：
    - c (int): 输入通道数。
    - h (int): 输入高度。
    - w (int): 输入宽度。
    - p_h (int): 汇聚窗口高度。
    - p_w (int): 汇聚窗口宽度。
    - s_h (int): 步幅高度。
    - s_w (int): 步幅宽度。

    返回：
    - total_operations (int): 总的计算操作数。
    """
    out_h = (h + 2*0 - p_h) // s_h + 1  # 假设填充为0
    out_w = (w + 2*0 - p_w) // s_w + 1
    operations_per_window = p_h * p_w
    total_operations = c * out_h * out_w * operations_per_window
    return total_operations

# 示例计算
c, h, w = 3, 32, 32
p_h, p_w = 2, 2
s_h, s_w = 2, 2
cost = pooling_cost(c, h, w, p_h, p_w, s_h, s_w)
print(f"输入大小: {c}x{h}x{w}, 汇聚窗口: {p_h}x{p_w}, 步幅: {s_h}x{s_w}")
print(f"计算成本: {cost} 次操作")

print("\n" + "-"*50 + "\n")

# 练习4: 最大汇聚层和平均汇聚层的工作方式不同的原因
print("练习4: 为什么最大汇聚层和平均汇聚层的工作方式不同？\n")
print("""
最大汇聚层通过选择窗口中的最大值，能够更好地保留显著的特征，如边缘和纹理。
平均汇聚层通过计算窗口的平均值，更适合平滑特征，减弱噪声。
两者在特征提取和信息保留上有不同的侧重点，适用于不同的任务和场景。
""")

print("\n" + "-"*50 + "\n")

# 练习5: 是否需要最小汇聚层？可以用已知函数替换它吗？
print("练习5: 我们是否需要最小汇聚层？可以用已知函数替换它吗？\n")
print("""
最小汇聚层与最大汇聚层相似，但选择窗口中的最小值。虽然它在某些特定任务中可能有用，
但在大多数情况下，最大汇聚和平均汇聚已足够满足需求。最小汇聚层可以使用类似于最大汇聚层的方法实现，
例如通过在窗口中寻找最小值。
""")

print("\n" + "-"*50 + "\n")

# 练习6: 除了平均汇聚层和最大汇聚层，是否有其它函数可以考虑（提示：回想一下softmax）？为什么它不流行？
print("练习6: 除了平均汇聚层和最大汇聚层，是否有其它函数可以考虑？为什么它不流行？\n")
print("""
除了平均和最大汇聚层，还可以考虑使用加权汇聚方法，如加权平均或基于softmax的汇聚。
然而，这些方法通常更复杂，计算成本更高，且不一定在所有任务中表现优于简单的最大或平均汇聚。
因此，它们在实际应用中不如最大和平均汇聚层常用。
""")

print("\n" + "-"*50 + "\n")

# 总结
print("总结:\n")
print("""
本代码示例涵盖了PyTorch中汇聚层的基本使用，包括最大汇聚和平均汇聚。
演示了如何使用内置的汇聚层，如何自定义实现汇聚层作为卷积层的特殊情况，
以及如何处理多通道输入。通过计算汇聚层的计算成本和回答相关问题，
深入理解了汇聚层的工作机制和应用场景。

使用到的主要函数和类：
- nn.MaxPool2d: 定义二维最大汇聚层。
    参数：
        - kernel_size (int or tuple): 汇聚窗口的大小。
        - stride (int or tuple, optional): 步幅。默认与kernel_size相同。
        - padding (int or tuple, optional): 填充。默认是0。
- nn.AvgPool2d: 定义二维平均汇聚层。
    参数与MaxPool2d相同。
- torch.cat: 在指定维度上连接张量。
    参数：
        - tensors (sequence of Tensors): 要连接的张量序列。
        - dim (int): 连接的维度。
- 自定义类AvgPoolConv和MaxPoolConv: 分别将平均汇聚和最大汇聚作为卷积层的特殊情况实现。
    - AvgPoolConv通过设置卷积核权重为全1并归一化，实现了平均汇聚的效果。
    - MaxPoolConv直接使用MaxPool2d来模拟最大汇聚层。
- pool2d函数: 自定义实现2D汇聚操作，支持最大和平均汇聚模式。
    参数：
        - X (torch.Tensor): 输入张量，形状为 (高度, 宽度)。
        - pool_size (tuple): 汇聚窗口的大小 (p_h, p_w)。
        - mode (str): 汇聚模式，'max' 或 'avg'。

调用示例已在代码中展示。
""")
