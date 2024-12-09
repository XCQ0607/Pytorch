# ============================================================
# 6.1 从全连接层到卷积
# ============================================================
print("6.1 从全连接层到卷积")

# ------------------------------------------------------------
# 本示例将分步骤演示从全连接层过渡到卷积运算的概念，展示 PyTorch 中相关操作的实现和用法。
# 将涵盖以下内容：
# 1. 全连接层与高维输入张量的参数爆炸问题演示
# 2. 利用卷积层减少参数、利用局部感受野与平移不变性进行特征提取的思路
# 3. 卷积层参数（卷积核大小、输入通道、输出通道、步幅、填充等）的含义与用法
# 4. 在控制台中清晰打印各函数示例、以及分割线分块展示
# 5. 在代码与注释中解释函数的必选参数和可选参数，并给出中文说明
# 6. 举例展示卷积层对图像的特征提取、多通道输入输出示例
# 7. 回答文档中给出的练习问题（在注释中说明）
#
# 注：本代码不依赖d2l包，如需数据生成与可视化，将直接使用PyTorch和matplotlib等。
#
# ------------------------------------------------------------

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
print("\n" + "="*60)
print("演示1: 全连接层在高维输入上的参数爆炸问题")
print("="*60)

# 假设我们有一个输入图像为 (C,H,W) = (3, 256, 256) 的彩色图像
# 转换为向量后长度为 3*256*256 = 196,608 个特征。
# 若使用全连接层映射到 1000 个隐藏单元，则参数数量约为 196,608*1000 ≈ 1.96608e8 个参数。
C, H, W = 3, 256, 256
input_dim = C * H * W
hidden_units = 1000

fc_layer = nn.Linear(input_dim, hidden_units)
# 查看全连接层参数量
param_count_fc = sum(p.numel() for p in fc_layer.parameters())
print(f"全连接层参数数量: {param_count_fc}")

# 这在实际中非常庞大，且对训练数据的需求和计算需求都非常高。

# ------------------------------------------------------------
print("\n" + "="*60)
print("演示2: 使用卷积层减少参数量")
print("="*60)

# 卷积层参数只依赖于卷积核大小和输入输出通道数，与输入的H,W无关。
# 例如：一个卷积核大小为 3x3、输入通道3、输出通道16的卷积层。
# 参数数量 = (3 * 3 * 输入通道数 * 输出通道数) + 输出通道数的偏置 = (3*3*3*16) + 16 = 432 + 16 =448
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
param_count_conv = sum(p.numel() for p in conv_layer.parameters())
print(f"卷积层参数数量: {param_count_conv}")

# 与全连接层相比减少了几个数量级的参数量。

# ------------------------------------------------------------
print("\n" + "="*60)
print("演示3: 卷积层的平移不变性与局部性原则")
print("="*60)

# 我们来创建一个简单的输入张量(单个图片)，演示卷积的平移不变性。
# 假设输入为 (N, C, H, W) = (1, 1, 5, 5), 单通道小图像
# 我们定义一个简单的卷积核来检测某些局部模式，比如中间的值。
input_tensor = torch.zeros((1,1,5,5))
input_tensor[0,0,2,2] = 1.0  # 在中心位置放一个值为1的像素

# 定义卷积核为 3x3，大部分权重为0，中心为1，用来检测中心像素位置
simple_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
with torch.no_grad():
    # 初始化卷积核，让中间的权重为1，其余为0
    # 卷积核形状： (out_channels, in_channels, kernel_height, kernel_width) = (1,1,3,3)
    simple_conv.weight.fill_(0.0)
    simple_conv.weight[0,0,1,1] = 1.0

output = simple_conv(input_tensor)
print("初始输入张量:\n", input_tensor[0,0])
print("卷积输出张量:\n", output[0,0])

# 我们将输入平移一下，看输出是否仍然识别到同样的模式
input_tensor_shifted = torch.zeros((1,1,5,5))
input_tensor_shifted[0,0,2,3] = 1.0 # 将"1"像素向右移动一列
output_shifted = simple_conv(input_tensor_shifted)
print("平移后的输入张量:\n", input_tensor_shifted[0,0])
print("平移后的卷积输出张量:\n", output_shifted[0,0])

# 可见输出的高值也随输入特征的移动而相应移动，体现了平移不变性。


# ------------------------------------------------------------
print("\n" + "="*60)
print("演示4: 多通道输入与输出的卷积操作")
print("="*60)

# 当输入有多通道时，卷积核会对每个通道分别加权，再求和产生输出通道特征。
# 演示：输入通道数=3, 输出通道数=2, 卷积核=3x3
multi_channel_conv = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1, bias=True)

# 随机输入(N=1张图片, C=3通道, H=4, W=4)
multi_input = torch.randn(1, 3, 4, 4)
multi_output = multi_channel_conv(multi_input)

print("多通道输入张量形状:", multi_input.shape)
print("卷积后多通道输出张量形状:", multi_output.shape)
# 输出形状为 (1, 2, 4, 4), 即2个输出通道的特征图


# ------------------------------------------------------------
print("\n" + "="*60)
print("演示5: 卷积层的参数及用法介绍")
print("="*60)

# torch.nn.Conv2d函数参数介绍：
# 必选参数：
#   in_channels(int): 输入通道数，如RGB图像为3通道
#   out_channels(int): 输出通道数，即卷积产生的特征图数量
#   kernel_size(int或tuple): 卷积核大小，如3或(3,3)
#
# 可选参数（常用）：
#   stride(int或tuple, 默认1): 卷积核移动步幅，每次滑动多少像素
#   padding(int或tuple, 默认0): 在输入图像周围增加像素填充，控制输出大小
#   dilation(int或tuple, 默认1): 卷积核元素之间的间隔
#   groups(int, 默认1): 分组卷积, 控制输入输出之间的连接模式
#   bias(bool, 默认True): 是否使用偏置项
#
# 示例: Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
# 意义：输入3通道特征图，使用3x3卷积核，步幅为2，输出16通道特征图，
#       并在输入每边padding为1，从而控制输出空间尺寸。
#
# 实例演示：
conv_example = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=True)
example_input = torch.randn(1,3,64,64)
example_output = conv_example(example_input)
print("Conv2d函数示例输出形状:", example_output.shape)


# ------------------------------------------------------------
print("\n" + "="*60)
print("演示6: 将卷积等同于特殊情形下的全连接层（Δ=0情形）")
print("="*60)

# 当卷积核大小为1x1时，相当于在每个像素位置上对通道进行线性变换。
# 这就像对每个像素点的通道特征进行一个全连接层映射，不考虑相邻像素。
#
# 创建一个1x1卷积来验证此思想：
one_by_one_conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=1)
# 输入为 (N,C,H,W)，输出的 (N,5,H,W) 中，每个像素的5维向量是输入3维像素向量的线性变换。
one_by_one_input = torch.randn(1,3,10,10)
one_by_one_output = one_by_one_conv(one_by_one_input)
print("1x1卷积示例输出形状:", one_by_one_output.shape)
# 等价于对于每个像素位置，有一个线性变换 (3->5)，不依赖空间邻域，从而可视为为每个通道独立进行全连接。


# ------------------------------------------------------------
print("\n" + "="*60)
print("文档中练习问题的说明与回答（在注释中给出）")
print("="*60)

# 练习问题回答：
# 1. 假设卷积层覆盖区域Δ=0，即卷积核大小为1x1的情况。此时卷积层对每个空间位置的输出仅依赖该位置的输入通道值。
#    因此，对每个像素点的通道特征做线性组合即相当于一个全连接层（针对通道维度），不考虑空间邻域。这与上面one_by_one_conv的例子相同。
#
# 2. 为什么平移不变性可能不是好主意？
#    在某些任务中（如语义信息可能与绝对位置相关），简单的平移不变性会丢失全局定位信息。例如，在图像中有些物体只在特定区域出现，
#    强制平移不变性会让模型无法利用这些全局位置信号。
#
# 3. 当从图像边界像素获取隐藏表示时，需要考虑填充(padding)问题，以确保卷积核有足够的像素计算特征，
#    否则边缘像素的信息可能比中心像素处理不公平。例如使用padding=1可以在边界增加虚拟像素（通常为0）以保持输出尺寸和信息对称。
#
# 4. 类似的音频卷积层架构是使用1D卷积（nn.Conv1d）来处理时间序列数据，每个卷积核在时间轴上一维展开，
#    从而利用局部性和平移不变性在音频信号中提取特征。
#
# 5. 对文本数据同样适用卷积层（nn.Conv1d），将词向量序列当作时间序列，利用卷积核在词序列上滑动来提取n-gram特征。
#
# 6. 证明 f*g = g*f（卷积的交换律）：
#    卷积定义：(f*g)(i,j) = sum_a sum_b f(a,b)*g(i-a,j-b)，
#    若交换f和g的位置，(g*f)(i,j) = sum_a sum_b g(a,b)*f(i-a,j-b)。
#    通过变量替换(a' = i-a, b' = j-b)可得到两者表达式一致，从而 f*g = g*f。
#    此为数学性质，这里不在代码中演示。


# ------------------------------------------------------------
print("\n" + "="*60)
print("总结本代码示例使用的函数与参数介绍")
print("="*60)
# 在本代码示例中使用的主要函数与模块：
# 1. nn.Linear(in_features, out_features, bias=True): 全连接层
#    参数：
#      in_features(int): 输入特征维度
#      out_features(int): 输出特征维度
#      bias(bool): 是否使用偏置项
#    功能：实现 y = xW^T + b 的线性映射。
#
# 2. nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
#    参数：
#      in_channels(int): 输入通道数
#      out_channels(int): 输出通道数
#      kernel_size(int或tuple): 卷积核高宽
#      stride(int或tuple): 步幅
#      padding(int或tuple): 填充大小
#      bias(bool): 是否使用偏置
#    功能：对输入图像进行2D卷积操作，从局部区域提取特征并生成特征图。
#
# 3. torch.randn(*size): 生成正态分布的随机张量
# 4. tensor.fill_(value): 用指定的值填充张量
#
# 调用示例已在上方代码中给出，通过print函数在控制台输出说明。
#
# 整个示例演示了从全连接层到卷积层的过渡，展现了卷积的参数高效性、平移不变性、局部性、多通道特性，
# 以及将1x1卷积等价为通道维全连接的特例，帮助理解卷积神经网络的构造及优势。
#
# 通过上述代码和注释，可以更全面地理解卷积层在深度学习中的使用场景与原理。
