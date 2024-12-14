# 6.3. 填充和步幅
print("6.3. 填充和步幅\n")

import torch
from torch import nn

# 为了确保输出的清晰性，定义一个分割线函数
def print_separator():
    print("\n" + "-"*80 + "\n")

# 6.3.1. 填充
print("6.3.1. 填充\n")

# 介绍卷积层的参数
# nn.Conv2d 的主要参数包括：
# - in_channels: 输入数据的通道数
# - out_channels: 卷积产生的通道数
# - kernel_size: 卷积核的尺寸，可以是单个整数或元组 (高度, 宽度)
# - padding: 输入的每一条边补充的零的层数，可以是单个整数或元组 (高度填充, 宽度填充)
# - stride: 卷积的步幅，可以是单个整数或元组 (高度步幅, 宽度步幅)
# - dilation: 卷积核元素之间的间距
# - groups: 输入和输出之间的连接数
# - bias: 是否添加偏置项

# 定义一个函数来计算并展示卷积的输出形状
def comp_conv2d(conv2d, X):
    """
    计算卷积层的输出形状。

    参数:
    - conv2d: 卷积层对象
    - X: 输入张量

    返回:
    - 输出张量的形状
    """
    # 初始化卷积层的权重
    conv2d.reset_parameters()    # 初始化权重，初始化方法是PyTorch的默认方法kaiming_uniform_, 均匀分布，范围为[-sqrt(k), sqrt(k)]，k = 1 / fan_in，fan_in是输入通道数乘以卷积核的高度和宽度
    # 调整输入形状为 (批量大小, 通道数, 高度, 宽度)
    X = X.unsqueeze(0).unsqueeze(0)    # 添加批量和通道维度
    # 进行卷积操作
    Y = conv2d(X)    # 进行卷积操作
    # 返回去除批量和通道后的形状
    return Y.squeeze(0).squeeze(0).shape    # 返回去除批量和通道后的形状

# 示例 1：使用填充1的3x3卷积核
print("示例1：填充(padding=1) 的3x3卷积核")
conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
X1 = torch.rand(8, 8)  # 输入张量形状为 (8, 8)
print(f"输入形状: {X1.shape}")
output_shape1 = comp_conv2d(conv1, X1)
print(f"输出形状: {output_shape1}")  # 输出形状为 (8, 8)
#output_height = (8 + 2*1 - 3) / 1 + 1 = 8 (input_height - kernel_height + 2*padding) / stride + 1
print_separator()

# 示例 2：使用不同高度和宽度的填充
print("示例2：填充(padding=(2,1)) 的5x3卷积核")
conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,3), padding=(2,1))  #padding=(2,1)指的是在高度方向上填充2行，在宽度方向上填充1列
X2 = torch.rand(8, 8)
print(f"输入形状: {X2.shape}")
output_shape2 = comp_conv2d(conv2, X2)
#output_height = (8 + 2*2 - 5) / 1 + 1 = 8  (input_height - kernel_height + 2*padding_hight) / stride + 1
#output_width = (8 + 2*1 - 3) / 1 + 1 = 8  (input_width - kernel_width + 2*padding_width) / stride + 1
print(f"输出形状: {output_shape2}")
print_separator()

# 6.3.2. 步幅
print("6.3.2. 步幅\n")

# 示例3：使用步幅为2的3x3卷积核
print("示例3：步幅(stride=2) 的3x3卷积核")
conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2)
X3 = torch.rand(8, 8)
print(f"输入形状: {X3.shape}")
output_shape3 = comp_conv2d(conv3, X3)  #相当于conv3(X3)
print(f"输出形状: {output_shape3}")
print_separator()

# 示例4：使用不同高度和宽度的步幅
print("示例4：步幅(stride=(3,4)) 的3x5卷积核，填充=(0,1)")
conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,5), padding=(0,1), stride=(3,4))    #stride=(3,4)表示在高度方向上移动3个步长，在宽度方向上移动4个步长
X4 = torch.rand(8, 8)
print(f"输入形状: {X4.shape}")
output_shape4 = comp_conv2d(conv4, X4)
print(f"输出形状: {output_shape4}")
print_separator()

# 6.3.3. 小结
print("6.3.3. 小结\n")
print("填充可以增加输出的高度和宽度，常用于使输出与输入具有相同的尺寸。")
print("步幅可以减小输出的高度和宽度，通常用于降低数据的分辨率。")
print("填充和步幅结合使用，可以有效地调整数据的维度以适应网络结构的需求。")
print_separator()

# 6.3.4. 练习
print("6.3.4. 练习\n")

# 练习1：计算输出形状以验证实验结果一致性
print("练习1：验证最后一个示例的输出形状是否为 (2, 2)")
print(f"输出形状: {output_shape4} (应为 torch.Size([2, 2]))")
print_separator()

# 练习2：尝试其他填充和步幅组合
print("练习2：尝试不同的填充和步幅组合")

# 示例5：填充2，步幅3的3x3卷积
print("示例5：填充(padding=2) 的3x3卷积核，步幅(stride=3)")
conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=2, stride=3)
X5 = torch.rand(9, 9)
print(f"输入形状: {X5.shape}")
output_shape5 = comp_conv2d(conv5, X5)
print(f"输出形状: {output_shape5}")
print_separator()

# 示例6：无填充，步幅1的2x2卷积
print("示例6：无填充(padding=0) 的2x2卷积核，步幅(stride=1)")
conv6 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding=0, stride=1)
X6 = torch.rand(5, 5)
print(f"输入形状: {X6.shape}")
output_shape6 = comp_conv2d(conv6, X6)
print(f"输出形状: {output_shape6}")
print_separator()

# 练习3：对于音频信号，步幅2说明什么？
print("练习3：对于音频信号，步幅2说明什么？")
print("答：步幅2表示卷积核每次移动2个时间步长，这相当于对音频信号进行下采样，将采样率减半，从而减少数据的分辨率和计算量。")
print_separator()

# 练习4：步幅大于1的计算优势
print("练习4：步幅大于1的计算优势是什么？")
print("答：步幅大于1可以减少卷积操作的计算量和内存使用，同时实现下采样，提取更高层次的特征，增强模型的抽象能力。")
print_separator()

# 总结
print("代码示例总结：")
print("""
在本代码示例中，我们使用了PyTorch的nn.Conv2d类来演示填充和步幅的不同应用场景。
主要使用的函数和参数包括：

1. nn.Conv2d:
   - in_channels: 输入通道数
   - out_channels: 输出通道数
   - kernel_size: 卷积核大小，可以是整数或元组
   - padding: 填充大小，可以是整数或元组
   - stride: 步幅大小，可以是整数或元组

2. comp_conv2d:
   - 一个辅助函数，用于计算卷积层的输出形状。
   - 输入参数包括卷积层对象和输入张量。
   - 返回输出张量的形状。

通过不同的填充和步幅组合，我们展示了如何控制卷积操作后输出张量的尺寸。
此外，通过练习部分，我们进一步理解了步幅在实际应用中的意义和优势。
""")
