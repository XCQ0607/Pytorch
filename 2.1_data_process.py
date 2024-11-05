
import torch
import numpy as np
from six import print_


def print_separator(title):
    """打印分隔符，用于区分不同的示例输出"""
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50 + "\n")


def demonstrate_tensor_creation():
    """
    演示张量创建的各种方法
    """
    print_separator("张量创建示例")

    # 1. 从Python列表/元组创建张量
    # torch.tensor()参数说明:
    # -  数据源(列表、元组、numpy数组等)
    # - dtype: 数据类型(torch.float32, torch.int64等)
    # - device: 设备类型('cpu', 'cuda')
    # - requires_grad: 是否需要梯度计算
    print("1. 从Python列表创建张量:")
    tensor_from_list = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(tensor_from_list) #输出的张量形状为(2, 3)，数据类型为torch.float32。(2,3)中的2表示有2行，3表示有3列
    print(f"形状: {tensor_from_list.shape}, 数据类型: {tensor_from_list.dtype}\n")
    #数据类型torch.float32, torch.int64的应用区别：
    # - torch.float32: 单精度浮点数，用于存储一般的数值数据，如模型权重、输入数据等。
    # - torch.int64: 整数类型，用于存储整数数据，如索引、标签等。
    #torch.float32简写：torch.float，torch.int64简写：torch.long

    # torch.tensor()
    # 函数在PyTorch中用于创建一个张量（tensor），它可以接受多种数据类型，并根据提供的数据自动推断出相应的数据类型。然而，你也可以通过dtype参数显式地指定张量的数据类型。
    #
    # 对于torch.float32（或简写为torch.float）和torch.int64（或简写为torch.long），它们在PyTorch中有特定的应用场景：
    # torch.float32:
    # 这是一个32位浮点数类型，通常用于表示实数，如神经网络的权重和偏置。
    # 由于其精度适中且内存占用相对较小，因此它是深度学习中最常用的数据类型之一。
    # 当你使用torch.tensor()创建张量且未指定dtype参数时，如果输入数据包含浮点数，则默认会使用torch.float32类型。
    # torch.int64:
    # 这是一个64位整数类型，通常用于表示整数索引或进行整数运算。
    # 在某些情况下，如处理大规模数据集时，使用64位整数可以确保索引的唯一性和范围。
    # 当你使用torch.tensor()创建整数张量时，如果未指定dtype参数，默认可能会使用torch.int64（这取决于PyTorch的版本和平台）。然而，为了节省内存，有时可能会考虑使用更小的整数类型，如torch.int32或torch.int16，只要它们的范围满足你的需求。
    #
    # 总的来说，torch.float32和torch.int64是PyTorch中非常常用的数据类型，分别用于表示浮点数和整数。选择哪种类型取决于你的具体需求和数据范围。在创建张量时，你可以通过dtype参数显式地指定所需的数据类型。




    # 2. 创建特殊张量
    print("2. 特殊张量创建:")
    # arange()参数: start, end, step  (不包括end)
    # arange()参数可以只传入1个，如torch.arange(10)，则会生成一个包含0到9的一维张量。,即为[0,1,2,3,4,5,6,7,8,9]
    arange_tensor = torch.arange(0, 10, 2)
    test_arange_tensor = torch.arange(10)
    print(f"等差数列张量arange3参: {arange_tensor}")
    print(f"等差数列张量arange1参: {test_arange_tensor}")

    # linspace()参数: start, end, steps
    linspace_tensor = torch.linspace(0, 1, 5)
    print(f"均匀分布张量: {linspace_tensor}")

    # rand()参数: *sizes (可变参数指定形状)   随机均匀分布  rand范围为[0,1),如果想更改为[0,2)，则需要乘以2，更改为[1,2)，则需要加上1
    random_tensor = torch.rand(2, 3)
    print(f"随机均匀分布张量:\n{random_tensor}")
    random_tensor1 = torch.clone(torch.rand(2, 3)+1)    #或者这里直接torch.rand(2, 3)+1
    print(f"随机均匀分布张量1:\n{random_tensor1}")

    # randn()参数: *sizes (可变参数指定形状)  randn英文全称：normal distribution 正态分布
    normal_tensor = torch.randn(2, 3)    #正态分布，均值为0，标准差为1
    print(f"正态分布张量:\n{normal_tensor}\n")


def demonstrate_tensor_operations():
    """
    演示张量操作
    """
    print_separator("张量操作示例")

    # 创建示例张量
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)     #torch.tensor()参数说明: 数据源(列表、元组、numpy数组等)，dtype: 数据类型(torch.float32, torch.int64等)，device: 设备类型('cpu', 'cuda')，requires_grad: 是否需要梯度计算
    b = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)    #shape:(2,3)
    c = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    # 1. 基本运算
    print("1. 基本运算:")
    print(f"加法:\n{a + b}")
    print(f"乘法:\n{a * b}")
    print(f"矩阵乘法:\n{torch.matmul(a, b.T)}\n")   #torch.matmul()参数说明: 第一个参数是矩阵1，第二个参数是矩阵2，第三个参数是是否进行转置
    #.T 是 PyTorch 中张量（Tensor）的一个属性，用于返回该张量的转置。在矩阵的上下文中，转置意味着将矩阵的行变为列，将列变为行。
    # 如果b是一个形状为(m, n)的矩阵，那么b.T将是一个形状为(n, m)的矩阵
    #在矩阵乘法中，两个矩阵能够相乘的条件是第一个矩阵的列数必须等于第二个矩阵的行数。这个条件通常被称为“矩阵乘法的维度匹配规则”。

    # 线性代数操作
    print("线性代数操作:")
    # 矩阵转置
    print(f"转置:\n{a.T}")
    # 矩阵逆     矩阵必须是方阵，方阵指的是行数和列数相等的矩阵。
    #如：matrix = torch.tensor([[1, 2], [3, 4]])
    #inverse_matrix = torch.inverse(matrix)
    #print(inverse_matrix)
    #输出：tensor([[-2.0000, 1.0000], [1.5000, -0.5000]])
    #请注意，即使矩阵是方阵，也可能不存在逆矩阵（例如，如果矩阵是奇异的或退化的）
    print(f"逆:\n{torch.inverse(c)}")
    # 矩阵行列式   矩阵必须是方阵，方阵指的是行数和列数相等的矩阵。
    print(f"行列式:\n{torch.det(c)}\n")

    # 在使用torch.linalg.inv(A)之前，需要确保矩阵A是一个方阵（即其行数和列数相等），并且A是可逆的（即其行列式不为零，或者A是非奇异的）。如果A不是方阵或者不可逆，该函数将会抛出错误。

    print("torch.linalg.inv(A) 的简单示例：")
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
    A = A + A.t()  # 加上其转置以确保它是可逆的（仅作为示例，并非所有方阵加其转置都可逆）

    #A.t()与A.T是等价的，都是返回矩阵的转置。

    # 计算逆矩阵
    try:
        inv_A = torch.linalg.inv(A)
        print(f"A的逆矩阵:\n{inv_A}")
    except RuntimeError as e:
        print(f"无法计算逆矩阵: {e}\n")

    # 2. 高级索引和切片
    print("2. 高级索引和切片:")
    complex_tensor = torch.randn(4, 5)
    print(f"原始张量:\n{complex_tensor}")
    # 布尔索引
    mask = complex_tensor > 0
    print(f"正数元素:\n{complex_tensor[mask]}")
    # 花式索引
    indices = torch.tensor([0,1,2])    # 选择第0行和第1行第2行
    print(f"选择特定行:\n{complex_tensor[indices]}")
    print(f"选择特定列:\n{complex_tensor[:, 3]}")  #[:, 3]中，: 表示所有行，3表示第4列
    print(f"选择特定行和列:\n{complex_tensor[indices, 3]}\n")    #传入的indices是一个一维张量，它包含了要选择的行的索引。

def demonstrate_tensor_transformations():
    """
    演示张量变换
    """
    print_separator("张量变换示例")
    #---------------------------
    import torch
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib.pyplot as plt
    import numpy as np
    # 创建一个3D张量
    tensor_3d = torch.arange(24).reshape(2, 3, 4)

    # 创建一个SummaryWriter对象
    writer = SummaryWriter('runs/tensor_3d_visualization')

    # 打印张量的形状和内容
    print(f"原始3D张量形状: {tensor_3d.shape}")
    print(f"原始3D张量:\n{tensor_3d}")

    # 将3D张量的某些层面转换为2D图像，并使用TensorBoard展示
    # 创建一个SummaryWriter对象，确保它在循环外部创建，以便在整个循环过程中保持打开状态
    writer = SummaryWriter('runs/tensor_3d_visualization')

    for i in range(tensor_3d.shape[0]):  # 遍历第一个维度
        # 获取当前层面的2D张量（实际上是保持第二、三维度不变的3D张量的一个切片）
        layer_2d = tensor_3d[i]  # 这将是一个形状为 (3, 4) 的二维张量
        # 将2D张量转换为numpy数组，以便使用matplotlib绘图
        layer_np = layer_2d.numpy()

        # 创建一个图像，并绘制2D张量的热图
        fig, ax = plt.subplots()
        cax = ax.matshow(layer_np)
        plt.colorbar(cax)
        plt.title(f'Layer [{i}] of the 3D tensor')  # 修改了标题以反映只遍历了一个维度

        # 将图像保存到磁盘，并使用TensorBoard展示
        writer.add_figure(f'Layer_{i}', fig)  # 修改了标签以反映只遍历了一个维度
        plt.close(fig)  # 关闭图像，以避免在内存中积累太多图像对象

    # 关闭SummaryWriter对象
    writer.close()

    #------------------------

    # 1. 维度变换
    # transpose()参数: dim0, dim1 (要交换的维度)
    print("\n1. 维度变换:")
    transposed = tensor_3d.transpose(1, 2)  # 交换第二和第三个维度
    print(f"转置后形状: {transposed.shape}")

    # permute()参数: dims (新的维度顺序)
    permuted = tensor_3d.permute(2, 0, 1)    # 重新排列维度
    print(f"重排后形状: {permuted.shape}")

    # 2. 维度扩展与压缩
    # unsqueeze()参数: dim (在哪个位置插入新维度)
    print("\n2. 维度扩展与压缩:")
    expanded = tensor_3d.unsqueeze(0)    # 在第0个位置插入一个新维度.维度的shape从(2, 3, 4)变为(1, 2, 3, 4)
    print(f"扩展前形状: {tensor_3d.shape}")
    print(f"扩展前张量:\n{tensor_3d}")
    print(f"扩展后形状: {expanded.shape}")
    print(f"扩展后张量:\n{expanded}")
    #在PyTorch中，unsqueeze函数用于在指定位置插入一个新的维度，其大小为1。
    #这个函数只能在指定位置插入一个大小为1的维度。


    # # 假设 tensor_3d 是一个形状为 (2, 3, 4) 的张量
    tensor_3d1 = torch.randn(2, 3, 4)
    print(tensor_3d1.shape)  # 输出应该是 torch.Size([2, 3, 4])
    AA = tensor_3d1.unsqueeze(1)  # 先显式地添加维度
    print(AA.shape)
    expanded1 = AA.repeat(1, 2, 1, 1)  # 在第1个位置插入一个新维度
    print(expanded1.shape)  # 输出应该是 torch.Size([2, 2, 3, 4])
    # 在这个例子中，repeat(1, 2, 1, 1)表示在第0维保持不变（重复1次），在第1维重复2次（从而得到额外的“2”这一维度），而在第2维和第3维保持不变（各重复1次）。这样，你就得到了一个形状为(2, 2, 3, 4)的张量。
    # 原始形状: [2, 1, 3, 4]
    # repeat参数: (1, 2, 1, 1)
    #
    # 计算过程：
    # 第0维: 2 × 1 = 2
    # 第1维: 1 × 2 = 2
    # 第2维: 3 × 1 = 3
    # 第3维: 4 × 1 = 4
    #
    # 最终形状: [2, 2, 3, 4]


    # squeeze()参数: dim (要压缩的维度，可选)
    squeezed = expanded.squeeze(0)
    print(f"压缩后形状: {squeezed.shape}")

    # x.unsqueeze(0)
    # 会得到形状[1, 4]：
    # 原始: [1, 2, 3, 4]
    # 形状: [4]
    # unsqueeze(0): [[1, 2, 3, 4]]
    # 形状: [1, 4]
    #
    # x.unsqueeze(1)
    # 会得到形状[4, 1]：
    # 原始: [1, 2, 3, 4]
    # 形状: [4]
    # unsqueeze(1): [[1], [2], [3], [4]]
    # 形状: [4, 1]
    #
    # squeeze的效果：squeeze(0)只会移除第0维的维度1squeeze(1)只会移除第1维的维度1没有参数的squeeze()会移除所有维度为1的维度

    # unsqueeze既可以在指定维度上插入大小为1的维度，也可以去除维度为1的维度。

def demonstrate_advanced_operations():
    """
    演示高级操作
    """
    print_separator("高级操作示例")

    # 1. 广播机制
    print("1. 广播机制示例:")
    x = torch.randn(3, 1, 4)
    y = torch.randn(1, 2, 4)
    z = x + y  # 广播到 (3, 2, 4)
    #广播机制是指在进行操作时，PyTorch会自动将较小的张量广播到较大的张量的形状，以匹配它们的维度。
    print(f"广播后形状: {z.shape}\n")

    # 2. 聚合操作
    print("2. 聚合操作:")
    data = torch.randn(4, 5)
    print(f"原始数据:\n{data}")
    # reduce()参数说明:
    # - dim: 要归约的维度
    # - keepdim: 是否保持维度
    print(f"按行求均值: {data.mean(dim=0)}")   # 按行求均值，.mean()是求均值
    # dim=0表示按行求均值，dim=1表示按列求均值
    print(f"按列求和: {data.sum(dim=1)}")   # 按列求和,.sum()是求和
    #特定行的平均值
    print(f"keepdim平均值: {data.mean(dim=0, keepdim=True)}")
    #特定列的和
    print(f"keepdim和: {data.sum(dim=1, keepdim=True)}")

    # 3. 矩阵乘法
    print("\n3. 矩阵乘法:")
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    C = torch.matmul(A, B)  # 矩阵乘法
    print(f"矩阵乘法结果:\n{C}")

    # 4. 范数计算
    print("\n4. 范数计算:")
    print(f"L1范数: {torch.norm(data, p=1)}")
    print(f"L2范数: {torch.norm(data, p=2)}")

    # 5. 张量拼接
    print("\n5. 张量拼接:")
    data1 = torch.randn(2, 3)
    data2 = torch.randn(2, 3)
    concatenated = torch.cat([data1, data2], dim=0)  # 按行拼接
    print(f"按行拼接结果:\n{concatenated}")
    concatenated = torch.cat([data1, data2], dim=1)  # 按列拼接
    print(f"按列拼接结果:\n{concatenated}")

    # 6. 张量分割
    print("\n6. 张量分割:")
    split_data = torch.split(data, 2, dim=0)  # 按行分割
    print(f"按行分割结果:\n{split_data}")
    split_data = torch.split(data, 2, dim=1)  # 按列分割
    print(f"按列分割结果:\n{split_data}")

    # 7. 张量索引
    print("\n7. 张量索引:")
    print(f"原始数据:\n{data}")
    print(f"第0行: {data[0]}")
    print(f"第1列: {data[:, 1]}")
    print(f"第0行第1列: {data[0, 1]}")

    # 8. where操作
    print("\n8. where操作:")
    result = torch.where(data > 0, data, torch.zeros_like(data))
    print(f"条件替换结果:\n{result}")



def main():
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)

    # 运行所有演示
    demonstrate_tensor_creation()
    demonstrate_tensor_operations()
    demonstrate_tensor_transformations()
    demonstrate_advanced_operations()


if __name__ == "__main__":
    main()

    print("open http://localhost:6006 in your browser")
# import subprocess
# subprocess.Popen(["tensorboard", "--logdir=runs"])
# 当需要终止tensorboard进程时
# tb_process.terminate()

    import subprocess
    subprocess.check_output(["tensorboard", "--logdir=runs"])

# subprocess.Popen和subprocess.check_output在Python中用于执行子进程，但它们的行为和用途有所不同，这解释了为什么在使用tensorboard命令时，一个可以通过Ctrl+C取消，而另一个不可以。
#
# subprocess.Popen:
# Popen是“Process Open”的缩写，它用于启动一个子进程并与之进行通信。
# 当你使用subprocess.Popen(["tensorboard", "--logdir=runs"])时，它会在后台启动一个tensorboard进程，并且不会等待该进程完成。因此，Python脚本会继续执行后续的代码，而不会阻塞在当前行。
# 由于Popen只是启动了进程而没有等待它完成，所以你不能直接通过Ctrl+C来中断这个tensorboard进程。Ctrl+C通常只会中断前台进程，也就是你当前正在运行的Python脚本。如果tensorboard进程在后台运行，你需要通过其他方式（如使用任务管理器或kill命令）来终止它。
#
# subprocess.check_output:
# check_output是一个便利函数，用于运行命令并等待其完成，然后返回其输出。
# 当你使用subprocess.check_output(["tensorboard", "--logdir=runs"])时，Python脚本会暂停执行，并等待tensorboard命令完成。这意味着tensorboard进程是在前台运行的。
# 由于check_output会等待进程完成并捕获其输出，所以你可以通过Ctrl+C来中断这个进程。当你按下Ctrl+C时，它会发送一个中断信号给前台进程，也就是正在运行的tensorboard命令，从而终止它。
# 需要注意的是，虽然你可以通过Ctrl+C来中断使用check_output启动的tensorboard进程，但这并不是一种推荐的做法。check_output主要用于执行那些会快速完成并返回输出的命令。对于像tensorboard这样需要长时间运行的服务器进程来说，使用Popen并正确管理子进程可能是一个更好的选择。