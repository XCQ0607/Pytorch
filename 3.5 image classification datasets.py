import torch
from torch.utils import data
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import time
import numpy as np

# 如果在Jupyter Notebook中运行，可以使用以下代码设置Matplotlib显示格式为SVG
# from IPython import display
# display.set_matplotlib_formats('svg')

from matplotlib import rc
rc('font', **{'family': 'Microsoft YaHei', 'weight': 'bold', 'size': 12})
# rc与rcParams是Matplotlib库中用于设置全局参数的函数。
# 在Matplotlib库中，rc和rcParams都是用于设置全局绘图参数的机制，但它们在使用方式和设置参数的细节上有所不同。
'''
rcParams
rcParams是一个字典对象，它包含了Matplotlib中几乎所有的默认样式设置。你可以通过修改这个字典中的值来改变默认的绘图样式。例如，你可以修改线的宽度、颜色、字体大小等。
使用rcParams设置参数的基本语法是：
import matplotlib as mpl
mpl.rcParams['parameter'] = value
其中，'parameter'是你想要设置的参数名，value是你想要设置的新值。

rc
rc函数则提供了一个更简洁的方式来设置rcParams中的参数。rc函数接受任意数量的关键字参数，并将它们直接应用到rcParams字典中。这意味着你可以在一个函数调用中设置多个参数。
使用rc设置参数的基本语法是：
import matplotlib as mpl
mpl.rc('group', parameter=value)
或者，如果你想要同时设置多个参数，可以使用：
mpl.rc('group', parameter1=value1, parameter2=value2, ...)
其中，'group'是参数所属的组别（例如，'lines'、'axes'、'font'等），parameter是你想要设置的参数名，value是你想要设置的新值。

数据结构：rcParams是一个字典，而rc是一个函数。
使用方式：通过rcParams，你可以直接访问和修改字典中的键值对来设置参数。而通过rc，你可以使用函数调用的方式来设置参数，通常更加简洁和直观。
设置范围：两者都可以用来设置全局参数，影响之后的绘图操作。但是，它们本身并不直接应用于已经创建的图形对象；要更新已经存在的图形对象，通常需要重新绘制。
灵活性：rcParams由于是一个字典，因此你可以使用字典的所有操作来管理它，包括批量更新、检查键是否存在等。而rc函数则更侧重于提供一种便捷的、声明式的方式来设置常用参数。

在Matplotlib中，rcParams是一个包含大量绘图参数设置的字典。这些参数名和对应的值用于控制图表的各个方面，如线条样式、颜色、字体、坐标轴属性等。而rc函数则是通过组别（group）和参数名（parameter）来设置这些属性的便捷方式。
由于rcParams包含的参数众多，这里将按照一些常见的类别进行归纳，并列举部分重要的参数名：

1. 线条属性 (lines)
lines.linewidth: 线条宽度
lines.linestyle: 线条样式（如'-'、'--'、'-.'、':'）
lines.color: 线条颜色
lines.marker: 线条上的标记样式
lines.markersize: 标记大小

2. 坐标轴和网格 (axes 和 grid)
axes.titlesize: 坐标轴标题字体大小
axes.labelsize: 坐标轴标签字体大小
axes.unicode_minus: 是否使用Unicode字符来显示负号
grid.linestyle: 网格线样式
grid.color: 网格线颜色

3. 字体设置 (font)
font.size: 默认字体大小
font.family: 字体族（如'sans-serif'、'serif'、'monospace'）
font.sans-serif: 无衬线字体列表（如'Arial'、'DejaVu Sans'）
font.weight: 字体粗细

4. 图像属性 (figure)
figure.figsize: 图像大小（以英寸为单位）
figure.dpi: 图像分辨率（每英寸的点数）
figure.facecolor: 图像背景颜色

5. 颜色和样式 (colors 和 style)
image.cmap: 颜色映射（用于图像显示）
axes.prop_cycle: 属性循环设置（用于自动改变线条颜色等）
style.use: 加载预定义的风格设置（如'seaborn-darkgrid'）

6. 其他设置
savefig.dpi: 保存图像时的分辨率
animation.embed_limit: 嵌入动画的文件大小限制（以MB为单位）
backend: 使用的后端（如'TkAgg'、'Qt5Agg'等）

请注意，这只是rcParams中可用参数的一个子集。完整的参数列表和默认值可以通过查看Matplotlib的官方文档或在Python中执行matplotlib.rcParams.keys()来获取。

另外，rc函数中的group通常对应于rcParams字典中的一级键名，如'lines'、'axes'等。而parameter则是这些组下的具体设置项，如'linewidth'、'titlesize'等。通过rc函数，你可以以更简洁的方式同时设置同一组下的多个参数。
'''

# 定义一个计时器类，用于记录运行时间
class Timer:
    """记录多次运行时间。"""
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        """启动计时器。"""
        self.tik = time.time()
    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)
    def sum(self):
        """返回时间总和。"""
        return sum(self.times)
    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()

# 分割线


if __name__ == '__main__':
    print("3.5. 图像分类数据集")
    print("-" * 50)
    # 定义图像变换，包括将图像转换为张量和归一化
    # transforms.Compose是一个组合多个变换的函数
    # 可选的参数包括：
    #   - transforms.Resize(size): 调整图像大小到给定的size
    #   - transforms.CenterCrop(size): 从图像中心裁剪给定大小的图像
    #   - transforms.ToTensor(): 将PIL Image或numpy.ndarray转换为张量，并归一化到[0,1]
    #   - transforms.Normalize(mean, std): 使用给定的均值和标准差对图像进行标准化
    #   - transforms.RandomHorizontalFlip(p): 以给定的概率随机水平翻转
    #   - transforms.RandomRotation(degrees): 以给定的角度随机旋转图像

    # 定义训练集和测试集的图像变换
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(15),      # 随机旋转15度以内
        transforms.ToTensor(),              # 转换为张量并归一化到[0,1]
        transforms.Normalize((0.5,), (0.5,))# 标准化到[-1,1]
    ])
    # transforms.Compose处理顺序是从前往后依次执行的，因此先进行随机水平翻转和随机旋转，然后再将图像转换为张量并进行标准化。
    '''
数据预处理
transform_train 定义了对从 Fashion-MNIST 数据集中读取的每张图像执行的一系列操作。具体来说：
随机水平翻转：增加数据多样性，帮助模型学习不受物体方向影响的特征。
随机旋转：通过随机旋转图像，提升模型对图像旋转的鲁棒性。
转换为张量并归一化：将图像转为 PyTorch 张量，并将像素值归一化到 [0, 1]。
标准化：将像素值标准化为均值 0.5，标准差 0.5，使训练更加稳定和高效。


transform_train 是一个由多个变换操作组成的 transforms.Compose() 对象，具体功能如下：
1. transforms.RandomHorizontalFlip()
作用：对图像进行随机水平翻转。
用途：这种数据增强方法可以帮助模型学会从不同的角度识别图像。通过随机水平翻转图像，增加训练数据的多样性，从而提升模型的泛化能力。
例如，如果原图是一个穿着 T 恤的人，随机翻转后，它的左侧和右侧会被交换，这对于训练模型是有益的。
2. transforms.RandomRotation(15)
作用：对图像进行随机旋转，旋转角度在 -15 到 15 度之间。
用途：通过随机旋转，模型能学会对物体的旋转保持不变的特性，从而提高模型的鲁棒性，减少对物体朝向的依赖。
例如，如果图像中的物体原本是垂直的，经过旋转后，物体的朝向将有所变化，这样可以帮助模型学习到不同角度的物体特征。
3. transforms.ToTensor()
作用：将图像转换为 PyTorch 张量，并将图像的像素值缩放到 [0, 1] 范围内。
用途：PyTorch 训练时，通常需要输入是张量（tensor）。ToTensor() 会将图像从 (H, W, C) 的 NumPy 数组转换成 PyTorch 张量，同时会将像素值从 [0, 255] 缩放到 [0, 1] 之间。
例如，如果原图像的像素值范围是 [0, 255]，ToTensor() 会将它们除以 255，转换到 [0, 1] 范围内。这是因为在训练神经网络时，将数据归一化通常有助于模型收敛。
4. transforms.Normalize((0.5,), (0.5,))
作用：对图像进行标准化处理，将图像的像素值调整为均值为 0.5，标准差为 0.5 的分布。
用途：标准化操作有助于加速模型的训练，并提高模型的稳定性。在许多神经网络中，输入数据的标准化（即使数据的均值为 0，方差为 1）通常会提高训练的效率和收敛速度。
具体来说，这里：
mean=(0.5,)：表示所有通道的均值是 0.5。对于灰度图像，只有一个通道，所以均值是一个标量。
std=(0.5,)：表示所有通道的标准差是 0.5。
normalized value=(original value-mean)/std  如：原始值为0.8，经过标准化后，得到的值为(0.8-0.5)/0.5=0.6

在 datasets.FashionMNIST() 中传递 transform=transform_train 参数时，意味着在加载每个图像时，会按照 transform_train 中定义的步骤来处理图像。具体来说，transform_train 对原始数据（下载的数据）进行如下处理：
随机翻转。
随机旋转。
转换为张量并归一化。
标准化。
'''

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载并加载Fashion-MNIST数据集
    # datasets.FashionMNIST的参数包括：
    #   - root: 数据存储的路径
    #   - train: True表示加载训练集，False表示加载测试集
    #   - transform: 对图像进行的变换
    #   - target_transform: 对标签进行的变换
    #   - download: 如果本地没有数据，是否下载

    # 下载训练集
    mnist_train = datasets.FashionMNIST(
        root='./data', train=True, transform=transform_train, download=True)
    # datasets.FashionMNIST是一个用于加载Fashion-MNIST数据集的函数。Fashion-MNIST是一个替代MNIST手写数字集的图像数据集，其中包含10个类别的70,000张灰度图像，每个图像的大小为28x28像素。这个数据集通常用于测试和验证机器学习算法，特别是图像分类算法的性能。
    # 参数:datasets.FashionMNIST(root, train, transform, target_transform, download)
    #   root: 数据集的根目录路径，即数据集将被存储或加载的位置。
    #   train: 一个布尔值，如果为True，则加载训练集；如果为False，则加载测试集。
    #   transform: 一个函数或变换，用于对图像进行预处理。
    #   target_transform: 一个函数或变换，用于对标签进行预处理。
    #   download: 一个布尔值，如果为True，则在本地没有数据集的情况下，从互联网下载数据集。

    # 下载测试集
    mnist_test = datasets.FashionMNIST(
        root='./data', train=False, transform=transform_test, download=True)

    # 输出训练集和测试集的大小
    print(f"训练集大小: {len(mnist_train)}")
    print(f"测试集大小: {len(mnist_test)}")

    # 定义一个函数，用于将数字标签转换为文本标签
    def get_fashion_mnist_labels(labels):
        """返回Fashion-MNIST数据集的文本标签"""
        text_labels = ['T恤', '裤子', '套衫', '连衣裙', '外套',
                       '凉鞋', '衬衫', '运动鞋', '包', '短靴']
        return [text_labels[int(i)] for i in labels]
        # 这里定义了一个列表
        # text_labels，它包含了Fashion - MNIST数据集的10种类别的文本标签：
        # 0: t - shirt
        # 1: trouser
        # 2: pullover
        # 3: dress
        # 4: coat
        # 5: sandal
        # 6: shirt
        # 7: sneaker
        # 8: bag
        # 9: ankle
        # boot
        # 返回标签：
        # return [text_labels[int(i)] for i in labels]
        # 这行代码的作用是遍历labels中的每一个标签i，将每个数字i转换为对应的文本标签，并返回一个新的列表。使用text_labels[int(i)]可以通过数字索引访问文本标签。
        # int(i)：确保标签i是整数类型（如果i是浮点数或其他类型的话）。text_labels[int(i)]：用整数标签i在text_labels中查找对应的文本标签。
        # 示例：
        # 假设输入的labels是[0, 1, 3, 5]，那么函数将返回：
        # ['t-shirt', 'trouser', 'dress', 'sandal']


    # 定义一个函数，用于可视化一组图像
    def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
        """绘制图像列表
        参数：
            imgs: 图像张量列表
            num_rows: 行数
            num_cols: 列数
            titles: 每个图像的标题
            scale: 图像缩放比例
        """
        figsize = (num_cols * scale, num_rows * scale)  # 图像的尺寸，单位为英寸
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)   # 创建一个包含num_rows行和num_cols列的子图
        axes = axes.flatten()     # 将axes展平成一维数组
        #axes是一个列表，其中包含了num_rows*num_cols个子图，imgs是一个包含了num_rows*num_cols个图像张量的列表
        for i, (ax, img) in enumerate(zip(axes, imgs)):  # 遍历每一个图像，zip函数将axes和imgs打包成元组
            #ax代表一个子图，img代表一个图像张量   i代表当前图像的索引
            # 张量转换为numpy数组并调整通道顺序
            img = img.numpy().transpose((1, 2, 0))    # 将图像张量的通道顺序从 (C, H, W) 转换为 (H, W, C),transpose函数用于交换轴的顺序
            # 反标准化    这里的0.5和0.5是在训练数据集中计算得到的均值和标准差     因为 normalize 操作是将像素值归一化到[-1,1]之间，所以这里要将其还原到[0,1]之间
            #normalize value = (original value-mean)/std    所以 original value = normalize value*std+mean
            img = img * 0.5 + 0.5  # 逆归一化到[0,1]
            ax.imshow(img.squeeze())   # 显示图像，cmap='gray'表示使用灰度颜色映射，也就是将图像显示为灰度图像, ax.imshow(img.squeeze()，cmap='gray')，当然除了gray，还有其他颜色映射，如hot、cool、spring、summer、autumn、winter等。
            ax.axis('off')    # 隐藏坐标轴
            if titles:
                ax.set_title(titles[i])  #如果有标题，设置子图的标题
        plt.show()
        return axes
'''
1. plt.subplots 是什么？
plt.subplots() 是 Matplotlib 中用来创建多个子图（subplots）的一个非常常用的函数。它返回一个图形对象和一组子图对象（axes），这些子图可以用来绘制多个图像或图表。
函数签名：
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=None, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)
参数说明：
nrows: 子图的行数。默认值是 1。
ncols: 子图的列数。默认值是 1。
figsize: 图形的大小，单位为英寸（inch）。figsize=(width, height)，例如 (10, 8) 会返回一个 10 英寸宽、8 英寸高的图形。
sharex: 如果为 True，所有子图共享 x 轴。
sharey: 如果为 True，所有子图共享 y 轴。
squeeze: 如果为 True，并且只有一个子图，则返回一个单个 Axes 对象，而不是一个 Axes 数组。如果为 False，则始终返回 Axes 数组。
subplot_kw: 可选字典，可以用来设置子图的额外参数。
gridspec_kw: 可选字典，可以用来设置网格的额外参数。
返回值：
fig：返回一个 matplotlib.figure.Figure 对象，代表整个图形。
axes：返回一个 matplotlib.axes.Axes 对象的数组或单个 Axes 对象，取决于 nrows 和 ncols 的设置。

2.axes = axes.flatten() 的作用
在你提供的代码中，plt.subplots(num_rows, num_cols) 创建了一个 num_rows x num_cols 的子图网格。这时，axes 是一个二维数组，其形状为 (num_rows, num_cols)，表示图形网格中的所有子图。为了便于遍历和操作每个子图，通常会将其展平成一维数组。
为什么要展为一维数组？
Matplotlib 中的 plt.subplots() 返回的 axes 是一个二维数组，其中每个元素代表一个子图。例如，假设你创建了一个 2x3 的子图网格，那么 axes 的形状就是 (2, 3)，它包含 6 个子图。如果你需要通过迭代访问每个子图，展平 axes 为一维数组会更方便，因为这样你可以用一个简单的 for 循环来遍历所有的子图。
展开前后的形状：
展开前：假设你有 2 行 3 列的子图网格，axes 的形状将是 (2, 3)。
展开后：使用 axes.flatten() 会把 axes 转换为形状 (6,) 的一维数组，每个元素对应一个子图。
举个例子：
# 假设我们有一个 2 行 3 列的子图
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
# 展开后的 axes 将变成一维数组，形状为 (6,)
axes = axes.flatten()
# 使用 for 循环遍历每个子图
for ax in axes:
    ax.plot([1, 2, 3], [4, 5, 6])  # 在每个子图上画一条线
在这个例子中，axes.flatten() 使得我们能够通过 for 循环轻松地访问所有的子图，并在每个子图上绘制不同的内容。

3. show_images 函数的工作流程
结合前面的内容来看，show_images 函数利用 plt.subplots 创建一个指定数量的子图网格，并用 axes.flatten() 将二维数组展平，以便遍历每个子图并绘制图像。具体的步骤如下：
创建子图网格：通过 fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize) 创建一个包含多个子图的图形对象 fig，和包含每个子图 Axes 对象的数组 axes。
展平 axes：通过 axes.flatten() 将二维数组展平成一维数组，以便于通过 for 循环遍历每个子图。
绘制图像：对于每个子图，通过 ax.imshow(img.squeeze(), cmap='gray') 绘制图像。
反标准化：通过 img = img * 0.5 + 0.5 对图像进行逆归一化，使图像像素值从 [-1, 1] 范围恢复到 [0, 1] 范围。
设置标题：如果有传入 titles，则为每个子图设置标题。
总结
plt.subplots()：用于创建一个多子图的图形，并返回一个 fig（图形对象）和一个 axes（子图对象的数组）。



axes 和 fig 的概念
在 Matplotlib 中，axes 和 fig 是两个非常核心的对象，它们分别代表了子图（即绘图区域）和图形（即包含所有绘图区域的整体画布）。它们在绘图过程中有着不同的职责。

1. axes（子图对象）
定义：axes 是一个 子图对象，用于表示图形中的单个绘图区域。每个 axes 对象可以在其内部进行绘制（如线条、文本、图像等）。如果你创建了多个子图（如 2 行 3 列），那么 axes 将是一个包含多个 Axes 对象的数组。
储存内容：axes 存储的是 子图的绘制区域 和 子图内的所有绘图元素。具体来说，它包含以下信息：
子图的坐标轴（x轴和y轴）的范围、刻度、标签等。
子图中的数据元素，比如 plot() 绘制的线条、scatter() 绘制的散点、imshow() 显示的图像等。
其他图形元素，如标题、图例、网格线等。
注意：axes 并不存储图像的 "地址" 或图像本身。它只是提供了一个区域，在该区域内你可以绘制和显示图形内容（如图像、线条、文本等）。
示例：
fig, axes = plt.subplots(2, 3)  # 创建一个 2x3 的子图网格
# axes 是一个 2x3 的数组，里面包含了多个 Axes 对象（子图对象）
# 绘制图像
axes[0, 0].imshow(img1)  # 在第一个子图中显示 img1
axes[0, 1].imshow(img2)  # 在第二个子图中显示 img2
在这个例子中，axes[0, 0] 和 axes[0, 1] 都是 Axes 对象，它们代表了绘图区域，在这些区域内可以进行绘图操作，如显示图像、绘制曲线等。

2. fig（图形对象）
定义：fig 是一个 图形对象，代表整个绘图的画布或容器。一个图形对象 (fig) 可以包含多个子图对象 (axes)，但它本身并不直接显示图形内容，而是作为一个容器，管理所有的子图和绘图内容。
储存内容：fig 存储的是整个图形的布局和结构信息，包括：
图形的大小（figsize）。
所有的子图（axes）和它们的布局。
图形级的属性，如标题、整体的图例、图形的背景颜色、坐标系等。
注意：fig 也不直接存储图像的数据或地址，它仅仅是一个容器，保存了图形的整体结构和子图信息。
示例：
fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 创建 2 行 3 列的子图网格
# fig 是图形对象，axes 是 2x3 的子图网格
fig.suptitle("Main Title")  # 给整个图形添加一个主标题
在这个例子中，fig 代表整个画布，而 axes 存储了 6 个子图对象。在 fig 中，你可以设置一些与整个图形相关的属性，例如主标题、图例等。
axes 和 fig 的关系
fig 是图形的容器，存储整个绘图区域的布局和结构（包括所有子图）。
axes 是具体的绘图区域，负责绘制图像、线条、坐标轴等内容。

fig：表示整个图形的容器，它会保存所有的子图（axes）以及图形的尺寸和其他全局属性。
axes：表示每一个具体的绘图区域，它存储了每个子图的内容。在 axes.flatten() 操作后，axes 是一个一维数组，每个元素是一个 Axes 对象。你可以通过它来访问每个子图，并在每个子图上绘制图像。
关键点总结
axes 存储的是每个子图（绘图区域）的信息，包括坐标轴、绘制的图像、线条、文本等。
fig 存储的是图形的整体信息，包括所有子图、图形尺寸、标题等。
axes 通过 fig 来管理和排列多个子图，在一个图形对象 (fig) 中可以有多个子图 (axes)，每个子图对应一个绘图区域。
'''
'''
    在你提供的代码中，iter()和next()是Python的内置函数，它们与迭代器（iterator）配合使用。下面我将逐步解释这两个函数的作用。

    1.iter()函数
    iter()函数是用来将一个可迭代对象（比如列表、元组、数据加载器等）转换为迭代器对象。
    迭代器：迭代器是一个对象，它实现了__iter__()和__next__()
    方法，可以用来逐个返回集合中的元素。
    用法：调用iter()后，你得到的对象是一个迭代器，它可以通过next()来获取下一个元素。
    在你的代码中：
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)X, y = next(iter(train_iter))train_iter是一个PyTorch的DataLoader对象，DataLoader本身是一个可迭代对象，可以返回一个批次的训练数据。通过iter(train_iter)，你把DataLoader转换成了一个迭代器对象。这个迭代器对象可以用于逐个批次地提取数据。

    2.next()函数
    用于获取迭代器的下一个元素。
    每次调用next()时，都会从迭代器中获取一个新的元素，并返回。
    如果迭代器中的元素已经被取完了，再调用next()会抛出StopIteration异常。
    在你的代码中：
    X, y = next(iter(train_iter))
    这行代码的作用是：
    通过iter(train_iter)获取一个train_iter的迭代器。使用next()获取train_iter迭代器的下一个元素。这里的元素是一个批次的数据，它会返回一个元组(X, y)，其中X是一个包含图像数据的张量（形状通常为[batch_size, channels, height, width]），y是该批次对应的标签（通常是一个大小为[batch_size]的张量）。
    
    示例：
    假设你的train_iter是一个包含1000个图像批次的数据加载器，每个批次包含32张图像。调用iter(train_iter)会返回一个迭代器对象，然后next()会返回一个批次的数据，例如32张图像和它们的标签。
    
    3.iter()和next()的工作原理
    iter(train_iter)将train_iter转换为一个迭代器。
    next(iter(train_iter))从该迭代器中获取下一个批次的数据（X和y）。
    总结：
    iter()：将一个可迭代对象（如DataLoader）转换为一个迭代器对象，使你可以逐步获取数据。
    next()：从迭代器中获取下一个元素。通常，next()会返回一个批次的数据（比如，X是图像，y是标签）。所以，next(iter(train_iter))这行代码的作用就是从train_iter这个数据加载器中获取下一个批次的数据，并将图像数据存储在X中，将标签存储在y中。
'''
# train_iter 的作用：将 mnist_train 按照 batch_size 划分成多个批次，并且每个批次的数据形状是 [batch_size, channels, height, width]。
# mnist_train 中每个图像的原始形状是 [channels, height, width]（例如 [1, 28, 28]）。
# 使用 DataLoader 后，每个批次的形状会变成 [batch_size, channels, height, width]，例如 [32, 1, 28, 28]，表示 32 张图像组成一个批次。

    # 从训练集中获取一小批数据
batch_size = 36
train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
#batch_size参数
# 获取一个批次的数据
X, y = next(iter(train_iter))   # 从训练集中获取一个批次的数据
# 显示图像及对应的标签
show_images(X, 6, 6, titles=get_fashion_mnist_labels(y))    # 显示图像及对应的标签，6行6列的原因是batch_size=36，所以6行6列的图像可以组成一个6行6列的图像
# 分割线
print("-" * 50)
# 定义一个函数，用于获取数据迭代器的工作进程数
def get_dataloader_workers():
    """返回使用的进程数"""
    return 4  # 可以根据电脑的CPU核心数进行调整

    # 创建DataLoader，使用不同的参数
    # DataLoader的参数包括：
    #   - dataset: 数据集
    #   - batch_size: 每个批次的大小   即一个批次有多少个样本
    #   - shuffle: 是否在每个epoch开始时打乱数据
    #   - num_workers: 使用的子进程数
    #   - pin_memory: 是否将数据保存在内存的固定位置，可以加快数据加载的速度
    #   - drop_last: 如果数据大小不能被batch_size整除，是否丢弃最后一个不完整的批次

    # 这里我们尝试不同的batch_size来观察对性能的影响
    batch_sizes = [1, 64, 256, 512, 1024]
    num_workers_list = [0, 2, 4, 8]  # 可以根据电脑的CPU核心数进行调整

    # 测试不同batch_size和num_workers下的数据加载时间
    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)
            start = time.time()
            for X, y in train_iter:  # 迭代读取数据
                pass    # 仅读取数据，不进行实际计算
            end = time.time()
            print(f"batch_size={batch_size}, num_workers={num_workers}, 耗时: {end - start:.2f}秒")

    # 分割线
    print("-" * 50)

    # 问题1：减少batch_size（如减少到1）是否会影响读取性能？
    print("问题1：减少batch_size（如减少到1）是否会影响读取性能？")
    print("答案：是的，batch_size过小会导致每个批次的数据过少，迭代次数增加，总体耗时增加。")

    # 问题2：数据迭代器的性能非常重要。当前的实现足够快吗？探索各种选择来改进它。
    print("问题2：数据迭代器的性能非常重要。当前的实现足够快吗？探索各种选择来改进它。")
    print("答案：可以通过增加num_workers，加快数据预处理和加载速度。同时，使用pin_memory=True也可以提高数据加载性能。")

    # 测试使用pin_memory=True的情况
    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True)    # 使用pin_memory=True，将数据保存在CPU的固定内存中，加快数据加载速度
    start = time.time() # 记录开始时间
    for X, y in train_iter:
        pass
    end = time.time()   # 记录结束时间
    print(f"使用pin_memory=True, batch_size={batch_size}, num_workers=4, 耗时: {end - start:.2f}秒")

    # 分割线
    print("-" * 50)

    # 问题3：查阅框架的在线API文档。还有哪些其他数据集可用？
    print("问题3：查阅框架的在线API文档。还有哪些其他数据集可用？")
    print("答案：torchvision.datasets提供了多个数据集，如MNIST、CIFAR10、CIFAR100、ImageNet、COCO、VOC、CelebA等。")

    # 示例：加载CIFAR10数据集
    cifar10_train = datasets.CIFAR10(root='../data', train=True, transform=transforms.ToTensor(), download=True)
    print(f"CIFAR10训练集大小: {len(cifar10_train)}")

    # 总结
    # 本代码示例展示了如何使用PyTorch加载和处理Fashion-MNIST数据集，包括数据变换、数据加载器的参数设置和性能测试。
    # 主要使用的函数和参数包括：
    # 1. transforms.Compose(变换列表): 将多个变换组合在一起。
    # 2. datasets.FashionMNIST(root, train, transform, download): 加载Fashion-MNIST数据集。
    # 3. data.DataLoader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last): 创建数据加载器。
    #    - batch_size: 每个批次的样本数。
    #    - shuffle: 是否在每个epoch开始时打乱数据。
    #    - num_workers: 用多少个子进程加载数据。
    #    - pin_memory: 是否将数据保存在内存的固定位置。
    #    - drop_last: 是否丢弃最后一个不完整的批次。
    # 4. torchvision.datasets提供了多个常用的数据集，可以根据需要加载使用。

    # 通过本示例，可以更深入地理解图像分类数据集的加载和处理方式，以及如何优化数据加载的性能。



# 3.5.3. 整合所有组件
# 现在我们定义load_data_fashion_mnist函数，用于获取和读取Fashion-MNIST数据集。 这个函数返回训练集和验证集的数据迭代器。 此外，这个函数还接受一个可选参数resize，用来将图像大小调整为另一种形状。
# def load_data_fashion_mnist(batch_size, resize=None):  #@save
#     """下载Fashion-MNIST数据集，然后将其加载到内存中"""
#     trans = [transforms.ToTensor()]
#     if resize:
#         trans.insert(0, transforms.Resize(resize))
#     trans = transforms.Compose(trans)
#     mnist_train = torchvision.datasets.FashionMNIST(
#         root="../data", train=True, transform=trans, download=True)
#     mnist_test = torchvision.datasets.FashionMNIST(
#         root="../data", train=False, transform=trans, download=True)
#     return (data.DataLoader(mnist_train, batch_size, shuffle=True,
#                             num_workers=get_dataloader_workers()),
#             data.DataLoader(mnist_test, batch_size, shuffle=False,
#                             num_workers=get_dataloader_workers()))
# 下面，我们通过指定resize参数来测试load_data_fashion_mnist函数的图像大小调整功能。
# MXNET
# PYTORCH
# TENSORFLOW
# PADDLE
# train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
# for X, y in train_iter:
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     break
# Copy to clipboard
# torch.Size([32, 1, 64, 64]) torch.float32 torch.Size([32]) torch.int64
# 我们现在已经准备好使用Fashion-MNIST数据集，便于下面的章节调用来评估各种分类算法。

# 原始 Fashion-MNIST 图像的尺寸
# Fashion-MNIST 数据集中的每张图像都是 28×28 像素的灰度图像（即单通道图像）。每张图像的形状为 (28, 28)，如果按批次加载，数据的形状通常是 [batch_size, 1, 28, 28]，其中：
#
# batch_size 是一次处理的样本数量。
# 1 代表图像的通道数（灰度图只有一个通道）。
# 28, 28 是图像的宽度和高度。
# 转换为 Tensor 后的形状
# 当通过 transforms.ToTensor() 将图像转换为 PyTorch 的张量时，图像的尺寸会变成 [batch_size, 1, 28, 28]。具体来说：
#
# batch_size 是一个批次中的图像数量（例如，32张图片）。
# 1 是图像的通道数（因为 Fashion-MNIST 是灰度图，只有一个通道）。
# 28, 28 是图像的高度和宽度。
# 因此，转换后的张量形状是：[32, 1, 28, 28]，如果批量大小为 32。
#
# 使用 transforms.Resize(64) 后的变化
# 在调用 transforms.Resize(64) 时，图像的大小会被调整为 64×64 像素，而不是原来的 28×28。这里的 64 是目标尺寸，表示图像的 宽度和高度都调整为64。所以，如果你使用 resize=64，所有的图像都会被拉伸或压缩成 64×64，并且它们仍然是灰度图（单通道），因此变成 1 个通道。
#
# 调整大小后，图像的形状会变为 [batch_size, 1, 64, 64]，其中：
#
# batch_size 是批次中样本的数量（例如 32）。
# 1 是图像的通道数（仍然是灰度图）。
# 64, 64 是调整后的图像的宽度和高度。
# 所以，对于 resize=64，转化后的张量形状就是 [32, 1, 64, 64]，也就是说：
#
# 每个图像现在是 64×64 的大小。
# 图像有 1 个通道（灰度图）。
# 批量大小为 32，因此整个批次的形状是 [32, 1, 64, 64]。

'''
当你使用 transforms.Resize(64) 将图像从原来的 28x28 调整为 64x64，确实会增加图像的像素数量，从每张图像的 784个像素（28×28）增加到 4096个像素（64×64）。那么，问题来了：这些新增的像素是从哪里来的呢？答案是：插值。

插值（Interpolation）
在图像调整大小（resize）的过程中，PyTorch 使用了一种叫做插值的技术来“生成”新像素。简单来说，插值是一种基于已有像素计算新像素值的方法。具体来说，当我们将图像从较小的尺寸（如28×28）放大到较大的尺寸（如64×64）时，插值会估算出这些新增的像素的值。

插值的类型
在 transforms.Resize() 中，PyTorch 默认使用的是 双线性插值（Bilinear Interpolation）。这是图像调整大小时常用的一种方法，它通过考虑目标像素周围的像素值来估算新像素的值。

具体过程：
每个原始像素被放大：例如，在 28x28 到 64x64 的变换过程中，每个原始像素会“扩展”到一个更大的区域。为了填补目标区域的像素值，PyTorch 会根据原始图像中的相邻像素值来插入新像素。
插值计算：例如，如果我们要放大一个像素区块，插值方法将通过计算周围像素的加权平均值来确定新像素的值。
具体来说，如果目标尺寸是 64x64，而原始图像的尺寸是 28x28，PyTorch 会计算目标图像中每个像素点的位置，并且基于原图像中的像素位置（使用插值算法）来推断目标像素的值。
新增数据的来源：
插值生成的数据并不是真正的“原始数据”，而是通过原图的像素和相邻像素之间的关系推算出来的。可以理解为这部分数据是“人工生成”的，它并没有来自于原始图像，而是通过插值算法计算的“估计值”。
如果我们将图像看作是由像素点组成的矩阵，那么新的图像中的一些像素值并不是来自原始的 28x28 图像，它们是通过数学计算（插值）“推测”出来的。
示例说明
假设你有一个非常简单的图像：
原始图像（2x2）： 
[ 1, 2 ]
[ 3, 4 ]
如果你将其调整为 4x4，插值可能会生成以下的“推测”结果（具体结果取决于插值算法的细节）：
调整后图像（4x4）： 
[ 1, 1.5, 2, 2.5 ]
[ 2, 2.5, 3, 3.5 ]
[ 3, 3.5, 4, 4.5 ]
[ 4, 4.5, 5, 5 ]
在这个例子中，原始图像的 4 个像素被扩展成了 16 个像素，新增的像素（如 1.5、2.5 等）并没有直接来自原图，而是通过插值计算得出的。
小结
新增的像素并不来自原始的 28x28 图像，它们是通过插值方法生成的。
插值根据原始图像的像素值计算出新的像素值，从而填补目标图像的空白区域。
这种插值过程是数据增强的一种方式，它使得图像在尺寸调整时能够尽可能保留原图的视觉信息，但同时也引入了新的估算数据。
因此，虽然增加了更多的像素数据，但这些数据是基于原始图像的像素通过插值推算出来的，而不是直接从原图中获取的“真实”像素值。
'''