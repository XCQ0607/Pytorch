print("3.5. 图像分类数据集")

# 导入必要的包
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
#rc与rcParams是Matplotlib库中用于设置全局参数的函数。
在Matplotlib库中，rc和rcParams都是用于设置全局绘图参数的机制，但它们在使用方式和设置参数的细节上有所不同。
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
    root='../data', train=True, transform=transform_train, download=True)

# 下载测试集
mnist_test = datasets.FashionMNIST(
    root='../data', train=False, transform=transform_test, download=True)

# 输出训练集和测试集的大小
print(f"训练集大小: {len(mnist_train)}")
print(f"测试集大小: {len(mnist_test)}")

# 定义一个函数，用于将数字标签转换为文本标签
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['T恤', '裤子', '套衫', '连衣裙', '外套',
                   '凉鞋', '衬衫', '运动鞋', '包', '短靴']
    return [text_labels[int(i)] for i in labels]

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
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # 张量转换为numpy数组并调整通道顺序
        img = img.numpy().transpose((1, 2, 0))
        # 反标准化
        img = img * 0.5 + 0.5  # 逆归一化到[0,1]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes

# 从训练集中获取一小批数据
batch_size = 36
train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# 获取一个批次的数据
X, y = next(iter(train_iter))

# 显示图像及对应的标签
show_images(X, 6, 6, titles=get_fashion_mnist_labels(y))

# 分割线
print("-" * 50)

# 定义一个函数，用于获取数据迭代器的工作进程数
def get_dataloader_workers():
    """返回使用的进程数"""
    return 4  # 可以根据电脑的CPU核心数进行调整

# 创建DataLoader，使用不同的参数
# DataLoader的参数包括：
#   - dataset: 数据集
#   - batch_size: 每个批次的大小
#   - shuffle: 是否在每个epoch开始时打乱数据
#   - num_workers: 使用的子进程数
#   - pin_memory: 是否将数据保存在内存的固定位置，可以加快数据加载的速度
#   - drop_last: 如果数据大小不能被batch_size整除，是否丢弃最后一个不完整的批次

# 这里我们尝试不同的batch_size来观察对性能的影响
batch_sizes = [1, 64, 256, 512, 1024]
num_workers = [0, 2, 4, 8]

# 测试不同batch_size和num_workers下的数据加载时间
for batch_size in batch_sizes:
    for workers in num_workers:
        train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                     num_workers=workers)
        start = time.time()
        for X, y in train_iter:
            pass
        end = time.time()
        print(f"batch_size={batch_size}, num_workers={workers}, 耗时: {end - start:.2f}秒")

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
                             num_workers=4, pin_memory=True)
start = time.time()
for X, y in train_iter:
    pass
end = time.time()
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
