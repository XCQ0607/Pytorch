import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# 输出目录和子目录的名称
print("7.3. 网络中的网络（NiN）")


# 7.3.1 NiN块的定义
# 该块包括一个常规卷积层和两个1x1卷积层，逐像素的全连接层
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    """
    NiN块定义，包含一个普通卷积层和两个1x1卷积层。

    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 卷积核的大小
    :param strides: 步幅
    :param padding: 填充大小
    :return: 包含多个卷积层和ReLU激活函数的Sequential模块
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


# 7.3.2 NiN模型的定义
def create_nin_model(num_classes=10):
    """
    创建一个NiN模型，包括多个NiN块和全局平均池化层。

    :param num_classes: 模型的输出类别数量
    :return: 包含NiN块和全局平均池化层的模型
    """
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, num_classes, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),   # 全局平均池化层，将每个通道的特征图转换为1x1的特征图
        nn.Flatten()                    # 展平层，将特征图转换为一维向量
    )
    return net


# 7.3.3 训练数据加载和训练函数定义
def load_data_fashion_mnist(batch_size, resize=None):
    """
    加载并返回Fashion-MNIST数据集。

    :param batch_size: 每个批次的样本数量
    :param resize: 如果指定，图像将会调整到该大小
    :return: 训练和测试数据的迭代器
    """
    transform = [transforms.ToTensor()] # 定义数据变换，将图像转换为张量
    if resize:  # 如果指定了调整大小，则添加Resize变换
        transform.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform)   # transforms.Compose是一个组合变换的类，它可以将多个变换组合成一个操作
    '''
    1. transforms是什么？
    transforms通常指的是图像处理库中的一个模块，它提供了一系列用于图像预处理和增强的函数。在PyTorch的torchvision库中，transforms模块包含了多种图像变换操作，如裁剪、旋转、调整大小、归一化等。这些操作可以用于数据增强，提高模型的泛化能力，或者将图像数据转换为模型可以接受的格式。
    2. transforms.Compose有什么作用？
    transforms.Compose是transforms模块中的一个类，它用于将多个图像变换操作组合成一个序列。这样，当需要对图像应用一系列变换时，可以方便地将这些变换封装成一个单一的操作。在上面的代码中，transforms.Compose(transform)就是将transform列表中的变换操作组合起来，形成一个可以一次性应用于图像的变换序列。
    3. resize是什么？
    resize是一个变量，它通常表示要将图像调整到的目标大小。在上面的代码中，resize是一个可选的参数，如果指定了它的值（通常是一个元组，表示新的宽度和高度），则会将transforms.Resize(resize)添加到变换序列中。
    4. transform.insert是什么？
    transform.insert是Python列表（list）的一个方法，用于在列表的指定位置插入一个元素。在上面的代码中，transform.insert(0, transforms.Resize(resize))的作用是在transform列表的开头（索引为0的位置）插入一个transforms.Resize(resize)变换操作。这样，当应用这个变换序列时，Resize变换会首先被应用。
    5. resize的变量类型是什么？
    resize的变量类型通常是一个元组（tuple），它包含两个整数，分别表示调整后的图像宽度和高度。例如，resize=(256, 256)表示将图像调整为256x256像素的大小。有些实现也可能允许使用单一的整数作为参数，这通常意味着图像将被调整为该整数指定的正方形大小（即宽度和高度相等）。
    '''
    # 加载训练集和测试集
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)    # 使用transforms.Compose将变换组合成一个操作
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter    # 返回训练集和测试集的迭代器


# 训练函数
def train_model(net, train_iter, test_iter, num_epochs, lr, device):
    """
    训练NiN模型。

    :param net: 要训练的模型
    :param train_iter: 训练数据加载器
    :param test_iter: 测试数据加载器
    :param num_epochs: 训练轮数
    :param lr: 学习率
    :param device: 设备（CPU或GPU）
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # 将模型加载到设备上
    net.to(device)

    for epoch in range(num_epochs):
        net.train()
        train_loss, train_correct = 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (output.argmax(1) == y).sum().item()
        #在大多数深度学习框架中（如PyTorch），数据迭代器（DataLoader）通常封装了一个数据集（Dataset）
        train_acc = train_correct / len(train_iter.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_iter)}, Train Accuracy: {train_acc:.4f}")

        # 测试模型
        net.eval()
        test_correct = 0
        with torch.no_grad():    # 关闭梯度计算，以避免对测试过程产生影响
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                output = net(X)
                test_correct += (output.argmax(1) == y).sum().item()

        test_acc = test_correct / len(test_iter.dataset)
        print(f"Test Accuracy: {test_acc:.4f}")


# 7.3.4 代码演示

# 输出分隔线
print("\n" + "-" * 50 + "\n")

# 创建模型
num_classes = 10
net = create_nin_model(num_classes)

# 打印模型结构
print("NiN模型结构：")
print(net)

# 设置训练参数
batch_size = 128
num_epochs = 10
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

# 训练模型
train_model(net, train_iter, test_iter, num_epochs, lr, device)

# 总结:
# 1. nin_block: 定义了NiN块，由一个常规卷积层和两个1x1卷积层组成。
#    参数：in_channels（输入通道数），out_channels（输出通道数），kernel_size（卷积核大小），
#          strides（步幅），padding（填充大小）。
# 2. create_nin_model: 创建一个NiN模型，包含多个NiN块和全局平均池化层。
#    参数：num_classes（类别数目）。
# 3. load_data_fashion_mnist: 加载Fashion-MNIST数据集并返回数据迭代器。
#    参数：batch_size（批次大小），resize（可选，图片缩放大小）。
# 4. train_model: 用于训练模型，计算损失和准确度。
#    参数：net（模型），train_iter（训练数据迭代器），test_iter（测试数据迭代器），
#          num_epochs（训练轮数），lr（学习率），device（设备：CPU或GPU）。

