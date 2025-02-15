import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

print("7.2. 使用块的网络（VGG）")


# 定义一个VGG块，VGG块由多个卷积层和一个池化层构成
def vgg_block(num_convs, in_channels, out_channels):
    """
    构建VGG块的函数
    num_convs: 卷积层的数量
    in_channels: 输入通道数
    out_channels: 输出通道数
    返回一个nn.Sequential模型，包含多个卷积层和池化层
    """
    layers = [] # 存储卷积层和池化层的列表
    for _ in range(num_convs):
        #append()方法将元素添加到列表末尾
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels  # 更新输入通道数为输出通道数
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 添加最大池化层，池化窗口2x2，步幅2
    return nn.Sequential(*layers)   #*layers是一个可变参数，表示将列表中的所有元素作为位置参数传递给函数


# VGG网络定义
def vgg(conv_arch):
    """
    构建VGG网络的函数
    conv_arch: 每个VGG块的结构，格式为[(num_convs, out_channels)]，例如[(2, 64), (2, 128), ...]
    返回一个由卷积块和全连接层组成的VGG模型
    """
    conv_blks = []  # 存储卷积块的列表
    in_channels = 1  # 输入通道为1
    # 逐个创建卷积块
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels  # 更新输入通道数
    return nn.Sequential(
        *conv_blks, nn.Flatten(),  # 展平特征图，传入全连接层
        # 全连接层
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),  # 随机丢弃50%的神经元
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)  # 输出10个类别
    )


# 设置VGG结构，定义卷积块
conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
net = vgg(conv_arch)  # 使用VGG网络

# 打印网络结构
print(net)

# 创建一个示例数据输入，模拟224x224大小的单通道图像
X = torch.randn(size=(1, 1, 224, 224))

# 输出每一层的形状
for blk in net:
    X = blk(X)  # 逐层传入数据
    print(f"{blk.__class__.__name__} output shape:\t{X.shape}")

print("===================================")
# 使用简化的网络训练Fashion-MNIST数据集
print("开始训练模型")

# 下载Fashion-MNIST数据集并进行预处理
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  #shuffle=True表示每次迭代时数据集都会被随机打乱
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 定义损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

import signal
import sys
# 定义一个函数来处理中断信号
def handle_interrupt(sig, frame):
    print('Interrupt signal received, saving model...')
    torch.save(net.state_dict(), 'interrupted_model.pth')
    print('Model saved as interrupted_model.pth')
    sys.exit(0)

# 注册中断信号处理器
signal.signal(signal.SIGINT, handle_interrupt)  #signal.signal()方法注册中断信号处理器，当接收到中断信号时，会调用handle_interrupt函数

num_epochs = 10
try:
    for epoch in range(num_epochs):
        net.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 清零梯度
            outputs = net(inputs)  # 前向传播
            loss = loss_fn(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)  # size(0)返回的是张量维度的大小，即张量中元素的个数

            correct += (predicted == labels).sum().item()  # 计算预测正确的数量

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

except Exception as e:
    print(f'An error occurred: {e}')
    torch.save(net.state_dict(), 'interrupted_model.pth')
    print('Model saved as interrupted_model.pth due to an error')
    sys.exit(1) # 退出程序
'''
labels.size(0)与batch_size的关系：
是的，labels.size(0)的大小确实等于train_loader的batch_size。
这是因为DataLoader在迭代数据集时，会将数据集分成多个批次，每个批次包含batch_size个样本。
因此，在每个批次中，labels张量的第一维大小（即labels.size(0)）就是该批次的样本数量，也就是batch_size。
'''


        # 测试模型
net.eval()  # 设置为评估模式
correct = 0
total = 0
with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

print("训练完成")
