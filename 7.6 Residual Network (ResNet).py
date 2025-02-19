import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary   #用于打印模型结构

import torchvision  #用于加载数据集
from torch.utils.data import DataLoader #用于加载数据集
from torchvision import transforms #用于加载数据集


# 1. 定义残差块
class ResidualBlock(nn.Module):
    """
    定义ResNet中的基本残差块（Residual Block）。
    """

    # def __init__(self, input_channels, output_channels, use_1x1conv=False, stride=1):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, stride=1, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)    # 卷积层
        self.bn1 = nn.BatchNorm2d(output_channels)  # 批量归一化层
        self.dropout = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

        # 1x1卷积用于匹配形状
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        '''
1x1卷积在残差块中的作用
1x1卷积在深度学习中，尤其是在卷积神经网络（CNN）的架构中，具有多种重要用途。在残差网络（ResNet）的上下文中，1x1卷积主要用于调整特征图的通道数，以实现跳跃连接（skip connection）时的形状匹配。以下是几个具体的作用和例子：

1. 通道数调整
在残差块中，输入特征图X和经过两个卷积层后的输出特征图Y可能具有不同的通道数。当input_channels不等于output_channels，或者步长（stride）大于1导致特征图尺寸变化时，直接相加会导致形状不匹配。此时，1x1卷积可以用来调整X的通道数，使其与Y的通道数一致，从而能够进行元素级的相加。

例子：
假设输入特征图X有64个通道，而残差块内部卷积层希望输出128个通道的特征图Y。
如果直接相加，通道数不匹配。
通过在跳跃连接中使用1x1卷积（self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=stride)），可以将X的通道数从64调整到128，使其与Y的通道数相匹配。
        '''
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))  # 第一个卷积层
        Y = self.dropout(Y)
        Y = self.bn2(self.conv2(Y))  # 第二个卷积层

        if self.conv3:
            X = self.conv3(X)  # 跳跃连接（如果需要使用1x1卷积）

        Y += X  # 跳跃连接
        return F.relu(Y)  # 输出经过激活函数


# 2. 定义ResNet-18模型
class ResNet(nn.Module):
    """
    定义ResNet-18架构，包括卷积层、残差块、全局平均池化层和全连接层。
    """

    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        # 第一层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差模块
        self.layer1 = self.make_layer(64, 64, 2, stride=1)  # layer1保持64个通道
        self.layer2 = self.make_layer(64, 128, 2, stride=2)  # layer2通道数增加到128
        self.layer3 = self.make_layer(128, 256, 2, stride=2)  # layer3通道数增加到256
        self.layer4 = self.make_layer(256, 512, 2, stride=2)  # layer4通道数增加到512

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        创建多个残差块，第一层使用步幅进行下采样，后续层保持相同的通道数
        """
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, use_1x1conv=True, stride=stride))  # 第一个块，改变尺寸
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))  # 后续块保持尺寸不变
        return nn.Sequential(*layers)
        '''
分析make_layer函数中的use_1x1conv=True设置
在ResNet架构中，make_layer函数负责创建一组残差块（Residual Blocks），这些残差块通常用于处理具有相同或不同通道数的特征图。在您提供的代码片段中，make_layer函数的第一个残差块使用了use_1x1conv=True的设置，而后续的残差块则没有明确指出是否使用1x1卷积。这里，我们详细分析这一设置的原因和目的：

1. 通道数匹配与下采样
通道数匹配：当残差块的输入通道数（in_channels）与输出通道数（out_channels）不同，或者需要改变特征图的尺寸（通过stride参数）时，使用1x1卷积可以帮助调整输入特征图的通道数，使其与输出特征图的通道数相匹配。这是实现跳跃连接（skip connection）时形状匹配的关键步骤。
下采样：在ResNet架构中，通常会在每个阶段的第一个残差块中进行下采样（通过设置stride>1）。下采样会改变特征图的尺寸，但可能不直接改变通道数。然而，为了保持网络内部的一致性，通常会同时调整通道数。此时，1x1卷积不仅用于通道数匹配，还可能与步幅（stride）结合使用，以实现下采样。

2. 为什么只在第一个块使用use_1x1conv=True
在ResNet的设计中，每个阶段（layer）的开始通常需要一个“过渡”残差块来处理通道数和尺寸的变化。这个过渡块使用1x1卷积来调整输入，以便与后续残差块的输出相匹配。
在每个阶段的后续残差块中，由于输入和输出的通道数（out_channels）已经相同，且特征图尺寸也保持不变，因此无需再使用1x1卷积进行通道数调整。这样做可以减少不必要的计算量，提高网络效率。
        '''
    def forward(self, X):
        X = self.relu(self.bn1(self.conv1(X)))  # 第一层
        X = self.maxpool(X)  # 最大池化层

        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)

        X = self.avgpool(X)  # 全局平均池化
        X = torch.flatten(X, 1)  # 展平
        X = self.fc(X)  # 全连接层

        return X



# 3. 定义训练函数和评估函数
class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def __enter__(self):
        # 在这里进行初始化或者准备工作
        print("Initializing Trainer...")
        return self  # 返回当前实例

    def __exit__(self, exc_type, exc_value, traceback):
        # 在退出 `with` 语句时执行清理工作
        print("Exiting Trainer...")
        # 如果有异常，则会在这里进行处理
        if exc_type:
            print(f"An error occurred: {exc_value}")

    def train(self, num_epochs):
        writer = SummaryWriter(log_dir='./logs/resnet_model')  # 创建TensorBoard日志
        self.model.to(self.device)

        for epoch in range(num_epochs):
            self.model.train()  # 切换到训练模式
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 梯度归零
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # 计算精度
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # 每个epoch的训练损失和准确率
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100 * correct / total
            print(f"第 {epoch + 1}/{num_epochs} 轮，训练损失：{epoch_loss:.4f}，训练精度：{epoch_acc:.2f}%")
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_acc, epoch)

            # 测试数据
            self.evaluate(epoch, writer)

    def evaluate(self, epoch, writer):
        self.model.eval()  # 切换到评估模式
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                # 计算精度
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / len(self.test_loader)
        test_acc = 100 * correct / total
        print(f"第 {epoch + 1} 轮，测试损失：{test_loss:.4f}，测试精度：{test_acc:.2f}%")
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)


def fashionMNIST_loader(batch_size, resize=None):
    """
    加载并返回Fashion-MNIST数据集。
    :param batch_size: 每个批次的样本数量
    :param resize: 可选参数，图片缩放到指定大小
    :return: 返回训练和测试数据的DataLoader
    """
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform)

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter
# 4. 创建数据加载器（假设fashionMNIST_loader定义好了）

# 5. 使用 ResNet 训练模型
if __name__ == '__main__':
    # 配置参数
    BATCH_SIZE = 256
    EPOCHS_NUM = 30
    LEARNING_RATE = 0.01

    # 创建ResNet模型
    model = ResNet(num_classes=10)
    train_loader, test_loader = fashionMNIST_loader(BATCH_SIZE, resize=96)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    #optimizer = optim.SGD(model.parameters(), LEARNING_RATE, weight_decay=1e-4)  # weight_decay是L2正则化的参数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用Trainer进行训练
    with Trainer(model, train_loader, test_loader, criterion, optimizer, device) as trainer:    #使用with语句可以自动调用__enter__和__exit__方法
        trainer.train(EPOCHS_NUM)

# 6. 使用 torchinfo 输出模型概况
print("模型的概况：")
summary(model, input_size=(BATCH_SIZE, 1, 96, 96))
