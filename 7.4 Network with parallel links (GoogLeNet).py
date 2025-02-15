import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# 输出目录和子目录的名称
print("7.4. 含并行连结的网络（GoogLeNet）")


# 7.4.1 Inception块的定义
# Inception块包含四条并行路径，每条路径使用不同的卷积核进行特征提取
class Inception(nn.Module):
    """
    Inception模块的定义，包含四条路径：每条路径执行不同大小的卷积操作
    :param in_channels: 输入的通道数
    :param c1: 第一路径的输出通道数
    :param c2: 第二条路径的输出通道数（包含1x1卷积和3x3卷积）
    :param c3: 第三条路径的输出通道数（包含1x1卷积和5x5卷积）
    :param c4: 第四条路径的输出通道数（包含3x3最大池化和1x1卷积）
    """

    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1：1x1卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # 线路2：1x1卷积后接3x3卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # 线路3：1x1卷积后接5x5卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # 线路4：3x3最大池化后接1x1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连接四条路径的输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# 7.4.2 GoogLeNet模型的定义
def create_googlenet_model(num_classes=10):
    """
    创建GoogLeNet模型，包含多个Inception块和全局平均池化层
    :param num_classes: 输出的类别数（例如：Fashion-MNIST的输出类别数为10）
    :return: GoogLeNet模型
    """
    # 各个模块的定义
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, num_classes))
    return net


# 7.4.3 数据加载和训练函数定义
def load_data_fashion_mnist(batch_size, resize=None):
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


def train_model(net, train_iter, test_iter, num_epochs, lr, device):
    """
    训练GoogLeNet模型。
    :param net: 要训练的模型
    :param train_iter: 训练数据的DataLoader
    :param test_iter: 测试数据的DataLoader
    :param num_epochs: 训练的轮数
    :param lr: 学习率
    :param device: 设备（例如：CPU或GPU）
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

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

        train_acc = train_correct / len(train_iter.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_iter)}, Train Accuracy: {train_acc:.4f}")

        net.eval()
        test_correct = 0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                output = net(X)
                test_correct += (output.argmax(1) == y).sum().item()

        test_acc = test_correct / len(test_iter.dataset)
        print(f"Test Accuracy: {test_acc:.4f}")


# 7.4.4 代码演示

print("\n" + "-" * 50 + "\n")

# 创建GoogLeNet模型
num_classes = 10
net = create_googlenet_model(num_classes)

# 打印模型结构
print("GoogLeNet模型结构：")
print(net)

# 设置训练参数
batch_size = 128
num_epochs = 10
lr = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)

# 训练模型
train_model(net, train_iter, test_iter, num_epochs, lr, device)

# 总结：
# 1. Inception: 定义了GoogLeNet中的Inception块，用于不同卷积操作的并行计算。
#    参数：in_channels（输入通道数），c1、c2、c3、c4（各条路径的输出通道数）。
# 2. create_googlenet_model: 创建GoogLeNet模型，包含多个Inception块和全局平均池化层。
#    参数：num_classes（类别数目）。
# 3. load_data_fashion_mnist: 加载Fashion-MNIST数据集并返回数据迭代器。
#    参数：batch_size（批次大小），resize（可选，图片缩放大小）。
# 4. train_model: 用于训练模型，计算损失和准确度。
#    参数：net（模型），train_iter（训练数据迭代器），test_iter（测试数据迭代器），
#          num_epochs（训练轮数），lr（学习率），device（设备：CPU或GPU）。
