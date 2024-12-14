# 6.6. 卷积神经网络（LeNet）
print("6.6. 卷积神经网络（LeNet）\n")

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")

# 6.6.1. LeNet

print("6.6.1. LeNet\n")

# 定义LeNet模型使用nn.Sequential
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 第一卷积层
    nn.Sigmoid(),  # Sigmoid激活
    nn.AvgPool2d(kernel_size=2, stride=2),  # 平均汇聚层
    nn.Conv2d(6, 16, kernel_size=5),  # 第二卷积层
    nn.Sigmoid(),  # Sigmoid激活
    nn.AvgPool2d(kernel_size=2, stride=2),  # 平均汇聚层
    nn.Flatten(),  # 展平层
    nn.Linear(16 * 5 * 5, 120),  # 第一个全连接层
    nn.Sigmoid(),  # Sigmoid激活
    nn.Linear(120, 84),  # 第二个全连接层
    nn.Sigmoid(),  # Sigmoid激活
    nn.Linear(84, 10)  # 输出层
)

print("LeNet模型结构:")
print(net)

print("\n通过一个28x28的随机输入张量，查看各层的输出形状:\n")
# 创建一个随机的输入张量，形状为(batch_size=1, channels=1, height=28, width=28)
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
print("输入张量X的形状:", X.shape)

# 逐层通过LeNet并打印每一层的输出形状
for layer in net:
    X = layer(X)
    print(f"{layer.__class__.__name__} 输出形状: {X.shape}")

print("\n" + "-" * 80 + "\n")

# 6.6.2. 模型训练

print("6.6.2. 模型训练\n")

# 数据预处理：将图像转换为Tensor并标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载训练和测试数据集
train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

# 定义数据加载器
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义评估准确率的函数
def accuracy(y_pred, y_true):
    """
    计算预测准确率

    参数:
    - y_pred (torch.Tensor): 模型的预测输出，形状为 (batch_size, num_classes)
    - y_true (torch.Tensor): 真实标签，形状为 (batch_size)

    返回:
    - acc (float): 准确率
    """
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    return correct / y_true.size(0)


# 定义在GPU上评估模型准确率的函数
def evaluate_accuracy_gpu(net, data_loader, device):
    """
    在GPU上计算模型的准确率

    参数:
    - net (nn.Module): 已训练好的模型
    - data_loader (DataLoader): 数据加载器
    - device (torch.device): 设备

    返回:
    - acc (float): 准确率
    """
    net.eval()  # 设置模型为评估模式
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total


# 定义训练函数
def train_model(net, train_loader, test_loader, num_epochs, lr, device):
    """
    训练模型

    参数:
    - net (nn.Module): 要训练的模型
    - train_loader (DataLoader): 训练数据加载器
    - test_loader (DataLoader): 测试数据加载器
    - num_epochs (int): 训练轮数
    - lr (float): 学习率
    - device (torch.device): 设备

    返回:
    - net (nn.Module): 训练好的模型
    """
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        net.train()  # 设置模型为训练模式
        running_loss = 0.0
        running_acc = 0.0
        start_time = time.time()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # 清零梯度
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item() * X.size(0)
            running_acc += accuracy(outputs, y) * X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        test_acc = evaluate_accuracy_gpu(net, test_loader, device)
        end_time = time.time()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
              f"Train Acc: {epoch_acc:.4f}, Test Acc: {test_acc:.4f}, "
              f"Time: {end_time - start_time:.2f} sec")

    print("\n训练完成!\n")
    return net


# 实例化LeNet模型
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

# 定义学习率和训练轮数
learning_rate = 0.9
num_epochs = 10

# 训练LeNet模型
trained_net = train_model(net, train_loader, test_loader, num_epochs, learning_rate, device)

print("\n" + "-" * 80 + "\n")

# 6.6.3. 小结

print("6.6.3. 小结\n")
print("""
卷积神经网络（CNN）是一类使用卷积层的网络。
在卷积神经网络中，我们组合使用卷积层、非线性激活函数和汇聚层。
为了构造高性能的卷积神经网络，我们通常对卷积层进行排列，逐渐降低其表示的空间分辨率，同时增加通道数。
在传统的卷积神经网络中，卷积块编码得到的表征在输出之前需由一个或多个全连接层进行处理。
LeNet是最早发布的卷积神经网络之一。
""")

print("\n" + "-" * 80 + "\n")

# 6.6.4. 练习

print("6.6.4. 练习\n")

# 练习1: 将平均汇聚层替换为最大汇聚层，会发生什么？

print("练习1: 将平均汇聚层替换为最大汇聚层，会发生什么？\n")

# 定义一个新的LeNet模型，将AvgPool2d替换为 MaxPool2d
net_maxpool = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

# 实例化并训练新的LeNet模型
print("使用最大汇聚层替换平均汇聚层后的LeNet模型训练:\n")
trained_net_maxpool = train_model(net_maxpool, train_loader, test_loader, num_epochs, learning_rate, device)

print("\n" + "-" * 80 + "\n")

# 练习2: 尝试构建一个基于LeNet的更复杂的网络，以提高其准确性。

print("练习2: 构建一个更复杂的LeNet网络以提高准确性\n")


class LeNet_Complex(nn.Module):
    def __init__(self):
        super(LeNet_Complex, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 增加通道数和减小卷积核
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # 第一卷积层 + ReLU
        x = self.pool(x)  # 第一汇聚层
        x = self.relu(self.conv2(x))  # 第二卷积层 + ReLU
        x = self.pool(x)  # 第二汇聚层
        x = self.relu(self.conv3(x))  # 第三卷积层 + ReLU
        x = self.pool(x)  # 第三汇聚层
        x = self.flatten(x)  # 展平
        x = self.relu(self.fc1(x))  # 第一个全连接层 + ReLU
        x = self.relu(self.fc2(x))  # 第二个全连接层 + ReLU
        x = self.fc3(x)  # 输出层
        return x


# 实例化并训练更复杂的LeNet模型
net_complex = LeNet_Complex()
print("训练更复杂的LeNet模型:\n")
trained_net_complex = train_model(net_complex, train_loader, test_loader, num_epochs=15, lr=0.01, device=device)

print("\n" + "-" * 80 + "\n")

# 练习3: 调整卷积窗口大小。

print("练习3: 调整卷积窗口大小\n")


# 定义一个新的LeNet模型，允许调整卷积核大小
class LeNet_VarKernel(nn.Module):
    def __init__(self, kernel_size=5):
        super(LeNet_VarKernel, self).__init__()
        padding = (kernel_size - 1) // 2  # 以保持输入尺寸
        self.conv1 = nn.Conv2d(1, 6, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=kernel_size, padding=padding)
        self.pool = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  # 根据输入尺寸调整
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = self.pool(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


# 实例化并训练LeNet模型，使用不同的卷积核大小
print("使用不同卷积核大小的LeNet模型训练:\n")
kernel_sizes = [3, 5, 7]
for ks in kernel_sizes:
    print(f"卷积核大小: {ks}x{ks}")
    net_varkernel = LeNet_VarKernel(kernel_size=ks)
    trained_net_varkernel = train_model(net_varkernel, train_loader, test_loader, num_epochs=10, lr=0.9, device=device)
    print("\n" + "-" * 50 + "\n")

# 练习4: 调整输出通道的数量。

print("练习4: 调整输出通道的数量\n")


# 定义一个新的LeNet模型，允许调整输出通道数量
class LeNet_VarChannels(nn.Module):
    def __init__(self, conv1_out=6, conv2_out=16):
        super(LeNet_VarChannels, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, 5, padding=2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv2_out * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = self.pool(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


# 实例化并训练LeNet模型，使用不同的通道数
print("使用不同输出通道数的LeNet模型训练:\n")
channel_configs = [(6, 16), (12, 32), (24, 64)]
for conv1_out, conv2_out in channel_configs:
    print(f"第一卷积层输出通道: {conv1_out}, 第二卷积层输出通道: {conv2_out}")
    net_varchannels = LeNet_VarChannels(conv1_out=conv1_out, conv2_out=conv2_out)
    trained_net_varchannels = train_model(net_varchannels, train_loader, test_loader, num_epochs=10, lr=0.9,
                                          device=device)
    print("\n" + "-" * 50 + "\n")

# 练习5: 调整激活函数（如ReLU）。

print("练习5: 调整激活函数（如ReLU）\n")


# 定义一个新的LeNet模型，使用ReLU激活函数
class LeNet_ReLU(nn.Module):
    def __init__(self):
        super(LeNet_ReLU, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # 使用ReLU激活函数
        x = self.pool(x)
        x = self.relu(self.conv2(x))  # 使用ReLU激活函数
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.relu(self.fc2(x))  # 使用ReLU激活函数
        x = self.fc3(x)
        return x


# 实例化并训练LeNet模型，使用ReLU激活函数
net_relu = LeNet_ReLU()
print("使用ReLU激活函数的LeNet模型训练:\n")
trained_net_relu = train_model(net_relu, train_loader, test_loader, num_epochs=10, lr=0.9, device=device)

print("\n" + "-" * 80 + "\n")

# 练习6: 调整卷积层的数量。

print("练习6: 调整卷积层的数量\n")


# 定义一个新的LeNet模型，允许调整卷积层数量
class LeNet_VarConvLayers(nn.Module):
    def __init__(self, num_conv_layers=3):
        super(LeNet_VarConvLayers, self).__init__()
        layers = []
        in_channels = 1
        out_channels = 6
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, 5, padding=2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
            out_channels *= 2  # 每增加一层，输出通道数翻倍
        self.conv = nn.Sequential(*layers)
        # 假设输入为28x28，经过3次2x2汇聚后为3x3
        self.fc1 = nn.Linear(out_channels // 2 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 实例化并训练LeNet模型，调整卷积层数量
print("调整卷积层数量的LeNet模型训练:\n")
conv_layers_configs = [2, 3]
for num_conv in conv_layers_configs:
    print(f"卷积层数量: {num_conv}")
    net_varconv = LeNet_VarConvLayers(num_conv_layers=num_conv)
    trained_net_varconv = train_model(net_varconv, train_loader, test_loader, num_epochs=10, lr=0.01, device=device)
    print("\n" + "-" * 50 + "\n")

# 练习7: 调整全连接层的数量。

print("练习7: 调整全连接层的数量\n")


# 定义一个新的LeNet模型，允许调整全连接层数量
class LeNet_VarFCLayers(nn.Module):
    def __init__(self, num_fc_layers=3):
        super(LeNet_VarFCLayers, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential()
        in_features = 16 * 5 * 5
        out_features = 120
        for i in range(num_fc_layers):
            self.fc_layers.add_module(f'fc{i + 1}', nn.Linear(in_features, out_features))
            self.fc_layers.add_module(f'relu{i + 1}', nn.ReLU())
            in_features = out_features
            out_features = max(out_features // 2, 10)  # 防止特征数过小
        self.fc_final = nn.Linear(out_features, 10)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = self.pool(x)
        x = torch.sigmoid(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        x = self.fc_final(x)
        return x


# 实例化并训练LeNet模型，调整全连接层数量
print("调整全连接层数量的LeNet模型训练:\n")
fc_layers_configs = [2, 4]
for num_fc in fc_layers_configs:
    print(f"全连接层数量: {num_fc}")
    net_varfc = LeNet_VarFCLayers(num_fc_layers=num_fc)
    trained_net_varfc = train_model(net_varfc, train_loader, test_loader, num_epochs=10, lr=0.01, device=device)
    print("\n" + "-" * 50 + "\n")

# 练习8: 调整学习率和其他训练细节（例如，初始化和轮数）。

print("练习8: 调整学习率和其他训练细节\n")

print("调整学习率和训练轮数的LeNet模型训练:\n")
learning_rates = [0.1, 0.01, 0.001]
epochs = [5, 10, 15]
for lr in learning_rates:
    for epoch in epochs:
        print(f"学习率: {lr}, 训练轮数: {epoch}")
        net_adjust = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        trained_net_adjust = train_model(net_adjust, train_loader, test_loader, num_epochs=epoch, lr=lr, device=device)
        print("\n" + "-" * 50 + "\n")

# 练习9: 显示不同输入（例如毛衣和外套）时，LeNet第一层和第二层的激活值。

print("练习9: 显示不同输入时，LeNet第一层和第二层的激活值\n")


# 定义函数来获取指定层的激活值
def get_activations(model, layer_indices, x):
    """
    获取指定层的激活值

    参数:
    - model (nn.Module): 模型
    - layer_indices (list): 要获取激活值的层的索引
    - x (torch.Tensor): 输入张量

    返回:
    - activations (dict): 每个指定层的激活值
    """
    activations = {}

    def hook_fn(module, input, output):
        activations[len(activations)] = output.cpu()

    hooks = []
    layers = list(model.children())
    for idx in layer_indices:
        hooks.append(layers[idx].register_forward_hook(hook_fn))

    model(x)

    for hook in hooks:
        hook.remove()

    return activations


# 类别标签
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 获取测试集中两个不同类别的样本
test_iter_single = DataLoader(test_dataset, batch_size=1, shuffle=True)
sample_classes = [2, 4]  # 2: Pullover, 4: Coat
samples = {cls: None for cls in sample_classes}

for X, y in test_iter_single:
    if y.item() in sample_classes and samples[y.item()] is None:
        samples[y.item()] = X
    if all(v is not None for v in samples.values()):
        break

# 获取并显示激活值
for cls, X in samples.items():
    print(f"类别: {class_labels[cls]}")
    activations = get_activations(trained_net, [0, 2], X.to(device))  # 第一卷积层和第二卷积层
    conv1_act = activations[0]
    conv2_act = activations[1]
    print(f"第一卷积层激活形状: {conv1_act.shape}")
    print(f"第二卷积层激活形状: {conv2_act.shape}")
    # 可视化第一卷积层的部分激活图
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    for i in range(6):
        axs[i].imshow(conv1_act[0, i].detach().numpy(), cmap='gray')
        axs[i].axis('off')
    plt.suptitle(f"{class_labels[cls]} - 第一卷积层激活图")
    plt.show()
    # 可视化第二卷积层的部分激活图
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    for i in range(6):
        axs[i].imshow(conv2_act[0, i].detach().numpy(), cmap='gray')
        axs[i].axis('off')
    plt.suptitle(f"{class_labels[cls]} - 第二卷积层激活图")
    plt.show()

print("\n" + "-" * 80 + "\n")

# 总结

print("总结:\n")
print("""
本代码示例涵盖了如何使用PyTorch实现卷积神经网络（LeNet），并在Fashion-MNIST数据集上进行训练和评估。
通过调整LeNet的各个参数，如汇聚层类型、卷积核大小、输出通道数、激活函数、卷积层和全连接层的数量，以及学习率和训练轮数，
可以观察模型性能的变化。此外，通过可视化不同输入样本的激活值，深入理解卷积层在特征提取中的作用。

使用到的主要函数和类：
- nn.Conv2d: 定义二维卷积层。
    参数：
        - in_channels (int): 输入通道数。
        - out_channels (int): 输出通道数。
        - kernel_size (int or tuple): 卷积核的大小。
        - padding (int or tuple, optional): 填充的大小。默认是0。
- nn.Linear: 定义全连接层。
    参数：
        - in_features (int): 输入特征数。
        - out_features (int): 输出特征数。
- nn.Module: 所有神经网络模块的基类。
- nn.Sequential: 按顺序排列一系列网络层。
- nn.Flatten: 将多维张量展平为一维向量。
- nn.Sigmoid: 定义Sigmoid激活函数。
    参数：
        - input (Tensor): 输入张量。
- nn.ReLU: 定义ReLU激活函数。
    参数：
        - input (Tensor): 输入张量。
- nn.AvgPool2d: 定义二维平均汇聚层。
    参数：
        - kernel_size (int or tuple): 汇聚窗口的大小。
        - stride (int or tuple, optional): 步幅。默认与kernel_size相同。
- nn.MaxPool2d: 定义二维最大汇聚层。
    参数与AvgPool2d相同。
- DataLoader: 定义数据加载器。
    参数：
        - dataset (Dataset): 数据集。
        - batch_size (int, optional): 每个批次的样本数。
        - shuffle (bool, optional): 是否打乱数据。
- datasets.FashionMNIST: 下载和加载Fashion-MNIST数据集。
    参数：
        - root (string): 数据集存储的位置。
        - train (bool): 是否加载训练集。
        - download (bool): 是否下载数据。
        - transform (callable, optional): 对样本进行的变换。
- optim.SGD: 定义随机梯度下降优化器。
    参数：
        - params (iterable): 要优化的参数。
        - lr (float): 学习率。
- nn.CrossEntropyLoss: 定义交叉熵损失函数。
- matplotlib.pyplot: 用于绘制图像。
- get_activations函数: 获取指定层的激活值。
    参数：
        - model (nn.Module): 模型。
        - layer_indices (list): 要获取激活值的层的索引。
        - x (torch.Tensor): 输入张量。
    返回:
        - activations (dict): 每个指定层的激活值。

调用示例已在代码中展示，如定义不同的LeNet变体并进行训练和评估。
""")
