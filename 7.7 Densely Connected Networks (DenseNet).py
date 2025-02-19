import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# 自定义卷积块
def conv_block(input_channels, num_channels):
    """
    定义一个卷积块：批量归一化 + 激活函数（ReLU） + 卷积层
    参数：
    - input_channels: 输入通道数
    - num_channels: 输出通道数
    返回：Sequential模型
    """
    blk = nn.Sequential()
    blk.add_module('bn', nn.BatchNorm2d(input_channels))
    blk.add_module('relu', nn.ReLU())
    blk.add_module('conv', nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
    return blk


# 定义DenseBlock
class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels, growth_rate):
        """
        定义DenseBlock，由多个卷积块组成
        参数：
        - num_convs: 每个DenseBlock中的卷积块数
        - num_channels: 每个卷积块的输入通道数
        - growth_rate: 每个卷积块的输出增长率
        """
        super(DenseBlock, self).__init__()
        self.net = nn.ModuleList()
        for _ in range(num_convs):
            self.net.append(conv_block(num_channels, growth_rate))
            num_channels += growth_rate  # 每次增加growth_rate

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # dim=1表示在通道维度上连接
        return X


# 定义TransitionBlock（过渡层）
def transition_block(input_channels, num_channels):
    """
    定义过渡层：包含批量归一化、ReLU、1x1卷积、平均池化
    参数：
    - input_channels: 输入通道数
    - num_channels: 输出通道数
    返回：Sequential模型
    """
    blk = nn.Sequential()
    blk.add_module('bn', nn.BatchNorm2d(input_channels))
    blk.add_module('relu', nn.ReLU())
    blk.add_module('conv', nn.Conv2d(input_channels, num_channels, kernel_size=1))
    blk.add_module('pool', nn.AvgPool2d(2, stride=2))  # 步幅为2的平均池化
    return blk


# 定义DenseNet模型
class DenseNet(nn.Module):
    def __init__(self, growth_rate, num_blocks, num_convs_per_block, num_classes=10):
        """
        定义DenseNet模型
        参数：
        - growth_rate: 增长率（每个卷积块的输出通道数）
        - num_blocks: DenseNet中稠密块的数量
        - num_convs_per_block: 每个稠密块中的卷积层数
        - num_classes: 最后的分类输出类别数
        """
        super(DenseNet, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 创建多个DenseBlock
        num_channels = 64
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = DenseBlock(num_convs_per_block, num_channels, growth_rate)
            self.blocks.append(block)
            num_channels += num_convs_per_block * growth_rate
            if i != num_blocks - 1:
                # 每个块之间添加一个过渡层
                self.blocks.append(transition_block(num_channels, num_channels // 2))
                num_channels //= 2  # 过渡层减少通道数一半

        # 全局平均池化和全连接层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 初始化DenseNet模型
model = DenseNet(growth_rate=12, num_blocks=4, num_convs_per_block=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard记录
writer = SummaryWriter('./logs/densenet')


# 训练函数
def train_model(model, trainloader, criterion, optimizer, num_epochs=20):
    """
    训练DenseNet模型
    参数：
    - model: 需要训练的模型
    - trainloader: 训练数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮次
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 零梯度
            optimizer.zero_grad()

            # 正向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计损失和准确度
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # 写入TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)


# 训练模型
train_model(model, trainloader, criterion, optimizer)

# 打印TensorBoard日志生成提示
print("TensorBoard logs generated. To view them, run:")
print("tensorboard --logdir=logs")
print("Then open http://localhost:6006 in your browser")


# 评估模型
def evaluate_model(model, testloader):
    """
    在测试集上评估模型的表现
    参数：
    - model: 需要评估的模型
    - testloader: 测试数据加载器
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


# 评估模型
evaluate_model(model, testloader)
