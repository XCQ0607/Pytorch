import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 打印课程目录
print("7.1. 深度卷积神经网络（AlexNet）")

# Step 1: 数据集准备
# Fashion-MNIST数据集加载
# 这里使用了一个数据预处理步骤，通过rescale调整图片大小至224x224（与AlexNet的要求一致）
print("\n加载Fashion-MNIST数据集并进行预处理...")

transform = transforms.Compose([
    transforms.Resize(224),  # 将图像大小调整为224x224
    transforms.ToTensor(),   # 转换为Tensor格式
    transforms.Normalize(mean=[0.5], std=[0.5])  # 对图像进行标准化
])

train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

# 设置num_workers为0，以避免Windows上多进程的问题
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

# Step 2: 构建AlexNet模型
print("\n构建AlexNet模型...")

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),  # 第一层卷积，96个卷积核
            nn.ReLU(),  # 激活函数ReLU
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层

            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 第二层卷积
            nn.ReLU(),  # 激活函数ReLU
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层

            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 第三层卷积
            nn.ReLU(),  # 激活函数ReLU

            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # 第四层卷积
            nn.ReLU(),  # 激活函数ReLU

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 第五层卷积
            nn.ReLU(),  # 激活函数ReLU
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平层，将3D输出转换为2D
            nn.Linear(6400, 4096),  # 第一个全连接层，输出4096个节点
            nn.ReLU(),  # 激活函数ReLU
            nn.Dropout(0.5),  # Dropout层，50%的节点会被随机丢弃
            nn.Linear(4096, 4096),  # 第二个全连接层，输出4096个节点
            nn.ReLU(),  # 激活函数ReLU
            nn.Dropout(0.5),  # Dropout层
            nn.Linear(4096, num_classes)  # 输出层，10个分类
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 实例化模型并确保将模型移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AlexNet().to(device)

# Step 3: 选择损失函数和优化器
print("\n选择损失函数和优化器...")
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于分类问题
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 使用Adam优化器

# Step 4: 训练模型
print("\n开始训练模型...")

def train(model, train_loader, loss_fn, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for inputs, labels in train_loader:
            # 将输入和标签数据放到GPU上（如果有的话）
            inputs, labels = inputs.to(device), labels.to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 计算损失和精度
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc*100:.2f}%, Time: {epoch_time:.2f} sec")

# 启动训练
if __name__ == '__main__':
    train(net, train_loader, loss_fn, optimizer, num_epochs=10)

# Step 5: 评估模型
print("\n评估模型性能...")

def evaluate(model, test_loader):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 将输入和标签数据放到GPU上（如果有的话）
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算精度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 启动评估
evaluate(net, test_loader)

# 代码总结部分
print("\n------------------ 代码总结 ------------------")
print("""
1. 数据加载：
   - 使用torchvision加载Fashion-MNIST数据集，并进行必要的图像预处理。
   - 使用transforms.Resize进行图像尺寸调整，transform.ToTensor将图像转换为Tensor，transform.Normalize进行标准化。

2. 模型定义：
   - AlexNet类是继承自nn.Module的自定义模型。
   - 主要结构包括卷积层、ReLU激活、池化层、全连接层以及Dropout层。

3. 训练：
   - 使用Adam优化器和交叉熵损失函数。
   - 在每个epoch，输出训练损失、准确率以及训练时间。

4. 评估：
   - 使用测试数据集进行模型评估并计算测试准确率。

函数参数说明：
- nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)：卷积层
  - in_channels：输入通道数
  - out_channels：输出通道数
  - kernel_size：卷积核大小
  - stride：步幅
  - padding：填充

- nn.Linear(in_features, out_features)：全连接层
  - in_features：输入特征数
  - out_features：输出特征数

- nn.Dropout(p)：Dropout层
  - p：丢弃概率

- nn.CrossEntropyLoss()：交叉熵损失函数，用于多类分类任务

- torch.optim.Adam(model.parameters(), lr)：Adam优化器
  - model.parameters()：模型的所有参数
  - lr：学习率

------------------ 代码总结完毕 ------------------
""")
