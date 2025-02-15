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
        super(AlexNet, self).__init__() # 调用父类的构造函数
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),  # 第一层卷积，96个卷积核
            nn.ReLU(),  # 激活函数ReLU
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层
            #最大池化层（nn.MaxPool2d）的主要作用是减少数据的空间维度（即高度和宽度），但它不会改变数据的通道数（channels）。因此，经过最大池化层后，数据的通道数保持不变。

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
'''
特征提取部分（self.features）：
这部分通常由多个卷积层（nn.Conv2d）、激活函数层（nn.ReLU）和池化层（nn.MaxPool2d）组成。
它的主要作用是提取输入图像的特征，将原始图像数据转换为高层次的特征表示。
在AlexNet中，self.features包含了五层卷积层，每层卷积层后面都跟着一个ReLU激活函数，部分卷积层后面还跟着最大池化层。

分类器部分（self.classifier）：
这部分通常由全连接层（nn.Linear）、激活函数层（如nn.ReLU或nn.Dropout）和最终的分类层（如nn.Softmax或配合nn.CrossEntropyLoss使用的nn.Linear）组成。
它的主要作用是对特征提取部分输出的特征进行分类，输出每个类别的概率或得分。
在提供的代码片段中，self.classifier的具体内容被省略了，但通常它会包含几个全连接层，最后一个是输出层，其神经元数量与分类任务的类别数相匹配。
'''

def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x

# 实例化模型并确保将模型移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AlexNet().to(device)  # 将模型移到GPU

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

        for inputs, labels in train_loader:  # 遍历训练数据
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
            """
optimizer.step() 的作用如下：

参数更新：
在每次训练迭代中，计算完损失（loss）并调用 loss.backward() 后，每个参数的梯度（gradient）会被计算出来并存储在参数的 .grad 属性中。
optimizer.step() 会根据这些梯度以及优化器自身的算法（如 SGD、Adam 等）来更新模型的参数，即调整权重以减小损失。
优化器状态更新：
某些优化器（如 Adam）在更新参数时还会维护一些内部状态（如动量、方差等），optimizer.step() 也会更新这些状态。
训练过程中的关键步骤：
在典型的训练循环中，optimizer.step() 通常紧跟在 loss.backward() 之后，并且在 optimizer.zero_grad() 之前（用于清除梯度，防止梯度累加）。
            """

            # 计算损失和精度
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)   # 获取预测结果,最大的lable概率值不记录，记录的是label的索引，这里的1是维度，即列数。0是行数，1是列数,2是第三维
            """
为什么选择维度1：

在分类任务中，模型的输出outputs通常是一个二维张量，其形状为(batch_size, num_classes)。
每一行代表一个样本对所有类别的预测概率。
每一列代表一个类别在所有样本中的预测概率。
因此，要选择每个样本预测概率最大的类别，就需要沿着维度1（即列）来找出最大值。

            """
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
