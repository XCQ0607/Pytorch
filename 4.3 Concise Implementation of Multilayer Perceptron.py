import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 控制台输出标题
print("4.3. 多层感知机的简洁实现")

# =========================== 数据加载与预处理 ===========================
# 加载 FashionMNIST 数据集，并进行标准化
print("\n加载和预处理数据...")
batch_size = 256
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 均值为0.5，标准差为0.5的正态分布
])
# 加载数据集
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
# 分割训练集和验证集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    # 训练集需要打乱
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # 测试集不需要打乱

print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
print("-" * 50)

# =========================== 模型定义 ===========================
# 自定义多层感知机模型
print("\n定义多层感知机模型...")
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], output_size=10, activation_fn=nn.ReLU): #activation_fn=nn.LeakyReLU是使用LeakyReLU激活函数
        """
        初始化模型
        :param input_size: 输入特征维度
        :param hidden_sizes: 隐藏层单元数量列表
        :param output_size: 输出特征维度（分类类别数量）
        :param activation_fn: 激活函数（默认为ReLU）
        """
        super(MLP, self).__init__()  # 调用父类构造函数
        layers = [nn.Flatten()]  #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              将输入展平
        prev_size = input_size    # 上一层的输出维度
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation_fn())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))  # 输出层
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 创建模型实例
net = MLP(hidden_sizes=[512, 256, 128], activation_fn=nn.LeakyReLU)  # 实例化模型，activation_fn=nn.LeakyReLU是使用LeakyReLU激活函数
print(net)  # 打印模型结构

print("-" * 50)

# =========================== 权重初始化 ===========================
print("\n初始化权重...")
def init_weights(m):    #初始化了一个函数，用于初始化模型的权重
    """
    初始化权重
    :param m: 模型层
    """
    if isinstance(m, nn.Linear):    # 如果是线性层
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')    # 使用Kaiming初始化方法初始化权重，nonlinearity='leaky_relu'表示使用LeakyReLU激活函数
        nn.init.zeros_(m.bias)    # 偏置初始化为0


net.apply(init_weights)  # 应用初始化函数到模型的每一层
print("权重初始化完成")
print("-" * 50)

# =========================== 损失函数和优化器 ===========================
print("\n定义损失函数和优化器...")
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)    # Adam优化器，lr=0.001是学习率
print("损失函数: CrossEntropyLoss")
print("优化器: Adam")
print("-" * 50)

# =========================== 训练和评估函数 ===========================
def train_epoch(net, data_loader, loss_fn, optimizer):
    net.train()    # 训练模式
    total_loss, correct = 0, 0
    for X, y in data_loader:
        optimizer.zero_grad()    # 梯度清零
        y_hat = net(X)         # 前向传播
        loss = loss_fn(y_hat, y)    # 计算损失
        loss.backward()    # 反向传播
        optimizer.step()    # 更新参数
        total_loss += loss.item()    # 累加损失
        correct += (y_hat.argmax(1) == y).sum().item()    # 累加正确预测的样本数
    return total_loss / len(data_loader), correct / len(train_loader.dataset)    # 返回平均损失和准确率

def evaluate(net, data_loader, loss_fn):
    net.eval()        # 评估模式
    total_loss, correct = 0, 0  #
    with torch.no_grad():
        for X, y in data_loader:
            y_hat = net(X)     # 前向传播
            loss = loss_fn(y_hat, y)     # 计算损失
            total_loss += loss.item()     # 累加损失
            correct += (y_hat.argmax(1) == y).sum().item()     # 累加正确预测的样本数
    return total_loss / len(data_loader), correct / len(test_loader.dataset)    # 返回平均损失和准确率

# =========================== 训练模型 ===========================
print("\n开始训练...")
# 初始化列表用于保存训练过程中的损失和准确率
num_epochs = 10
train_losses, test_losses = [], []    # 用于保存训练集和测试集的损失
train_accuracies, test_accuracies = [], []    # 用于保存训练集和测试集的准确率

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(net, train_loader, loss_fn, optimizer)
    test_loss, test_acc = evaluate(net, test_loader, loss_fn)
    train_losses.append(train_loss)   # 保存训练集损失
    test_losses.append(test_loss)     # 保存测试集损失
    train_accuracies.append(train_acc)   # 保存训练集准确率
    test_accuracies.append(test_acc)     # 保存测试集准确率

    print(f"Epoch {epoch + 1}/{num_epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

print("-" * 50)

# =========================== 结果可视化 ===========================
print("\n绘制训练过程...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.show()

# =========================== 总结 ===========================
"""
总结:
1. 使用 nn.Sequential 快速定义多层感知机，包括输入展平、隐藏层和激活函数。
2. 使用 nn.init 模块初始化权重，例如 kaiming_normal_ 初始化。
3. 使用 CrossEntropyLoss 作为分类任务的损失函数。
4. 优化器可以选择如 Adam、SGD 等，参数如学习率需要调试。
5. 模型通过 train_epoch 和 evaluate 函数进行训练和评估。
6. 输出详细的训练日志和可视化结果。
"""
