print("4.2. 多层感知机的从零开始实现")
print("="*50)

# 导入必要的库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 设置设备为GPU（如果可用）或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义超参数
num_inputs = 784  # 输入层节点数 (28x28像素展开成784维向量)
num_outputs = 10  # 输出层节点数 (10个类别)
num_hiddens = [256, 128]  # 隐藏层节点数列表，可根据需要添加更多层
num_epochs = 10  # 训练轮数
batch_size = 256  # 批量大小
learning_rate = 0.1  # 学习率

print(f"使用的设备: {device}")
print(f"超参数设置: num_hiddens={num_hiddens}, num_epochs={num_epochs}, learning_rate={learning_rate}")
print("="*50)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)  # 标准化
])

train_dataset = datasets.FashionMNIST(
    root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("数据集加载完成。")
print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")
print("="*50)

# 定义ReLU激活函数
def relu(X):
    """ReLU激活函数

    参数:
        X (Tensor): 输入张量

    返回:
        Tensor: 应用ReLU激活后的张量
    """
    return torch.maximum(X, torch.zeros_like(X))

# 定义多层感知机模型
class MLP(nn.Module):
    """多层感知机模型

    参数:
        num_inputs (int): 输入层节点数
        num_hiddens (list): 隐藏层节点数列表
        num_outputs (int): 输出层节点数
    """
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        input_size = num_inputs

        # 动态创建隐藏层
        for hidden_size in num_hiddens:
            layer = nn.Linear(input_size, hidden_size)
            self.layers.append(layer)
            input_size = hidden_size

        # 输出层
        self.output_layer = nn.Linear(input_size, num_outputs)

    def forward(self, X):
        X = X.view(-1, num_inputs)
        for layer in self.layers:
            X = relu(layer(X))
        X = self.output_layer(X)
        return X

# 初始化模型、损失函数和优化器
model = MLP(num_inputs, num_hiddens, num_outputs).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print("模型结构:")
print(model)
print("="*50)

# 训练和评估函数
def train(model, train_loader, optimizer, loss_function):
    """训练模型

    参数:
        model (nn.Module): 待训练的模型
        train_loader (DataLoader): 训练数据加载器
        optimizer (Optimizer): 优化器
        loss_function (Loss): 损失函数
    """
    model.train()
    total_loss = 0
    total_correct = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = output.argmax(dim=1)
        total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / len(train_loader.dataset)
    print(f"训练集 - 平均损失: {avg_loss:.4f}, 准确率: {avg_acc*100:.2f}%")

def evaluate(model, test_loader, loss_function):
    """评估模型

    参数:
        model (nn.Module): 待评估的模型
        test_loader (DataLoader): 测试数据加载器
        loss_function (Loss): 损失函数
    """
    model.eval()
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_function(output, y)
            total_loss += loss.item()
            predictions = output.argmax(dim=1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / len(test_loader)
    avg_acc = total_correct / len(test_loader.dataset)
    print(f"测试集 - 平均损失: {avg_loss:.4f}, 准确率: {avg_acc*100:.2f}%")
    return avg_acc

# 开始训练
best_acc = 0.0
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    train(model, train_loader, optimizer, loss_function)
    acc = evaluate(model, test_loader, loss_function)
    if acc > best_acc:
        best_acc = acc
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
    print("-"*50)

print(f"训练完成。最佳测试准确率: {best_acc*100:.2f}%")
print("="*50)

# 可视化部分测试结果
import matplotlib.pyplot as plt

# 定义标签名称
labels_map = {
    0: "T恤/上衣",
    1: "裤子",
    2: "套衫",
    3: "连衣裙",
    4: "外套",
    5: "凉鞋",
    6: "衬衫",
    7: "运动鞋",
    8: "包",
    9: "踝靴",
}

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 获取一些测试数据
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data, example_targets = example_data.to(device), example_targets.to(device)

with torch.no_grad():
    output = model(example_data)
preds = output.argmax(dim=1)

# 显示图像及其预测结果
fig = plt.figure(figsize=(12, 6))
for i in range(1, 13):
    plt.subplot(3, 4, i)
    plt.tight_layout()
    plt.imshow(example_data[i][0].cpu(), cmap='gray', interpolation='none')
    plt.title(f"真值: {labels_map[example_targets[i].item()]}\n预测: {labels_map[preds[i].item()]}")
    plt.xticks([])
    plt.yticks([])
plt.show()

# 总结
print("总结:")
print("本代码实现了一个多层感知机模型，对Fashion-MNIST数据集进行分类。")
print("我们可以通过调整超参数（如隐藏层数、每层的节点数、学习率）来优化模型性能。")
print("在训练过程中，我们记录了每个epoch的损失和准确率，并保存了最佳模型。")
print("="*50)
