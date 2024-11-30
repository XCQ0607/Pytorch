# 4.6. 暂退法（Dropout）
print("4.6. 暂退法（Dropout）")

import torch
from torch import nn
import torch.nn.functional as F # 导入 PyTorch 的函数库
import matplotlib.pyplot as plt

# ------------------------ 分割线 ------------------------

# 定义一个函数，用于实现暂退法层
def dropout_layer(X, dropout):
    """
    实现暂退法层的函数
    参数：
    - X: 输入张量
    - dropout: 暂退概率，取值范围为[0,1]
    返回：
    - 输出张量，应用了暂退法
    """
    assert 0 <= dropout <= 1, "dropout 概率必须在0到1之间"
    if dropout == 0:    #dropout表示抛弃率
        return X
    elif dropout == 1:
        return torch.zeros_like(X)
    else:
        # 生成与X同形状的掩码矩阵
        mask = (torch.rand(X.shape) > dropout).float()  #输出0/1矩阵，1表示保留，0表示丢弃
        return mask * X / (1.0 - dropout)   # 应用掩码矩阵，并进行缩放

# 测试 dropout_layer 函数
print("\ndropout_layer 函数示例：")
X = torch.arange(16, dtype=torch.float32).reshape((2, 8))   # 生成一个形状为(2, 8)的张量，并将其值设置为0到15
print("输入张量 X：\n", X)

print("\n暂退概率为 0：")
print(dropout_layer(X, 0.0))

print("\n暂退概率为 0.5：")
print(dropout_layer(X, 0.5))

print("\n暂退概率为 1：")
print(dropout_layer(X, 1.0))

# ------------------------ 分割线 ------------------------

# 定义多层感知机模型，包含两个隐藏层和暂退法
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True, dropout1=0.2, dropout2=0.5):
        """
        初始化模型参数
        参数：
        - num_inputs: 输入层大小
        - num_outputs: 输出层大小
        - num_hiddens1: 第一个隐藏层大小
        - num_hiddens2: 第二个隐藏层大小
        - is_training: 是否为训练模式
        - dropout1: 第一个隐藏层的暂退概率
        - dropout2: 第二个隐藏层的暂退概率
        """
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        """
        前向传播函数
        参数：
        - X: 输入张量
        返回：
        - 输出张量
        """
        X = X.view(-1, self.num_inputs)
        H1 = self.relu(self.lin1(X))
        if self.training:
            H1 = dropout_layer(H1, self.dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout2)
        out = self.lin3(H2)
        return out

# ------------------------ 分割线 ------------------------

# 定义训练函数
def train(net, train_loader, test_loader, num_epochs, loss_fn, optimizer):
    """
    训练模型的函数
    参数：
    - net: 神经网络模型
    - train_loader: 训练数据加载器
    - test_loader: 测试数据加载器
    - num_epochs: 训练周期数
    - loss_fn: 损失函数
    - optimizer: 优化器
    """
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        net.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                y_hat = net(X)
                loss = loss_fn(y_hat, y)
                total_loss += loss.item()
            avg_test_loss = total_loss / len(test_loader)
            test_losses.append(avg_test_loss)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    # 绘制训练损失和测试损失曲线
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# ------------------------ 分割线 ------------------------

# 准备数据集（使用Fashion-MNIST数据集）
import torchvision
from torchvision import transforms

# 定义转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载训练和测试数据集
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
batch_size = 256
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------ 分割线 ------------------------

# 初始化模型、损失函数和优化器
num_inputs = 28 * 28
num_outputs = 10
num_hiddens1 = 256
num_hiddens2 = 256

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
          is_training=True, dropout1=0.2, dropout2=0.5)

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

net.apply(init_weights)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# ------------------------ 分割线 ------------------------

# 训练模型
num_epochs = 10
train(net, train_loader, test_loader, num_epochs, loss_fn, optimizer)

# ------------------------ 分割线 ------------------------

# 评估模型在测试集上的准确率
def evaluate_accuracy(net, data_loader):
    """
    计算模型在数据集上的准确率
    参数：
    - net: 神经网络模型
    - data_loader: 数据加载器
    返回：
    - 准确率
    """
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            y_hat = net(X)
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

print(f"\n在测试集上的准确率: {evaluate_accuracy(net, test_loader):.4f}")

# ------------------------ 分割线 ------------------------

# 练习1：改变第一层和第二层的暂退概率，观察结果
print("\n练习1：改变第一层和第二层的暂退概率，观察结果")

# 定义不同的暂退概率组合
dropout_configs = [
    (0.1, 0.1),
    (0.2, 0.5),
    (0.5, 0.2),
    (0.5, 0.5),
]

for dropout1, dropout2 in dropout_configs:
    print(f"\n暂退概率：第一层 {dropout1}, 第二层 {dropout2}")
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
              is_training=True, dropout1=dropout1, dropout2=dropout2)
    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    train(net, train_loader, test_loader, num_epochs=5, loss_fn=loss_fn, optimizer=optimizer)
    print(f"在测试集上的准确率: {evaluate_accuracy(net, test_loader):.4f}")

# 总结：暂退概率的选择会影响模型的性能。一般来说，靠近输入层的暂退概率应较小，靠近输出层的暂退概率可适当增大。

# ------------------------ 分割线 ------------------------

# 练习2：增加训练轮数，比较使用和不使用暂退法的结果
print("\n练习2：增加训练轮数，比较使用和不使用暂退法的结果")

# 使用暂退法
net_dropout = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                  is_training=True, dropout1=0.2, dropout2=0.5)
net_dropout.apply(init_weights)
optimizer_dropout = torch.optim.SGD(net_dropout.parameters(), lr=0.5)
train(net_dropout, train_loader, test_loader, num_epochs=20, loss_fn=loss_fn, optimizer=optimizer_dropout)
acc_dropout = evaluate_accuracy(net_dropout, test_loader)
print(f"使用暂退法的准确率: {acc_dropout:.4f}")

# 不使用暂退法
net_no_dropout = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                     is_training=True, dropout1=0.0, dropout2=0.0)
net_no_dropout.apply(init_weights)
optimizer_no_dropout = torch.optim.SGD(net_no_dropout.parameters(), lr=0.5)
train(net_no_dropout, train_loader, test_loader, num_epochs=20, loss_fn=loss_fn, optimizer=optimizer_no_dropout)
acc_no_dropout = evaluate_accuracy(net_no_dropout, test_loader)
print(f"不使用暂退法的准确率: {acc_no_dropout:.4f}")

# 总结：增加训练轮数时，使用暂退法可以有效防止过拟合，取得更好的泛化性能。

# ------------------------ 分割线 ------------------------

# 练习3：比较应用和不应用暂退法时，每个隐藏层中激活值的方差
print("\n练习3：比较应用和不应用暂退法时，每个隐藏层中激活值的方差")

# 定义一个新模型，用于记录激活值
class NetWithActivations(Net):
    def forward(self, X):
        X = X.view(-1, self.num_inputs)
        H1 = self.relu(self.lin1(X))
        H1_ = H1.clone().detach()
        if self.training:
            H1 = dropout_layer(H1, self.dropout1)
        H2 = self.relu(self.lin2(H1))
        H2_ = H2.clone().detach()
        if self.training:
            H2 = dropout_layer(H2, self.dropout2)
        out = self.lin3(H2)
        return out, H1_, H2_

# 训练模型并记录激活值的方差
def train_with_activation_variance(net, train_loader, num_epochs, loss_fn, optimizer):
    """
    训练模型并记录每个隐藏层的激活值方差
    """
    activation_variances = {'H1': [], 'H2': []}
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            y_hat, H1_, H2_ = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            # 计算激活值的方差
            activation_variances['H1'].append(H1_.var().item())
            activation_variances['H2'].append(H2_.var().item())
    return activation_variances

# 比较有无暂退法的激活值方差
net_with_dropout = NetWithActivations(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                                      is_training=True, dropout1=0.2, dropout2=0.5)
net_with_dropout.apply(init_weights)
optimizer_with_dropout = torch.optim.SGD(net_with_dropout.parameters(), lr=0.5)
activation_variances_with_dropout = train_with_activation_variance(
    net_with_dropout, train_loader, num_epochs=1, loss_fn=loss_fn, optimizer=optimizer_with_dropout)

net_without_dropout = NetWithActivations(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                                         is_training=True, dropout1=0.0, dropout2=0.0)
net_without_dropout.apply(init_weights)
optimizer_without_dropout = torch.optim.SGD(net_without_dropout.parameters(), lr=0.5)
activation_variances_without_dropout = train_with_activation_variance(
    net_without_dropout, train_loader, num_epochs=1, loss_fn=loss_fn, optimizer=optimizer_without_dropout)

# 绘制激活值方差曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(activation_variances_with_dropout['H1'], label='有暂退法')
plt.plot(activation_variances_without_dropout['H1'], label='无暂退法')
plt.title('隐藏层1激活值方差')
plt.xlabel('批次')
plt.ylabel('方差')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(activation_variances_with_dropout['H2'], label='有暂退法')
plt.plot(activation_variances_without_dropout['H2'], label='无暂退法')
plt.title('隐藏层2激活值方差')
plt.xlabel('批次')
plt.ylabel('方差')
plt.legend()
plt.show()

# 总结：应用暂退法时，激活值的方差会增大，因为暂退法引入了随机性。

# ------------------------ 分割线 ------------------------

# 练习4：为什么在测试时通常不使用暂退法？
print("\n练习4：为什么在测试时通常不使用暂退法？")
print("回答：在测试时，我们希望得到确定性的结果，因此不再对神经元进行随机丢弃。"
      "同时，在训练时使用暂退法后，网络的权重已经适应了这种随机性，"
      "在测试时直接使用全部神经元可以获得更好的性能。")

# ------------------------ 分割线 ------------------------

# 练习5：比较使用暂退法和权重衰减的效果，同时使用会发生什么情况？
print("\n练习5：比较使用暂退法和权重衰减的效果，同时使用会发生什么情况？")

# 定义一个函数，训练模型并返回测试集准确率
def train_and_evaluate(net, train_loader, test_loader, num_epochs, loss_fn, optimizer):
    train(net, train_loader, test_loader, num_epochs, loss_fn, optimizer)
    acc = evaluate_accuracy(net, test_loader)
    return acc

# 仅使用权重衰减
net_weight_decay = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                       is_training=True, dropout1=0.0, dropout2=0.0)
net_weight_decay.apply(init_weights)
optimizer_weight_decay = torch.optim.SGD(net_weight_decay.parameters(), lr=0.5, weight_decay=1e-4)
acc_weight_decay = train_and_evaluate(net_weight_decay, train_loader, test_loader, 10, loss_fn, optimizer_weight_decay)
print(f"仅使用权重衰减的准确率: {acc_weight_decay:.4f}")

# 同时使用暂退法和权重衰减
net_both = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
               is_training=True, dropout1=0.2, dropout2=0.5)
net_both.apply(init_weights)
optimizer_both = torch.optim.SGD(net_both.parameters(), lr=0.5, weight_decay=1e-4)
acc_both = train_and_evaluate(net_both, train_loader, test_loader, 10, loss_fn, optimizer_both)
print(f"同时使用暂退法和权重衰减的准确率: {acc_both:.4f}")

# 总结：同时使用暂退法和权重衰减可以取得更好的泛化性能，两者的效果是累加的。

# ------------------------ 分割线 ------------------------

# 练习6：如果将暂退法应用于权重矩阵的各个权重，而不是激活值，会发生什么？
print("\n练习6：将暂退法应用于权重矩阵的各个权重，而不是激活值")

# 定义新的模型，将暂退法应用于权重
class NetDropoutWeights(Net):
    def forward(self, X):
        X = X.view(-1, self.num_inputs)
        W1 = dropout_layer(self.lin1.weight, self.dropout1)
        H1 = self.relu(F.linear(X, W1, self.lin1.bias))
        W2 = dropout_layer(self.lin2.weight, self.dropout2)
        H2 = self.relu(F.linear(H1, W2, self.lin2.bias))
        out = self.lin3(H2)
        return out

net_weight_dropout = NetDropoutWeights(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                                       is_training=True, dropout1=0.2, dropout2=0.5)
net_weight_dropout.apply(init_weights)
optimizer_weight_dropout = torch.optim.SGD(net_weight_dropout.parameters(), lr=0.5)
train(net_weight_dropout, train_loader, test_loader, num_epochs=10, loss_fn=loss_fn, optimizer=optimizer_weight_dropout)
acc_weight_dropout = evaluate_accuracy(net_weight_dropout, test_loader)
print(f"将暂退法应用于权重后的准确率: {acc_weight_dropout:.4f}")

# 总结：将暂退法应用于权重矩阵会导致训练过程不稳定，性能下降。这是因为权重的随机性会直接影响模型的学习能力。

# ------------------------ 分割线 ------------------------

# 练习7：发明另一种用于在每一层注入随机噪声的技术，尝试在Fashion-MNIST上性能优于暂退法的方法
print("\n练习7：在每一层注入高斯噪声")

# 定义一个新的模型，在激活值上添加高斯噪声
class NetGaussianNoise(Net):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True, sigma1=0.1, sigma2=0.1):
        super(NetGaussianNoise, self).__init__(num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def forward(self, X):
        X = X.view(-1, self.num_inputs)
        H1 = self.relu(self.lin1(X))
        if self.training:
            H1 = H1 + torch.randn_like(H1) * self.sigma1
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = H2 + torch.randn_like(H2) * self.sigma2
        out = self.lin3(H2)
        return out

net_gaussian = NetGaussianNoise(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                                is_training=True, sigma1=0.1, sigma2=0.1)
net_gaussian.apply(init_weights)
optimizer_gaussian = torch.optim.SGD(net_gaussian.parameters(), lr=0.5)
train(net_gaussian, train_loader, test_loader, num_epochs=10, loss_fn=loss_fn, optimizer=optimizer_gaussian)
acc_gaussian = evaluate_accuracy(net_gaussian, test_loader)
print(f"使用高斯噪声的准确率: {acc_gaussian:.4f}")

# 总结：在激活值上添加高斯噪声可以作为一种正则化手段，效果与暂退法类似，但实际性能需要根据具体情况进行比较。

# ------------------------ 分割线 ------------------------

# 总结代码示例中使用的函数和参数：

# 1. dropout_layer(X, dropout)
#    - 功能：对输入张量 X 应用暂退法
#    - 参数：
#      - X: 输入张量
#      - dropout: 暂退概率，0 <= dropout <= 1
#    - 示例：
#      X = torch.randn(10, 5)
#      X_dropout = dropout_layer(X, 0.5)

# 2. Net 类
#    - 功能：定义了一个包含暂退法的多层感知机模型
#    - 参数：
#      - num_inputs: 输入层大小
#      - num_outputs: 输出层大小
#      - num_hiddens1: 第一个隐藏层大小
#      - num_hiddens2: 第二个隐藏层大小
#      - is_training: 是否为训练模式
#      - dropout1: 第一个隐藏层的暂退概率
#      - dropout2: 第二个隐藏层的暂退概率
#    - 示例：
#      net = Net(784, 10, 256, 256, True, 0.2, 0.5)

# 3. train 函数
#    - 功能：训练模型并绘制损失曲线
#    - 参数：
#      - net: 神经网络模型
#      - train_loader: 训练数据加载器
#      - test_loader: 测试数据加载器
#      - num_epochs: 训练周期数
#      - loss_fn: 损失函数
#      - optimizer: 优化器
#    - 示例：
#      train(net, train_loader, test_loader, 10, loss_fn, optimizer)

# 4. evaluate_accuracy(net, data_loader)
#    - 功能：评估模型在数据集上的准确率
#    - 参数：
#      - net: 神经网络模型
#      - data_loader: 数据加载器
#    - 示例：
#      acc = evaluate_accuracy(net, test_loader)
