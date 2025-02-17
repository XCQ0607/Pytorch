import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
# 7.5. 批量规范化
print("7.5. 批量规范化")


# ----------------------------------------------------------
# 1. 批量规范化函数 (batch_norm)
# ----------------------------------------------------------
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    批量规范化函数
    :param X: 输入数据，形状为 (batch_size, num_features) 或 (batch_size, num_channels, height, width)
    :param gamma: 拉伸参数，形状与 X 相同
    :param beta: 偏移参数，形状与 X 相同
    :param moving_mean: 移动平均的均值（训练时用），形状与 X 相同
    :param moving_var: 移动平均的方差（训练时用），形状与 X 相同
    :param eps: 为了避免除零错误的一个小常数，通常取 1e-5
    :param momentum: 更新 moving_mean 和 moving_var 的动量系数
    :return: 规范化后的数据，更新后的 moving_mean 和 moving_var
    """
    # 判断是否处于训练模式（是否进行梯度计算）
    if not torch.is_grad_enabled(): # 如果不是训练模式，则直接返回移动平均的均值和方差
        # 在预测模式下，直接使用 moving_mean 和 moving_var
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)    # 计算 X 的标准化形式 X_hat
        #标准化后的数据 X_hat = (X - 均值) / 标准差
        #标准差 = 方差的开方，这里加上了 eps 是为了防止除零错误
    else:
        # 训练模式下，计算当前小批量的均值和方差
        mean = X.mean(dim=0)    # 计算当前小批量的均值，X.mean(dim=0) 表示沿着第一维度（batch_size）求均值
        var = ((X - mean) ** 2).mean(dim=0)  # 计算当前小批量的方差, 这里使用了广播机制，即每个元素减去均值后再平方，再求均值

        '''
X.mean(dim=0) 表示对张量 X 沿着第一维度（即 batch_size 维度）计算均值。
具体来说，如果 X 的形状是 (batch_size, num_features)，那么 X.mean(dim=0) 会计算每个特征在所有样本上的均值，返回一个形状为 (num_features,) 的张量。
如果 X 是四维的，如 (batch_size, num_channels, height, width)，那么 X.mean(dim=0) 会计算每个通道、每个位置在所有样本上的均值，返回一个形状为 (num_channels, height, width) 的张量。

在批量规范化（Batch Normalization）中，我们需要计算当前小批量的均值和方差，以便对数据进行标准化。
通过沿着第一维度（batch_size 维度）求均值，我们可以得到当前小批量数据中每个特征的平均水平，这是进行标准化的重要一步。
标准化后的数据具有更稳定的分布，有助于加速模型的收敛，提高模型的泛化能力。
        '''

        # 使用当前小批量的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)

        # 更新 moving_mean 和 moving_var
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean   # 更新移动平均的均值，更新方式为 动量*移动平均的均值+(1-动量)*当前小批量的均值
        moving_var = momentum * moving_var + (1.0 - momentum) * var # 更新移动平均的方差，更新方式为 动量*移动平均的方差+(1-动量)*当前小批量的方差
        '''
为什么这样更新：

平滑性：移动平均均值的目的是为了提供一个更加平滑、稳定的均值估计，以减少数据波动对模型训练的影响。通过引入动量系数，我们可以控制新均值在移动平均中的贡献程度，从而实现平滑过渡。
历史信息保留：momentum * moving_mean 部分保留了历史信息，即上一轮迭代后的移动平均均值。这有助于模型在训练过程中保持对之前数据的记忆，避免因为当前小批量的数据偏差而导致模型训练的不稳定。
当前信息融入：(1.0 - momentum) * mean 部分则融入了当前小批量的均值信息。这使得移动平均均值能够及时反映数据的最新变化，保持与当前数据分布的一致性。
动量系数的作用：动量系数 momentum 的取值范围通常在 [0, 1] 之间。当 momentum 接近 1 时，移动平均均值更新较慢，更多地保留历史信息；当 momentum 接近 0 时，移动平均均值更新较快，更多地反映当前信息。通过调整 momentum 的值，我们可以根据具体任务需求来平衡历史信息和当前信息的权重。
        '''
    # 缩放和移位
    Y = gamma * X_hat + beta    #Y是标准化后的数据，等于 拉伸参数*标准化后的数据+偏移参数
    return Y, moving_mean.data, moving_var.data


# ----------------------------------------------------------
# 2. 批量规范化层 (BatchNorm)
# ----------------------------------------------------------
class BatchNorm(nn.Module):
    """
    批量规范化层：该层用于训练过程中对输入进行标准化
    """

    def __init__(self, num_features, num_dims):
        """
        :param num_features: 全连接层的输出数量或卷积层的输出通道数
        :param num_dims: 如果是全连接层，num_dims=2；如果是卷积层，num_dims=4
        """
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        # 初始化拉伸（gamma）和偏移（beta）参数
        self.gamma = nn.Parameter(torch.ones(shape))  # 拉伸参数，nn.Parameter() 是 PyTorch 中的一个类，用于创建可学习的参数，即可以通过梯度下降进行优化,初始为shape相同的1
        self.beta = nn.Parameter(torch.zeros(shape))  # 偏移参数，初始为shape相同的0

        # 初始化用于计算的均值和方差的移动平均
        self.moving_mean = torch.zeros(shape)   #设置为shape相同的均值，初始值为0
        self.moving_var = torch.ones(shape)     #设置为shape相同的方差，初始值为1

    def forward(self, X):
        """
        前向传播：应用批量规范化
        :param X: 输入数据
        :return: 规范化后的输出数据
        """
        if self.moving_mean.device != X.device: # 如果当前设备与移动平均不一致，则将其移动到当前设备上
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        # 调用 batch_norm 函数进行规范化处理
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# ----------------------------------------------------------
# 3. LeNet模型 - 应用批量规范化
# ----------------------------------------------------------
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            #BN 层在Conv2d层之后，Sigmoid层之前, 卷积层之后，激活函数之前
            nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
            # 平均池化层
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
            nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, X):
        return self.net(X)


# ----------------------------------------------------------
# 4. 模型训练和验证 - 使用Fashion-MNIST数据集
# ----------------------------------------------------------
def load_data_fashion_mnist(batch_size):
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loss, train_correct = 0.0, 0
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (y_hat.argmax(1) == y).sum().item()

        train_lossratio = train_loss / len(train_iter.dataset)
        train_acc = train_correct / len(train_iter.dataset)
        '''
loss.item()：这是当前批次（batch）的损失值，是一个具体的数值，表示当前批次数据通过模型计算后得到的损失。
train_loss：这是在整个训练集上累积的损失总和，通过 train_loss += loss.item() 在每个批次结束后累加得到。
train_lossratio：这是平均训练损失，计算方式为 train_loss / len(train_iter.dataset)，即将累积的总损失除以训练集的总样本数，以得到每个样本的平均损失。

        '''


        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        writer.add_scalar('Loss/train', train_lossratio, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        test_acc = correct / total
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {100 * test_acc:.2f}%')


# ----------------------------------------------------------
# 5. 设置和训练
# ----------------------------------------------------------
print("--------------------")
print("BatchNorm函数示例：")

# 设置训练超参数
batch_size = 256
num_epochs = 10
lr = 0.01

writer = SummaryWriter(log_dir='./logs/batch_model')
# 加载数据
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 实例化模型并训练
net = LeNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ch6(net, train_iter, test_iter, num_epochs, lr, device)

# 关闭TensorBoard writer
writer.close()



print("--------------------")
print("训练过程的学习曲线：")


print("TensorBoard logs generated. To view them, run:")
print("tensorboard --logdir=logs")
print("Then open http://localhost:6006 in your browser")
# ----------------------------------------------------------
# 总结代码所用到的函数：
# ----------------------------------------------------------
# batch_norm:
#  - 必选参数：X (输入数据), gamma (拉伸参数), beta (偏移参数), moving_mean, moving_var
#  - 可选参数：eps (避免除零的小常数, 默认1e-5), momentum (更新参数的动量，默认0.9)
#
# BatchNorm类：
#  - 必选参数：num_features (通道数或特征数), num_dims (全连接层用2，卷积层用4)
#  - 在forward方法中调用batch_norm进行前向传播。
#
# train_ch6:
#  - 必选参数：net (模型), train_iter (训练数据加载器), test_iter (测试数据加载器), num_epochs (训练周期数), lr (学习率), device (设备)
'''
练习：
删除偏置参数： 如果我们在每一层使用批量规范化，是否仍然需要在全连接层或卷积层中使用偏置参数？为什么？

答：不需要。批量规范化已经通过拉伸和偏移参数（gamma 和 beta）取代了偏置的作用，偏置参数可以被移除，因为批量规范化提供了更好的稳定性和学习能力。
比较LeNet在使用和不使用批量规范化情况下的学习率：

使用批量规范化，学习率通常可以设置更高，因为批量规范化使得训练更加稳定。
训练和测试准确率：

训练时，模型能快速收敛，测试准确率通常会更高，批量规范化有助于减少过拟合。
是否每个层都需要进行批量规范化？尝试删除某一层的批量规范化看效果：

在某些情况下，可以不对每一层应用批量规范化。可以通过删除某些层的BN看看其对训练效果的影响。
用批量规范化替换暂退法：

批量规范化和暂退法有不同的作用，前者主要是通过稳定内部分布来加速训练，而后者是通过丢弃神经元来减少过拟合。两者不能完全替代。
分析beta和gamma：

在训练过程中，gamma（拉伸参数）通常会变大，beta（偏移参数）则会根据层的激活值进行调整。
'''