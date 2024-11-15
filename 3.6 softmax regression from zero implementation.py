print("3.6. softmax回归的从零开始实现")
print("="*50)

# 导入必要的库
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("加载Fashion-MNIST数据集")
print("-"*50)

# 定义数据集的转换（将图像转换为张量）
transform = transforms.Compose([
    transforms.ToTensor()
])

# 设置批量大小
batch_size = 256

# 加载训练集和测试集
train_dataset = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("初始化模型参数")
print("-"*50)

# 定义输入和输出的维度
num_inputs = 784  # 28*28像素
num_outputs = 10  # 10个类别

# 初始化权重和偏置
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

print("定义softmax函数（带有数值稳定性）")
print("-"*50)

def softmax(X):
    """
    计算softmax函数，输入为二维张量X，返回与X形状相同的张量，表示每个样本的概率分布。
    参数：
    - X: 张量，形状为 (batch_size, num_classes)
    返回：
    - 张量，形状与X相同，表示softmax计算结果
    """
    # 为了数值稳定性，减去每行的最大值
    X_max = X.max(dim=1, keepdim=True)[0]
    # 计算指数
    X_exp = torch.exp(X - X_max)
    # 计算每行的和
    partition = X_exp.sum(dim=1, keepdim=True)
    # 计算softmax值
    return X_exp / partition  # 返回每行和为1的概率分布

print("定义模型")
print("-"*50)

def net(X):
    """
    定义softmax回归模型，返回logits
    参数：
    - X: 输入张量，形状为 (batch_size, 1, 28, 28)
    返回：
    - logits张量，形状为 (batch_size, num_outputs)
    """
    # 将输入展平为二维张量 (batch_size, 784)
    X = X.reshape(-1, num_inputs)
    # 计算线性部分
    logits = torch.matmul(X, W) + b
    return logits  # 返回logits，不经过softmax

print("定义交叉熵损失函数（带有数值稳定性）")
print("-"*50)

def cross_entropy(logits, y):
    """
    计算交叉熵损失，直接使用logits，避免数值不稳定性
    参数：
    - logits: 未经过softmax的logits，形状为 (batch_size, num_classes)
    - y: 实际标签，形状为 (batch_size)
    返回：
    - 标量，所有样本的平均损失
    """
    # 使用log-sum-exp技巧计算每个样本的对数分母
    log_sum_exp = torch.logsumexp(logits, dim=1)
    # 取出正确类别的logits
    correct_class_logits = logits[range(len(logits)), y]
    # 计算交叉熵损失
    loss = log_sum_exp - correct_class_logits
    return loss.mean()  # 返回平均损失

print("定义准确率计算函数")
print("-"*50)

def accuracy(y_hat, y):
    """
    计算预测正确的数量
    参数：
    - y_hat: logits张量，形状为 (batch_size, num_classes)
    - y: 实际标签，形状为 (batch_size)
    返回：
    - 正确预测的数量
    """
    # 如果y_hat是logits，使用argmax获取预测类别
    preds = y_hat.argmax(dim=1)
    # 比较预测类别与实际标签
    cmp = preds.type(y.dtype) == y
    # 返回正确预测的数量
    return float(cmp.type(y.dtype).sum())

print("定义评价函数")
print("-"*50)

def evaluate_accuracy(net, data_iter):
    """
    计算模型在指定数据集上的精度
    参数：
    - net: 模型
    - data_iter: 数据迭代器
    返回：
    - 模型在数据集上的精度
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数，总预测数
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]

print("定义累加器类")
print("-"*50)

class Accumulator:
    """
    在n个变量上累加
    """
    def __init__(self, n):
        self.data = [0.0] * n  # 初始化n个累加器

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # 累加

    def reset(self):
        self.data = [0.0] * len(self.data)  # 重置

    def __getitem__(self, idx):
        return self.data[idx]  # 获取第idx个累加器的值

print("定义训练函数")
print("-"*50)

def train_epoch_ch3(net, train_iter, loss, updater):
    """
    训练模型一个迭代周期
    参数：
    - net: 模型
    - train_iter: 训练数据迭代器
    - loss: 损失函数
    - updater: 更新器（优化算法）
    返回：
    - 平均训练损失，训练精度
    """
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)  # 损失总和，正确预测数，总预测数
    for X, y in train_iter:
        # 计算预测和损失
        y_hat = net(X)
        l = loss(y_hat, y)
        # 梯度清零
        updater.zero_grad()
        # 反向传播
        l.backward()
        # 更新参数
        updater.step()
        # 记录损失和准确率
        metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
    # 返回平均损失和精度
    return metric[0]/metric[2], metric[1]/metric[2]

print("开始训练模型")
print("-"*50)

# 定义学习率和迭代周期
lr = 0.1
num_epochs = 10

# 定义优化器
optimizer = torch.optim.SGD([W, b], lr=lr)

# 定义训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练模型
    参数：
    - net: 模型
    - train_iter: 训练数据迭代器
    - test_iter: 测试数据迭代器
    - loss: 损失函数
    - num_epochs: 迭代周期数
    - updater: 更新器（优化算法）
    """
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"epoch {epoch +1 }, loss {train_metrics[0]:.3f}, "
              f"train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}")

# 开始训练
train_ch3(net, train_loader, test_loader, cross_entropy, num_epochs, optimizer)

print("在测试数据上进行预测")
print("-"*50)

# 获取一批测试数据
X, y = next(iter(test_loader))
true_labels = y
pred_labels = net(X).argmax(dim=1)
# 获取类别名称
def get_fashion_mnist_labels(labels):
    """
    返回Fashion-MNIST数据集的文本标签
    参数：
    - labels: 数字标签张量
    返回：
    - 文本标签列表
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 显示真实标签和预测标签
true_text_labels = get_fashion_mnist_labels(true_labels)
pred_text_labels = get_fashion_mnist_labels(pred_labels)
print("真实标签：", true_text_labels[:10])
print("预测标签：", pred_text_labels[:10])

print("完成所有任务")
print("="*50)

# 总结：
"""
在本示例代码中，我们实现了softmax回归的从零开始实现，包括数据加载、模型定义、损失函数、训练和预测。
- softmax函数：实现了数值稳定性处理，参数为输入张量X，返回概率分布。
- cross_entropy损失函数：直接使用logits计算损失，避免数值不稳定，参数为logits和实际标签，返回平均损失。
- accuracy函数：计算预测的准确率，参数为预测的logits和实际标签，返回正确预测的数量。
- evaluate_accuracy函数：在指定数据集上评估模型的准确率。
- train_epoch_ch3函数：训练模型一个迭代周期，参数为模型、训练数据、损失函数、优化器，返回平均损失和精度。
- train_ch3函数：训练模型指定的迭代周期数，在每个周期后输出损失和精度。

在训练过程中，我们使用了数值稳定的softmax和交叉熵损失函数，避免了指数和对数运算中的数值溢出问题。
"""

# 练习题解答：
"""
1. 本节直接实现了基于数学定义softmax运算的softmax函数。这可能会导致什么问题？
   答：当输入中的数值很大时，计算exp可能导致数值溢出（overflow），例如exp(50)会得到一个非常大的数。

2. 本节中的函数cross_entropy是根据交叉熵损失函数的定义实现的。它可能有什么问题？
   答：如果预测概率中包含0，计算log(0)会导致数值问题（负无穷大），从而影响梯度计算。

3. 请想一个解决方案来解决上述两个问题。
   答：在softmax函数中，通过减去输入的最大值，避免指数过大导致的溢出。
       在cross_entropy函数中，直接使用log-sum-exp技巧计算损失，避免计算log(0)的问题。

4. 返回概率最大的分类标签总是最优解吗？例如，医疗诊断场景下可以这样做吗？
   答：不一定。在某些情况下，尤其是不同错误类型的代价不同（如医疗诊断中漏诊的代价远大于误诊），
       我们需要综合考虑概率和错误代价，可能需要调整决策阈值或使用其他策略。

5. 假设我们使用softmax回归来预测下一个单词，可选取的单词数目过多可能会带来哪些问题?
   答：当类别数量（词汇量）过大时，计算softmax的分母（所有类别的指数和）会导致计算量过大，影响模型的训练效率。
       同时，模型可能难以学习到稀有词汇的概率分布，影响预测效果。
"""
