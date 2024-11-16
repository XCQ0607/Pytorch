print("3.6. softmax回归的从零开始实现")
print("="*50)

# 导入必要的库
import torch
from torch import nn
from torchvision import datasets, transforms     # 导入数据集和图像转换工具
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
train_dataset = datasets.FashionMNIST(  #加载训练集，root指定数据集存储路径，train=True表示加载训练集，download=True表示如果数据集不存在则下载
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
train_loader = DataLoader(  #创建数据加载器，dataset指定数据集，batch_size指定每个批次的样本数量，shuffle=True表示打乱数据顺序
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
    '''
X.max(dim=1, keepdim=True) 是对张量 X 进行操作，下面是对各部分的详细解释：
X.max(dim=1, keepdim=True)：
X 是一个张量（通常是二维张量，矩阵）。
dim=1 表示我们希望沿着第二个维度（即每一行的元素）进行最大值的计算。
如果 X 的形状是 (m, n)，那么 dim=1 会对每一行进行操作，计算每行的最大值。
keepdim=True 表示保持输出张量的维度与输入张量相同，结果会保留原始维度，只是最大值所在的维度会变为 1。换句话说，这样的操作会将输出的维度保持为 (m, 1)，而不是将其压缩成一维。
[0]：
X.max(dim=1) 返回一个包含两个元素的元组，第一个元素是每行的最大值，第二个元素是最大值的索引。
[0] 选择的是第一个元素，即每行的最大值，而不关心最大值的索引。
总结：
这行代码的作用是：
对张量 X 沿着第二维（即每一行）计算最大值。
结果是一个新的张量 X_max，其形状是 (m, 1)，每个元素是原张量每行的最大值。
例如，假设 X 是一个 3x4 的张量：
X = torch.tensor([[1, 3, 2, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
那么 X.max(dim=1, keepdim=True) 的结果是：
X_max = X.max(dim=1, keepdim=True)[0]
X_max 将是:
tensor([[4],
        [8],
        [12]])
每行的最大值被保留在了一个新的张量中，并且维度仍然是 (3, 1)。
    '''
    # 计算指数
    X_exp = torch.exp(X - X_max)
    # 计算每行的和
    partition = X_exp.sum(dim=1, keepdim=True)
    # 计算softmax值
    return X_exp / partition  # 返回每行和为1的概率分布


'''
这段代码实现的是 Softmax 函数的计算，通常用于分类任务中，将原始的得分转换为概率分布。接下来，我们逐行解析这些操作：
1. 减去每行的最大值：
X_exp = torch.exp(X - X_max)
这行代码是为了提高数值稳定性。为什么要这样做呢？
数值稳定性：在计算 Softmax 时，我们通常会遇到指数计算。若某些元素的值非常大，计算指数时可能会导致溢出（例如，指数值过大）。为了避免这种情况，我们通常会从每个输入值中减去每行的最大值。这不会影响最终的结果，因为 Softmax 的输出是关于比例的，但是它能避免计算中可能出现的数值问题。
如何做：假设 X_max 是每行的最大值，X - X_max 就是每个元素减去该行的最大值。这样做后，指数函数的输入将会被缩小，避免了指数溢出。
例如，假设 X 是：
X = torch.tensor([[1000, 1001, 999],
                  [3, 3, 3]])
然后计算 X_max，每行的最大值是 1001 和 3：
X_max = X.max(dim=1, keepdim=True)  # 每行最大值
# X_max = tensor([[1001],
#                 [ 3]])
通过 X - X_max，我们得到：
X - X_max = tensor([[-1,  0, -2],
                    [ 0,  0,  0]])
然后再计算指数：
X_exp = torch.exp(X - X_max)
# X_exp = tensor([[0.3679, 1.0000, 0.1353],
#                 [1.0000, 1.0000, 1.0000]])

2. 计算每行的和：
partition = X_exp.sum(dim=1, keepdim=True)
这行代码是计算每行 X_exp 的和，即对每一行求和。partition 将存储每行元素的总和。
dim=1 表示按行求和。
keepdim=True 保持输出维度与原始张量一致，以便后续操作可以广播到正确的形状。
例如，在上面的例子中：
partition = X_exp.sum(dim=1, keepdim=True)
# partition = tensor([[1.5033],
#                     [3.0000]])

3. 计算 Softmax：
return X_exp / partition
这行代码最终计算 Softmax 值。每个元素的 Softmax 值等于该元素的指数除以该行的总和。结果是每行的元素和为 1，形成一个有效的概率分布。
例如，继续使用上面的 X_exp 和 partition：
softmax = X_exp / partition
# softmax = tensor([[0.2447, 0.6652, 0.0901],
#                   [0.3333, 0.3333, 0.3333]])
总结：
这段代码的作用是实现 Softmax 函数，具体步骤如下：
数值稳定性：通过从每行的元素中减去该行的最大值，防止指数计算中的溢出问题。
计算指数：对调整后的张量进行指数运算，得到指数值。
归一化：通过计算每行指数的总和，并将每个元素的指数除以该行的总和，最终得到一个概率分布。
Softmax 函数的输出每行的元素总和为 1，表示该行每个元素的概率，通常用于多分类问题的输出层。
'''
print("定义模型")
print("-"*50)

def net(X): # 定义softmax回归模型，返回logits
    """
    定义softmax回归模型，返回logits
    参数：
    - X: 输入张量，形状为 (batch_size, 1, 28, 28)
    返回：
    - logits张量，形状为 (batch_size, num_outputs)
    """
    # 将输入展平为二维张量 (batch_size, 784)
    X = X.reshape(-1, num_inputs)   #-1表示自动计算维度，将输入展平为二维张量
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

这个 accuracy 函数的作用是计算模型预测的准确性，具体是统计预测正确的样本数量。它接受两个输入：
y_hat: 这是模型的输出，通常是一个 logits 张量，形状为 (batch_size, num_classes)，表示模型对每个类别的预测值。这里的“logits”指的是未经激活函数处理的原始分数或输出，通常会传入softmax激活函数转化为概率分布。
y: 这是实际的标签，形状为 (batch_size)，每个元素是一个整数，表示对应样本的真实类别。

解释步骤：
preds = y_hat.argmax(dim=1):
y_hat.argmax(dim=1) 是为了得到每个样本预测的类别。y_hat 是一个形状为 (batch_size, num_classes) 的张量，它包含每个样本对每个类别的预测分数。argmax(dim=1) 将会在每一行（即每个样本的预测分数）中找出最大值的索引，即模型认为最可能的类别。这些最大值的索引就是模型的预测标签（preds）。
dim=1 表示在每行（每个样本的所有类别）上找出最大值所在的列索引。
cmp = preds.type(y.dtype) == y:
这一步比较预测结果和真实标签，preds 和 y 的类型可能不同，因此使用 preds.type(y.dtype) 将 preds 转换为与 y 相同的类型，确保它们的比较是有效的。
== 会返回一个布尔张量，表示每个样本的预测是否与真实标签一致。如果一致为 True，否则为 False。
float(cmp.type(y.dtype).sum()):
cmp.type(y.dtype) 将布尔类型的张量转换为与 y 相同的类型（通常是 float 类型）。True 会被转换为 1，False 会被转换为 0。
.sum() 会对整个张量求和，得到预测正确的样本数量。因为每个 True 被计为 1，每个 False 被计为 0，所以最终得到的是预测正确的样本总数。
float(...) 用于确保返回值是浮动类型（例如 float 类型的数字，而不是整数），以便支持更精确的计算和进一步的操作。
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
    if isinstance(net, torch.nn.Module):    #如果模型是nn.Module的实例
        net.train()  # 将模型设置为训练模式
    metric = Accumulator(3)  # 损失总和，正确预测数，总预测数  ACCUMULATOR 类用于累加损失、正确预测数和总预测数,这里3个累加器分别用于累加损失、正确预测数和总预测数
    for X, y in train_iter:
        # 计算预测和损失
        y_hat = net(X)
        l = loss(y_hat, y)    #计算损失
        # 梯度清零
        updater.zero_grad()
        # 反向传播
        l.backward()
        # 更新参数
        updater.step()
        # 记录损失和准确率
        metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())    #第一个累加器累加损失，第二个累加器累加正确预测数，第三个累加器累加总预测数
        #numel()返回张量中元素的个数
        #float(l) * len(y): 每个批次的损失乘以批次中样本的数量，累加到第一个累加器中
    # 返回平均损失和精度
    return metric[0]/metric[2], metric[1]/metric[2]    #返回平均损失和精度

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
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)    #训练一个迭代周期，返回平均损失和精度
        test_acc = evaluate_accuracy(net, test_iter)    #计算模型在测试数据上的精度
        print(f"epoch {epoch +1 }, loss {train_metrics[0]:.3f}, "   #打印训练信息，包括当前迭代周期、平均损失、训练精度和测试精度
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



train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
│
├── train_epoch_ch3(net, train_iter, loss, updater)
│   ├── Accuracy & Loss Calculation
│   └── optimizer.zero_grad(), l.backward(), optimizer.step()
│       └── Calls cross_entropy(loss function)
│           └── cross_entropy(logits, y)
│               ├── log_sum_exp = torch.logsumexp(logits, dim=1)
│               └── correct_class_logits = logits[range(len(logits)), y]
│
└── evaluate_accuracy(net, data_iter)
    ├── metric.add(accuracy(y_hat, y), y.numel())
    ├── Accuracy Calculation (Accuracy function)
    └── net.eval()
        └── accuracy(y_hat, y)
            └── preds = y_hat.argmax(dim=1)
树状结构说明
1.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)：
作用：负责整个训练过程的管理。包括多个训练周期（epochs）的迭代，每个周期后输出训练损失、训练准确率和测试准确率。
参数：
net：模型。
train_iter：训练数据迭代器。
test_iter：测试数据迭代器。
loss：损失函数（如交叉熵损失函数）。
num_epochs：训练周期数。
updater：优化器。
功能：
调用 train_epoch_ch3() 进行每个训练周期的训练。
在每个训练周期结束后，调用 evaluate_accuracy() 评估模型在测试集上的准确率。

2.train_epoch_ch3(net, train_iter, loss, updater)：
作用：执行一次训练周期（一个 epoch），计算损失并更新模型参数。
参数：
net：模型。
train_iter：训练数据迭代器。
loss：损失函数。
updater：优化器。
功能：
对每个训练数据批次进行预测（通过模型 net(X)）。
计算损失并反向传播梯度。
更新模型参数（通过优化器 updater.step()）。
返回该周期的平均损失和训练准确率。

3.evaluate_accuracy(net, data_iter)：
作用：评估模型在指定数据集（训练或测试集）上的准确率。
参数：
net：模型。
data_iter：数据迭代器。
功能：
设置模型为评估模式（net.eval()）。
对整个数据集进行预测，计算并返回准确率。

4.cross_entropy(logits, y)：
作用：计算交叉熵损失。这个函数直接操作 logits（未经过 softmax 的原始输出），避免了 softmax 数值不稳定性问题。
参数：
logits：未经过 softmax 的原始模型输出。
y：真实标签。
功能：
计算每个样本的损失，并返回损失的平均值。

5.accuracy(y_hat, y)：
作用：计算模型预测的准确率。
参数：
y_hat：模型的预测输出（通常是 logits）。
y：实际标签。
功能：
计算预测的标签与实际标签的匹配情况，并返回正确预测的数量。



加入 Softmax 后的函数关系树
当我们想要在模型中加入 Softmax 层时，我们通常会将它加在模型的输出层（net() 函数的最后一步）。因为 Softmax 操作会把模型的输出（即 logits）转换为概率分布，而交叉熵损失函数本身已经能够处理未经过 softmax 的 logits，因此只需要在模型输出时加入 softmax。
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)      # 训练模型
│
├── train_epoch_ch3(net, train_iter, loss, updater)   # 训练一个迭代周期，返回平均损失和精度
│   ├── Accuracy & Loss Calculation
│   └── optimizer.zero_grad(), l.backward(), optimizer.step()
│       └── Calls cross_entropy(loss function)
│           └── cross_entropy(logits, y)
│               ├── log_sum_exp = torch.logsumexp(logits, dim=1)
│               └── correct_class_logits = logits[range(len(logits)), y]
│
└── evaluate_accuracy(net, data_iter)     # 计算模型在指定数据集上的精度
    ├── metric.add(accuracy(y_hat, y), y.numel())
    ├── Accuracy Calculation (Accuracy function)
    └── net.eval()
        └── accuracy(y_hat, y)
            └── preds = y_hat.argmax(dim=1)

net(X) ──> softmax(logits)
    │
    └── Returns probabilities (after applying softmax)
更新后的结构树说明
net(X)（模型的前向传播）：
作用：计算模型的输出 logits，并在最后加入 softmax 操作，将 logits 转换为概率分布。
更新：我们在模型的最后一步加上 softmax(logits)，返回的是每个类别的概率，而不是 logits。
def net(X):
    X = X.reshape(-1, num_inputs)
    logits = torch.matmul(X, W) + b
    return softmax(logits)  # Apply softmax to logits
改变：
softmax(logits)：将 logits 转换为概率分布。
此时，net(X) 返回的是每个类别的概率，而不是未经过 softmax 的 logits。
cross_entropy(logits, y)：
作用：计算交叉熵损失。在这个新的结构中，cross_entropy 依然接收 logits（未经 softmax 处理的输出），因为 softmax 操作在模型内部完成。
其余函数的工作流程没有变化，只是在 net(X) 之后，softmax 已经被应用，模型的输出即为概率分布。
总结
函数关系树：
train_ch3 -> train_epoch_ch3 -> cross_entropy -> accuracy -> evaluate_accuracy。
train_ch3 负责整个训练流程，依次调用 train_epoch_ch3 进行每个 epoch 的训练，使用 cross_entropy 计算损失，调用 accuracy 计算准确率，并通过 evaluate_accuracy 在测试集上评估模型。
加入 Softmax 的位置：
在 net(X) 中，最终的模型输出应该是经过 softmax 的概率分布。具体操作是在 logits 计算之后应用 softmax(logits)。
'''