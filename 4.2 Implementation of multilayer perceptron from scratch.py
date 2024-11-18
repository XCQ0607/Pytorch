print("4.2. 多层感知机的从零开始实现")
print("="*50)

# 导入必要的库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import rcParams
# 设置绘图参数 微软雅黑 Microsoft YaHei
rcParams['font.family'] = 'Microsoft YaHei'

# 设置设备为GPU（如果可用）或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义超参数
num_inputs = 784  # 输入层节点数 (28x28像素展开成784维向量)
num_outputs = 10  # 输出层节点数 (10个类别)
num_hiddens = [256, 128]  # 隐藏层节点数列表，可根据需要添加更多层

# 在深度学习中，num_hiddens = [256, 128] 代表的是一个神经网络的隐藏层结构配置。具体来说，这表示神经网络有两个隐藏层，它们的节点数分别为：
# 第一个隐藏层有 256 个神经元。
# 第二个隐藏层有 128 个神经元。
# 256 和 128 是神经元（节点）数：
# 这些数字指定了每一层中有多少个神经元。神经元是计算单元，它们通过激活函数将输入信号转换为输出信号。每一层的神经元数量可以影响模型的表达能力、计算复杂度和过拟合风险。
# 层与层之间的关系：
# 这两个数字表示的是两层隐藏层的结构，它们并不直接“联系”，但是它们会影响神经网络的学习能力和训练效果。网络中的每一层都处理来自上一层的输出，并通过权重和偏置进行加权求和后传递给激活函数。
# 第一层的 256 个神经元会处理来自输入层的数据，并将结果传递给第二层，第二层有 128 个神经元，它们会进一步处理这些信息并传递给输出层。
# 层的设计原则：
# 更多的神经元（如 256）通常能让网络捕捉到更多的数据特征，从而增加模型的表示能力，但也可能增加计算成本和过拟合的风险。
# 较少的神经元（如 128）可能会让网络更简洁、计算更高效，但有时也会导致表达能力不足，从而降低模型的性能。
# 通常，网络的结构设计（包括每层的神经元数目）需要通过实验和调优来找到最优的配置。
# 如何选择这些数值：
# 层数（hidden layers）和每层的神经元数目取决于任务的复杂性和数据的特性。
# 对于复杂的任务（例如图像识别、语言处理等），可能需要更多的隐藏层和更多的神经元。
# 对于简单任务，可能不需要很多层和神经元，较少的层和神经元可以有效地减少过拟合。

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
class MLP(nn.Module):   # 继承自nn.Module
    """多层感知机模型

    参数:
        num_inputs (int): 输入层节点数
        num_hiddens (list): 隐藏层节点数列表
        num_outputs (int): 输出层节点数
    """

    '''
super(MLP, self).__init__() 是用来调用父类构造函数的一个特殊写法。
1. super(MLP, self).__init__() 中的 super()
super() 是一个 Python 内置函数，它用于调用父类（超类）的方法。在类继承结构中，super() 可以让你调用父类的方法，而不需要直接指定父类的名称。
super(MLP, self) 表示我们要调用的是 MLP 类的父类（即 nn.Module）的方法。具体来说，这里的 MLP 是当前类的名字，self 是当前实例对象。因此，super(MLP, self) 实际上返回的是 MLP 的父类 nn.Module，并让你能够访问父类的构造函数或方法。
__init__() 是 nn.Module 类的构造函数（初始化方法），它在 MLP 类实例化时会被调用。通过 super(MLP, self).__init__()，我们确保了 MLP 类能够正确地初始化其继承自 nn.Module 的部分。

2. 为什么要用 super(MLP, self).__init__()，而不是直接 super().__init__()？
super(MLP, self).__init__() 和 super().__init__() 都可以用于调用父类构造函数，但两者在使用上下文上略有不同。
super().__init__() 是 Python 3 中常见的简写方式。它的作用是自动推断出当前类和实例，适用于较简单的类结构，通常当你只继承单一父类时，它会自动推导父类。
super(MLP, self).__init__() 是 Python 2 和 Python 3 中都兼容的写法，能够明确指定当前类 MLP 和当前实例 self。它在更复杂的继承结构中特别有用，尤其是涉及多重继承时，使用 super(MLP, self) 可以避免一些不明确的继承关系问题（特别是在多重继承时，Python 2 的 super() 要求显式指定父类和实例）。在 Python 3 中，尽管 super() 也能推导出当前类和实例，但在某些情况下显式使用 super(MLP, self) 会更具可读性和明确性。
总结：虽然 super().__init__() 也能正常工作，但 super(MLP, self).__init__() 更加显式和健壮，尤其在多重继承或复杂代码中推荐使用。

3. self 是什么？
self 是 Python 中类的实例对象，它引用当前类的实例。具体到这里，self 是 MLP 类的一个实例。当你调用 super(MLP, self).__init__() 时，self 就代表当前的 MLP 对象。
通过 super(MLP, self).__init__()，我们实际上是在调用父类 nn.Module 的构造函数，并且确保父类的初始化代码（例如，nn.Module 会初始化网络的一些内部状态）被执行。这样，MLP 类就继承了 nn.Module 的功能，比如模型参数的注册、模型的 .forward() 方法等。
    '''
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(MLP, self).__init__()  # 调用父类的构造函数，初始化父类的参数
        self.layers = nn.ModuleList()   # 用于存储模型的层,layer是父类的一个属性
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

# 这段代码定义了一个 多层感知机（MLP） 模型，继承自 PyTorch 的 nn.Module。它是一个典型的前馈神经网络，包含多个隐藏层和一个输出层，用于进行预测或分类任务。
# 类的结构与函数
# 1. __init__ 构造函数
#
# def __init__(self, num_inputs, num_hiddens, num_outputs):
#     super(MLP, self).__init__()
#     self.layers = nn.ModuleList()
#     input_size = num_inputs
#     # 动态创建隐藏层
#     for hidden_size in num_hiddens:
#         layer = nn.Linear(input_size, hidden_size)
#         self.layers.append(layer)
#         input_size = hidden_size
#     # 输出层
#     self.output_layer = nn.Linear(input_size, num_outputs)
# 参数：
# num_inputs：输入层的节点数（即输入数据的特征维度）。
# num_hiddens：一个包含隐藏层节点数的列表。例如，[256, 128] 表示网络有两个隐藏层，第一层包含 256 个节点，第二层包含 128 个节点。
# num_outputs：输出层的节点数，通常是分类任务的类别数，或回归任务的输出维度。
# 功能：
# super(MLP, self).__init__()：调用父类 nn.Module 的构造函数，初始化父类的必要参数。
# self.layers = nn.ModuleList()：创建一个空的 ModuleList，它是一个容器，用于存储网络的隐藏层。每个隐藏层是 nn.Linear。
# input_size = num_inputs：初始化输入层大小。
# for hidden_size in num_hiddens：循环遍历隐藏层节点数列表，根据每一层的大小创建 nn.Linear 层，并将每一层添加到 self.layers 中。
# self.output_layer = nn.Linear(input_size, num_outputs)：创建最后一个输出层，将输入的维度（input_size）映射到输出维度 num_outputs。
# 2. forward 方法
# def forward(self, X):
#     X = X.view(-1, num_inputs)
#     for layer in self.layers:
#         X = relu(layer(X))
#     X = self.output_layer(X)
#     return X
# 参数：
# X：输入数据，通常是一个张量，形状为 (batch_size, num_inputs)。每一行是一个样本，num_inputs 是样本的特征维度。
# 功能：
# X = X.view(-1, num_inputs)：通过 view 方法调整输入 X 的形状，使其符合期望的维度。这里 -1 表示自动推断，确保 X 的形状为 (batch_size, num_inputs)。
# for layer in self.layers：遍历每一个隐藏层，将数据 X 输入到每一层，并进行激活函数处理。这里使用了 relu 激活函数。
# relu(layer(X))：对每一层的输出进行 ReLU 激活，ReLU 是一种常用的激活函数，可以增加网络的非线性。ReLU 函数的公式是 f(x) = max(0, x)。
# X = self.output_layer(X)：在经过所有隐藏层的处理后，最后通过输出层得到最终的预测结果。
# return X：返回经过输出层的结果，这就是模型的输出。
# 相关函数
# nn.Module（父类）
# nn.Module 是 PyTorch 中所有神经网络模型的基类，几乎所有的 PyTorch 模型都应该继承自它。它提供了如 __init__、forward 方法等基本功能，方便用户定义和构建网络结构。
# 其中 __init__ 方法用于初始化模型，forward 方法用于定义前向传播。
# nn.Linear（线性层）
# nn.Linear(in_features, out_features)：定义一个全连接层，它接受一个形状为 (batch_size, in_features) 的输入，并输出一个形状为 (batch_size, out_features) 的输出。
# 在这个类中，nn.Linear 被用于创建每一层的线性变换，即输入和输出通过一个权重矩阵相乘并加上偏置。
# relu（激活函数）
# relu(x)：ReLU 激活函数，f(x) = max(0, x)。它将负数部分置零，只保留正数部分。
# 在 forward 方法中，relu(layer(X)) 对每一层的输出应用 ReLU 激活函数。
# view（张量重塑）
# X.view(-1, num_inputs)：用于改变 X 张量的形状，这里 -1 表示自动推断维度，num_inputs 保证每个样本的特征数不变。
# 总结
# MLP 类是一个简单的多层感知机模型，它通过 nn.Linear 定义多个全连接层（隐藏层和输出层）。
# 构造函数 __init__ 初始化了隐藏层和输出层。
# forward 方法定义了前向传播过程，输入数据经过每一层线性变换和 ReLU 激活，最后得到模型的输出。
# 输入输出：
# 输入：形状为 (batch_size, num_inputs) 的张量 X。
# 输出：形状为 (batch_size, num_outputs) 的张量，表示模型的预测结果。

# 初始化模型、损失函数和优化器
model = MLP(num_inputs, num_hiddens, num_outputs).to(device)    #.to(device)将模型的参数和缓冲区转移到指定的设备（如GPU）上。
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   #优化器（Optimizer）是用于更新模型参数的算法。在这个例子中，使用了随机梯度下降（SGD）优化器。

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
    model.train()     # 切换模型为训练模式,.train来自于nn.Module，用于将模型设置为训练模式
    #初始化变量
    total_loss = 0
    total_correct = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)     #将输入和标签数据转移到指定的设备上
        optimizer.zero_grad()     # 清空梯度，防止梯度累积
        output = model(X)     # 前向传播
        loss = loss_function(output, y)     # 计算损失
        loss.backward()     # 反向传播，计算梯度
        optimizer.step()     # 更新模型参数
        #整个流程:      清空梯度-> 前向传播-> 计算损失-> 反向传播-> 更新模型参数
        total_loss += loss.item()     #累加损失
        predictions = output.argmax(dim=1)     # argmax返回dim=1维度上的最大值的索引，即预测的类别
        #output shape是 (batch_size, num_outputs)，argmax返回dim=1维度上的最大值的索引，即预测的类别
        total_correct += (predictions == y).sum().item()     #累加正确的预测数量

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
    model.eval()      # 切换模型为评估模式
    total_loss = 0
    total_correct = 0

    with torch.no_grad():     # 关闭梯度计算，减少内存消耗
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
best_acc = 0.0    # 初始化最佳准确率
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    train(model, train_loader, optimizer, loss_function)    # 训练模型
    acc = evaluate(model, test_loader, loss_function)     # 评估模型
    if acc > best_acc:     # 如果当前模型的准确率高于最佳准确率，更新最佳准确率，并保存模型
        best_acc = acc
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')    # 保存模型的参数，state_dict()返回一个字典，包含模型的所有参数,'best_model.pth'是保存的文件名
    print("-"*50)

print(f"训练完成。最佳测试准确率: {best_acc*100:.2f}%")
print("="*50)

# 可视化部分测试结果
import matplotlib.pyplot as plt

# 定义标签名称
labels_map = {   # 标签名称字典
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
model.load_state_dict(torch.load('best_model.pth'))   # 加载模型参数    从文件中加载模型参数到当前模型中

# 警告是因为你在使用 torch.load() 加载模型时没有显式指定 weights_only=True，而 PyTorch 在未来的版本中计划更改这一行为。警告的核心内容是，torch.load 的默认设置会使用 pickle 来加载数据，而 pickle 模块可能存在执行恶意代码的风险。因此，PyTorch 推荐你显式地使用 weights_only=True 来仅加载模型的权重，而不加载其他可能带有恶意代码的对象。
# 警告解释
# 原因：你使用了 torch.load('best_model.pth')，但没有指定 weights_only 参数。
# 问题：torch.load() 默认会使用 pickle 来加载数据。pickle 是一种序列化工具，可以将 Python 对象保存到磁盘并在需要时加载。但 pickle 也可能会执行存储在文件中的任意代码，存在安全隐患。
# 未来变化：PyTorch 在未来的版本中会将 weights_only 的默认值改为 True，这意味着它将仅加载模型的权重，而不是模型的完整状态，包括模型结构或其他数据。这样可以降低潜在的安全风险。
# 如何解决警告
# 为了避免这个警告，并确保你只加载模型的权重，你应该将 weights_only=True 显式地传递给 torch.load()。此外，你需要确保 best_model.pth 仅包含模型权重，而不包含其他非权重对象。
# 解决方法：
# model.load_state_dict(torch.load('best_model.pth', weights_only=True))
# # 获取一些测试数据
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# example_data, example_targets = example_data.to(device), example_targets.to(device)

with torch.no_grad():     # 关闭梯度计算，减少内存消耗
    output = model(example_data)    # 前向传播
preds = output.argmax(dim=1)     # argmax返回dim=1维度上的最大值的索引，即预测的类别

# 显示图像及其预测结果
fig = plt.figure(figsize=(12, 6))   # 创建一个图形对象，figsize指定图形的宽度和高度
for i in range(1, 13):
    plt.subplot(3, 4, i)      # 创建一个子图，3行4列，第i个子图
    plt.tight_layout()    # 自动调整子图之间的间距，使得子图之间不会重叠
    plt.imshow(example_data[i][0].cpu(), interpolation='none')  #imshow显示图像，interpolation='none'不使用插值
    plt.title(f"真值: {labels_map[example_targets[i].item()]}\n预测: {labels_map[preds[i].item()]}")
    plt.xticks([])    # 不显                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              示x轴刻度
    plt.yticks([])    # 不显示y轴刻度
plt.show()    # 显示图形

# 总结
print("总结:")
print("本代码实现了一个多层感知机模型，对Fashion-MNIST数据集进行分类。")
print("我们可以通过调整超参数（如隐藏层数、每层的节点数、学习率）来优化模型性能。")
print("在训练过程中，我们记录了每个epoch的损失和准确率，并保存了最佳模型。")
print("="*50)
