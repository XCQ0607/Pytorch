# softmax回归

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("softmax回归示例")

# -----------------------分割线-----------------------
# 设置超参数
batch_size = 64  # 每个批次处理的数据样本数量
learning_rate = 0.01  # 学习率，控制参数更新的速度
num_epochs = 5  # 训练循环的次数

# -----------------------分割线-----------------------
# 数据预处理，转换为Tensor并标准化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图片或numpy.ndarray转换为FloatTensor，并且将像素值归一化到[0,1]
    transforms.Normalize((0.5,), (0.5,))  # 标准化，使数据分布均值为0，标准差为1
])

# -----------------------分割线-----------------------
# 下载并加载训练集和测试集  MNIST是一个常用的手写数字数据集, 包含手写数字的灰度图像和对应的标签
train_dataset = datasets.MNIST(root='./data',  # 数据存储的路径
                               train=True,  # 是否是训练集
                               transform=transform,  # 数据预处理方式
                               download=True)  # 如有必要是否下载数据
# download=True表示如果数据不存在，会自动下载,数据是从网上下载的，下载后会存储在指定的./data目录下,目前数据内容是关于手写数字的灰度图像和对应的标签

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transform,
                              download=True)

# 使用DataLoader加载数据集
# DataLoader的参数：
# dataset：加载的数据集
# batch_size：每个批次的样本数
# shuffle：是否在每个epoch开始时打乱数据
# num_workers：使用多少子进程加载数据，默认0表示数据将在主进程中加载
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)  # 加载训练集，shuffle=True表示在每个epoch开始时打乱数据
# 这个数据的参数有input_size，num_classes，batch_size，shuffle，num_workers
# 这些参数的意义是：大小，类别，批量大小，是否打乱数据，加载数据的子进程数

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# -----------------------分割线-----------------------
# 定义Softmax回归模型
class SoftmaxRegression(nn.Module): # 定义Softmax回归类，继承自nn.Module，是一个神经网络的基本模块，用于构建神经网络模型
    def __init__(self, input_size, num_classes):
        #参数：super(类型名, self)用于调用父类的构造函数，确保父类的初始化也被执行
        super(SoftmaxRegression, self).__init__()   # 调用父类的构造函数，继承父类的属性和方法
        # nn.Linear用于定义全连接层（线性变换）
        # in_features：输入特征数，即每个样本的特征维度
        # out_features：输出特征数，即类别数
        # bias：如果设置为False，该层将不会学习偏置bias，默认为True
        self.linear = nn.Linear(in_features=input_size,
                                out_features=num_classes,
                                bias=True)
        #创建了一个全连接层（也称为线性层）的实例，并将其赋值给self.linear属性。nn.Linear是PyTorch中提供的一个类，用于表示神经网络中的全连接层。

    def forward(self, x):   # 定义前向传播函数
        # 将输入x展平成(batch_size, input_size)的形状
        x = x.view(-1, 28*28)     # 展平操作，将(batch_size, 1, 28, 28)的图像数据展平成(batch_size, 28*28)的张量
        # 前向传播，通过线性层
        out = self.linear(x)
        # 注意：在nn.CrossEntropyLoss中已经包含了softmax操作，所以这里不需要手动添加softmax激活函数
        return out  # 返回模型的输出

# 在SoftmaxRegression类中，self.linear是在构造函数__init__中定义的一个属性，它代表了一个全连接层（线性变换层），由nn.Linear类实例化得到。这个全连接层有input_size个输入特征和num_classes个输出特征，并且会学习一个偏置项（因为bias=True）。
# 在forward方法中，self.linear(x)这行代码调用了这个全连接层，并传入了处理后的输入数据x。这里的x已经被展平成了一个形状为(batch_size, 28*28)的张量，以匹配self.linear层所期望的输入形状。
# 因此，forward方法中调用的self.linear(x)与构造函数中初始化的self.linear属性确实有关系。构造函数负责创建和配置这个全连接层，而forward方法则负责在模型的前向传播过程中使用这个层来处理输入数据。
# 简而言之，构造函数中的初始化设置了模型的结构和参数，而forward方法则定义了数据如何通过这个结构进行流动和变换。


# 实例化模型，输入大小为28*28=784像素，类别数为10（数字0-9）
model = SoftmaxRegression(input_size=28*28,
                          num_classes=10)

# -----------------------分割线-----------------------
# 定义损失函数和优化器

# nn.CrossEntropyLoss用于多分类任务的交叉熵损失函数
# 它的参数：
# weight：可选，形状为[classes]，给每个类别的损失赋予不同的权重
# ignore_index：指定一个类别，在计算损失时会忽略该类别的样本
# reduction：指定损失计算方式，'none'|'mean'|'sum'，默认是'mean'求平均
criterion = nn.CrossEntropyLoss()   # 实例化交叉熵损失函数

# 使用随机梯度下降（SGD）优化器
# optim.SGD的参数：
# params：待优化的参数，一般传入model.parameters()
# lr：学习率
# momentum：动量因子，帮助加速优化，默认0
# weight_decay：权重衰减（L2惩罚），默认0
# dampening：动量抑制因子，默认0
# nesterov：布尔值，是否使用Nesterov动量，默认False
optimizer = optim.SGD(model.parameters(),   #   优化器，传入模型的参数
                      lr=learning_rate,
                      momentum=0.9)

# -----------------------分割线-----------------------
# 开始训练模型
print("开始训练模型")
for epoch in range(num_epochs):
    total_loss = 0  # 累积损失
    for i, (images, labels) in enumerate(train_loader): #emumerate函数用于同时获取索引和元素
        # 前向传播，计算模型输出
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度缓存
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

        total_loss += loss.item()  # 累加损失

        # 每100个批次打印一次训练信息
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)  # 计算平均损失
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

print("模型训练完成")

# -----------------------分割线-----------------------
# 在测试集上评估模型
print("开始评估模型")
model.eval()  # 将模型设置为评估模式    # 评估模式，不会计算梯度和更新参数
#评估模式这个属性是来自于nn.Module类，它是一个布尔值，表示模型是否处于评估模式。在评估模式下，模型不会计算梯度，也不会更新参数。这在测试集上进行模型评估时非常有用，因为我们不需要计算梯度和更新模型。
#除了eval(评估)，还有train(训练)模式，启用它会启用dropout和batch normalization等训练时才需要的操作。
with torch.no_grad():  # 在评估时，不需要计算梯度
    correct = 0  # 预测正确的样本数
    total = 0  # 总样本数
    for images, labels in test_loader:  #images类型是torch.Tensor，labels类型是torch.LongTensor
        outputs = model(images)        # 前向传播，计算模型输出，output类型是torch.Tensor储存的是每个样本属于每个类别的概率
        # outputs.data的形状是(batch_size, num_classes)
        # torch.max返回每行的最大值和索引，这里我们只需要索引，即预测的类别
        # 参数说明：torch.max(数据, 维度)，返回最大值和索引，这里我们只需要索引，即预测的类别，维度1也就是检查data数据里面的每一行的最大值和索引(列的维度上)
        _, predicted = torch.max(outputs.data, 1)  #比较output中的每个样本属于每个类别的概率，返回最大值和索引，这里我们只需要索引，即预测的类别
        total += labels.size(0)  # 累加总样本数
        correct += (predicted == labels).sum().item()  # 累加预测正确的样本数
        #predicted == labels是一个布尔类型的张量，它的元素是True或False，表示预测是否正确
        #<bool>.sum()将布尔类型的张量转换为整数类型的张量，True被转换为1，False被转换为0，然后求和，得到预测正确的样本数

    accuracy = correct / total  # 计算准确率
    print(f"模型在测试集上的准确率: {accuracy * 100:.2f}%")

# -----------------------分割线-----------------------
# 总结使用到的函数和参数

# 1. nn.Linear(in_features, out_features, bias=True)
#    - in_features：输入特征数，即每个样本的特征向量长度
#    - out_features：输出特征数，即类别数
#    - bias：是否包含偏置项，默认True

# 2. nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean')
#    - weight：给每个类别的损失赋予不同的权重，可用于处理类别不平衡
#    - ignore_index：指定一个类别，在计算损失时会忽略该类别的样本
#    - reduction：指定损失计算方式，'none'表示不进行聚合，'mean'表示求平均，'sum'表示求和

# 3. optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
#    - params：待优化的参数，一般传入model.parameters()
#    - lr：学习率
#    - momentum：动量因子，用于加速优化，默认0
#    - dampening：动量抑制因子，默认0
#    - weight_decay：权重衰减系数，用于L2正则化，默认0
#    - nesterov：布尔值，是否使用Nesterov动量，默认False

print("示例结束")



# 为什么要调用父类构造函数？
# 在Python中，当你创建一个类，并且这个类继承自另一个类（即父类）时，你通常需要在子类的构造函数中调用父类的构造函数。这是为了确保父类中的初始化代码能够被执行，从而正确地设置子类对象的状态。在PyTorch中，nn.Module是一个基类，用于构建所有的神经网络模块。它提供了许多基本的功能，如参数管理、前向传播等。因此，当你创建一个自定义的神经网络模型（如SoftmaxRegression）并继承自nn.Module时，你需要在构造函数中调用super(SoftmaxRegression, self).__init__()来确保nn.Module的初始化代码被正确执行。
#
# SoftmaxRegression类干啥用的？
# SoftmaxRegression类是一个自定义的神经网络模型，用于执行softmax回归。softmax回归是一种多分类算法，它可以将输入数据映射到多个类别上的概率分布。在这个特定的实现中，SoftmaxRegression类包含一个线性层（通过nn.Linear定义），用于将输入特征转换为与类别数相等的输出特征。然后，这些输出特征可以被解释为属于每个类别的概率（尽管在这个实现中并没有显式地应用softmax函数，因为后续使用的损失函数nn.CrossEntropyLoss已经包含了softmax操作）。
#
# num_classes又是什么数据？
# num_classes是一个整数，表示分类任务中的类别总数。在这个上下文中，num_classes=10意味着模型被训练来识别10个不同的类别（例如，手写数字0-9）。这个参数被用来定义线性层的输出特征数，即每个类别对应一个输出特征。这样，模型的输出就可以被解释为属于每个类别的概率分布。