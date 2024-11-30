# -*- coding: utf-8 -*-
# 4.9. 环境和分布偏移
print("4.9. 环境和分布偏移")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 分割线
print("=" * 50)

# 生成源域和目标域的数据
# 在这里，我们将创建一个简单的二分类问题，其中源域和目标域的分布不同

# 设置随机种子以保证可重复性
torch.manual_seed(0)

# 源域数据生成
# 生成均值为(2,2)的正态分布数据，标签为0
n_samples = 500
mean_source = [2, 2]
cov_source = [[1, 0], [0, 1]]
X_source = torch.randn(n_samples, 2) @ torch.tensor(cov_source).float() + torch.tensor(mean_source).float()
y_source = torch.zeros(n_samples, dtype=torch.long)

# 目标域数据生成
# 生成均值为(0,0)的正态分布数据，标签为0
mean_target = [0, 0]
cov_target = [[1, 0], [0, 1]]
X_target = torch.randn(n_samples, 2) @ torch.tensor(cov_target).float() + torch.tensor(mean_target).float()
y_target = torch.zeros(n_samples, dtype=torch.long)

# 测试数据（目标域）
# 生成均值为(0,0)和(2,2)的数据，标签分别为0和1
n_test_samples = 200
X_test_class0 = torch.randn(n_test_samples // 2, 2) @ torch.tensor(cov_target).float() + torch.tensor(
    mean_target).float()
y_test_class0 = torch.zeros(n_test_samples // 2, dtype=torch.long)

X_test_class1 = torch.randn(n_test_samples // 2, 2) @ torch.tensor(cov_target).float() + torch.tensor(
    mean_source).float()
y_test_class1 = torch.ones(n_test_samples // 2, dtype=torch.long)

X_test = torch.cat([X_test_class0, X_test_class1], dim=0)
y_test = torch.cat([y_test_class0, y_test_class1], dim=0)

# 可视化源域和目标域的数据分布
plt.figure(figsize=(8, 6))
plt.scatter(X_source[:, 0].numpy(), X_source[:, 1].numpy(), label='Source Domain', alpha=0.5)
plt.scatter(X_target[:, 0].numpy(), X_target[:, 1].numpy(), label='Target Domain', alpha=0.5)
plt.title('Source and Target Domain Data Distribution')
plt.legend()
plt.show()

# 分割线
print("=" * 50)


# 定义简单的分类模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化分类器
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出类别数
        """
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        return self.fc(x)


# 实例化模型
input_dim = 2
hidden_dim = 10
output_dim = 2
model = SimpleClassifier(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 分割线
print("=" * 50)

# 在源域数据上训练模型，不进行协变量偏移纠正
print("在源域数据上训练模型（未纠正协变量偏移）")

num_epochs = 20
batch_size = 64

# 创建数据加载器
source_dataset = torch.utils.data.TensorDataset(X_source, y_source)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    for X_batch, y_batch in source_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# 在测试集上评估模型性能
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f"未纠正协变量偏移时的测试集准确率: {accuracy.item():.4f}")

# 分割线
print("=" * 50)

# 实现协变量偏移纠正

# 1. 构建二元分类器来区分源域和目标域数据

# 标记源域数据为0，目标域数据为1
X_combined = torch.cat([X_source, X_target], dim=0)
y_combined = torch.cat([torch.zeros(n_samples, dtype=torch.long), torch.ones(n_samples, dtype=torch.long)], dim=0)

# 定义区分源域和目标域的分类器
domain_classifier = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 2)
)

# 定义损失函数和优化器
domain_criterion = nn.CrossEntropyLoss()
domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=0.01)

# 创建数据加载器
domain_dataset = torch.utils.data.TensorDataset(X_combined, y_combined)
domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=batch_size, shuffle=True)

# 训练域分类器
print("训练域分类器以区分源域和目标域数据")
num_epochs_domain = 10
for epoch in range(num_epochs_domain):
    for X_batch, y_batch in domain_loader:
        domain_optimizer.zero_grad()
        outputs = domain_classifier(X_batch)
        loss = domain_criterion(outputs, y_batch)
        loss.backward()
        domain_optimizer.step()

# 分割线
print("=" * 50)

# 2. 计算每个源域样本的权重beta_i = p_T(x_i) / p_S(x_i)

# 对于每个源域样本，计算域分类器的输出
with torch.no_grad():
    outputs = domain_classifier(X_source)
    # 计算源域样本被判为目标域的概率，即p_T(x_i)
    p_target = torch.softmax(outputs, dim=1)[:, 1]
    # 避免除以零
    p_source = 1 - p_target + 1e-6
    # 计算权重beta_i
    beta = p_target / p_source

# 分割线
print("=" * 50)

# 3. 使用权重重新训练模型

# 定义新的模型、损失函数和优化器
model_corrected = SimpleClassifier(input_dim, hidden_dim, output_dim)
criterion_corrected = nn.CrossEntropyLoss(reduction='none')  # 设置reduction='none'以获得每个样本的损失
optimizer_corrected = optim.Adam(model_corrected.parameters(), lr=0.01)

# 训练模型
print("使用协变量偏移纠正训练模型")
for epoch in range(num_epochs):
    for X_batch, y_batch, beta_batch in torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_source, y_source, beta),
            batch_size=batch_size,
            shuffle=True):
        optimizer_corrected.zero_grad()
        outputs = model_corrected(X_batch)
        losses = criterion_corrected(outputs, y_batch)
        # 使用权重调整损失
        loss = (losses * beta_batch).mean()
        loss.backward()
        optimizer_corrected.step()

# 在测试集上评估纠正后的模型性能
with torch.no_grad():
    outputs = model_corrected(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f"纠正协变量偏移后的测试集准确率: {accuracy.item():.4f}")

# 分割线
print("=" * 50)

# 总结：
# 在这个代码示例中，我们首先创建了源域和目标域的数据，这两个域的输入分布不同，存在协变量偏移。
# 然后，我们在未纠正协变量偏移的情况下训练了一个模型，发现它在测试集上的性能不佳。
# 接着，我们实现了协变量偏移纠正，通过训练一个域分类器来估计每个源域样本的权重。
# 最后，我们使用这些权重重新训练模型，发现模型在测试集上的性能得到了提升。

# 函数和类：
# SimpleClassifier：一个简单的全连接神经网络，用于分类任务。
# __init__(self, input_dim, hidden_dim, output_dim)：初始化模型，定义网络结构。
# forward(self, x)：前向传播函数，定义输入如何经过网络得到输出。

# nn.CrossEntropyLoss：交叉熵损失函数。
# 参数：
#     weight (Tensor, optional)：一个手动指定每个类别的权重张量。
#     size_average (bool, optional)：已被reduction替代。
#     ignore_index (int, optional)：指定忽略计算loss的目标值。
#     reduce (bool, optional)：已被reduction替代。
#     reduction (str, optional)：指定应用于输出的降维方式：'none' | 'mean' | 'sum'。

# optim.Adam：Adam优化器。
# 参数：
#     params (iterable)：待优化参数的iterable或者定义了参数组的dict。
#     lr (float, optional)：学习率（默认：1e-3）。
#     betas (Tuple[float, float], optional)：用于计算梯度及梯度平方的运行平均系数（默认：(0.9, 0.999)）。
#     eps (float, optional)：为了提高数值稳定性而加到分母里的项（默认：1e-8）。
#     weight_decay (float, optional)：权重衰减（L2惩罚）（默认：0）。

# torch.utils.data.TensorDataset：将数据和标签包装成Dataset。
# torch.utils.data.DataLoader：DataLoader组合了数据集和采样器，并在数据集上提供单或多进程的迭代器。

# 通过以上函数和方法，我们实现了协变量偏移的检测和纠正。

'''
要了解 分布偏移 的不同类型，我们可以通过一些简单的可视化方法来解释这些概念。以下是三种常见的分布偏移类型：协变量偏移（Covariate Shift）、标签偏移（Label Shift） 和 概念偏移（Concept Shift）。我会通过图示和简单的例子来帮助你理解每种类型。
1. 协变量偏移（Covariate Shift）
定义：
协变量偏移发生在 输入特征的分布发生变化，但 条件标签分布 （即标签给定输入特征的分布）保持不变。换句话说，特征（X）的分布发生了变化，但给定特征的标签（Y）的分布不变。
可视化例子：
假设我们训练一个分类模型来区分猫和狗。训练集的图像是真实的猫和狗照片，而测试集的图像则是卡通风格的猫和狗图像。尽管图像的标签（猫或狗）依然保持不变，但图像的特征分布已经发生了变化。
训练集（真实猫狗照片）
测试集（卡通猫狗）
在训练集中，特征分布来自真实照片，这些照片在视觉上有许多细节，特征可能比较复杂（如毛发、脸部表情等）。
在测试集中，特征分布来自卡通图像，风格非常不同（例如，颜色简化、细节丢失），尽管标签仍然是“猫”和“狗”。
此时，协变量偏移 就是指输入特征的分布发生了变化，但标签分布本身没有变化。
如何处理：
使用适应新特征分布的技术，如 领域适应（Domain Adaptation）来处理这种偏移。

2. 标签偏移（Label Shift）
定义：
标签偏移发生在 标签的边缘分布发生变化，但是 给定输入的标签条件分布保持不变。也就是说，标签的发生概率发生了变化，但给定特征的标签关系保持不变。
可视化例子：
假设我们在一个医疗系统中训练模型来预测某种疾病（标签是“病”或“健康”）。我们首先用过去一年的数据进行训练，其中“健康”样本占比80%，而“病”样本占比20%。然后，我们希望模型能适应今年的数据，但今年可能更多人得病，导致“病”样本的比例上升，变成50%。
训练集标签分布：
健康：80%
病：20%
测试集标签分布：
健康：50%
病：50%
尽管特征（如病人的症状）在训练集和测试集之间没有变化，但标签（健康或病）的分布发生了变化。原来“健康”占主导地位，但现在“病”样本的比例增加了。
如何处理：
解决标签偏移的方法通常涉及调整样本的权重或重新标定标签的边缘分布。例如， 重标定方法（如使用重新加权的分类器）可以在标签分布发生变化时，帮助模型更好地适应新的标签分布。

3. 概念偏移（Concept Shift）
定义：
概念偏移发生在 标签的定义本身发生变化，即标签的含义随着时间或背景的变化而改变。这意味着即使输入数据没有变化，但标签的解释和分类标准可能发生了改变。
可视化例子：
假设我们有一个模型来判断一个人是否是“时髦”的，标签是“时髦”或“不时髦”。在不同的地区或不同的时间，这个定义可能会有所不同。
例如，在美国某些地方，“时髦”可能意味着穿着某种特定品牌的衣服，而在其他地方，“时髦”可能意味着跟随某种潮流或文化标准。
某地区的定义：
时髦 = 穿名牌衣服
不时髦 = 穿普通衣服
另一地区的定义：
时髦 = 跟随当地流行文化
不时髦 = 不符合当地潮流
随着地区或时间的变化，“时髦”的定义发生了变化，这就是概念偏移。
如何处理：
概念偏移通常需要我们在模型中考虑不同的文化、地区或时间因素，以便使模型能够适应新的定义和标准。为了处理这种偏移，我们可能需要收集更多的数据或重新定义标签。

总结与对比
类型	        特征分布变化	    标签分布变化	    适应方法
协变量偏移	是	            否	            领域适应、特征转换
标签偏移	    否	            是	            重新加权、重标定
概念偏移	    否           	是（标签定义变化）	更新标签定义、调整模型标准
'''