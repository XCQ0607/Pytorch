import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置字体为Times New Roman
rcParams['font.family'] = 'Microsoft YaHei'

print("4.9. 环境和分布偏移")

# -----------------------------

# 生成训练数据
# 定义函数生成二维高斯分布的数据
def generate_data(mean, cov, num_samples, label):
    """
    生成二维正态分布的数据

    参数：
    mean: 均值向量，形状为(2,)
    cov: 协方差矩阵，形状为(2,2)
    num_samples: 生成的数据样本数
    label: 数据标签，0或1

    返回：
    X: 生成的数据，形状为(num_samples, 2)
    y: 数据对应的标签，形状为(num_samples,)
    """
    X = np.random.multivariate_normal(mean, cov, num_samples)   #nean: 均值向量，cov: 协方差矩阵，num_samples: 生成的数据样本数，label: 数据标签，0或1
    #生成指定方差用np.
    #multivariate_normal方法生成服从指定均值和协方差的随机样本
    #协方差与方差：
    # 方差（Variance）
    # 定义：方差是衡量单个随机变量与其均值之间差异程度的指标。简单来说，它反映了数据分布的离散程度。
    # 协方差（Covariance）
    # 定义：协方差是衡量两个随机变量之间线性相关程度的指标。当两个随机变量的变化趋势一致时，它们的协方差为正；当变化趋势相反时，协方差为负；如果两个随机变量相互独立，则它们的协方差为0。
    # 协方差的符号表示两个随机变量的相关方向：正协方差表示正相关，负协方差表示负相关，协方差为0表示不相关。
    # 协方差的大小表示相关程度的强弱，但协方差的值受到变量量纲的影响，因此在实际应用中，通常使用相关系数（如皮尔逊相关系数）来标准化协方差，以便更直观地比较不同变量之间的相关性。
    y = np.full(num_samples, label)  #生成指定长度的数组，每个元素都是指定值
    return X, y

# 生成训练集数据
mean0 = [0, 0]  # 类别0的均值
mean1 = [2, 2]  # 类别1的均值
cov = [[1, 0], [0, 1]]  # 协方差矩阵
#协方差矩阵的一个重要性质：对称性。在协方差矩阵中，cov[i][j] 必须等于 cov[j][i]

num_samples = 1000  # 每个类别的样本数

X0_train, y0_train = generate_data(mean0, cov, num_samples, 0)  #生成训练集数据，类别0
X1_train, y1_train = generate_data(mean1, cov, num_samples, 1)  #生成训练集数据，类别1
#这里的mean1的shape为(2,)，cov的shape为(2,2)，num_samples的shape为(1,)，label的shape为(1,)
# mean 参数是一个长度为2的向量，表示二维正态分布的均值向量。在这个例子中，mean 是 [0, 0] 或 [2, 2]，意味着每个类别在二维空间中的中心点坐标。
# cov 参数是一个2x2的矩阵，表示二维正态分布的协方差矩阵。在这个例子中，cov 是 [[1, 0], [0, 1]]，表示两个维度（或变量）之间的协方差为0（即它们是独立的），且每个维度的方差为1。

# 如何理解协方差矩阵？
# 协方差矩阵通常用于描述多个随机变量之间的协方差。在二维情况下，协方差矩阵是一个 2×2 的矩阵，表示两个随机变量之间的关系：
# 对角线上的元素是每个变量的 方差（即该变量与自己之间的协方差）。
# 非对角线上的元素是两个变量之间的 协方差（即两个变量之间的关系）。
# 具体来说：
# (1, 1) 位置的元素是第一个随机变量的方差。
# (2, 2) 位置的元素是第二个随机变量的方差。
# (1, 2) 和 (2, 1) 位置的元素是两个变量之间的协方差。

# 协方差矩阵是：
# Σ=
# [1    0.8   0.5]
# [0.8  1     0  ]
# [0.5  0     1  ]
#
# 对角线元素：这些元素表示各个变量的方差：
# cov(1,1)=1  # 第一个变量的方差
# cov(2,2)=1  # 第二个变量的方差
# cov(3,3)=1  # 第二个变量的方差
#
# 非对角线元素：这些元素表示各个变量之间的协方差：
# cov(1,2)=cov(2,1)=0.8  # 第一个变量和第二个变量之间的协方差
# cov(1,3)=cov(3,1)=0.5  # 第一个变量和第三个变量之间的协方差
# cov(2,3)=cov(3,2)=0    # 第二个变量和第三个变量之间的协方差




X_train = np.vstack((X0_train, X1_train))   #合并训练集数据
y_train = np.hstack((y0_train, y1_train))   #合并训练集标签
#vstack和hstack的区别：
# vstack: 沿着垂直方向（行）堆叠数组。
# hstack: 沿着水平方向（列）堆叠数组。
# 例如，对于二维数组，vstack将按行方向堆叠，而hstack将按列方向堆叠。
# 对于一维数组，vstack将按列方向堆叠，而hstack将按行方向堆叠。

# 绘制训练数据的散点图
plt.figure(figsize=(6,6))
plt.scatter(X0_train[:,0], X0_train[:,1], c='blue', label='Class 0')    #[:,0]表示取所有行的第0列，[:,1]表示取所有行的第1列
#scatter方法用于绘制散点图，参数为x轴数据，y轴数据，c='blue'表示颜色为蓝色，label='Class 0'表示标签为类别0
#c='blue'表示颜色为蓝色，label='Class 0'表示标签为类别0
#
plt.scatter(X1_train[:,0], X1_train[:,1], c='red', label='Class 1')
plt.legend()
plt.title('训练集数据分布')
plt.show()

# -----------------------------
# 生成测试集数据，模拟协变量偏移
# 测试集的类别均值发生变化
mean0_test = [1, 1]  # 类别0的均值发生偏移
mean1_test = [3, 3]  # 类别1的均值发生偏移

X0_test, y0_test = generate_data(mean0_test, cov, num_samples, 0)
X1_test, y1_test = generate_data(mean1_test, cov, num_samples, 1)

X_test = np.vstack((X0_test, X1_test))
y_test = np.hstack((y0_test, y1_test))

# 绘制测试数据的散点图
plt.figure(figsize=(6,6))
plt.scatter(X0_test[:,0], X0_test[:,1], c='blue', label='Class 0')
plt.scatter(X1_test[:,0], X1_test[:,1], c='red', label='Class 1')
plt.legend()
plt.title('测试集数据分布（协变量偏移）')
plt.show()

# -----------------------------
# 将数据转换为PyTorch的Tensor
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()

# 定义一个简单的两层神经网络分类器
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        初始化神经网络

        参数：
        input_size: 输入特征的维度
        hidden_size: 隐藏层的神经元数量
        num_classes: 分类的类别数
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 全连接层1
        self.relu = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 全连接层2

    def forward(self, x):
        """
        前向传播函数

        参数：
        x: 输入数据，形状为(batch_size, input_size)

        返回：
        输出类别的logits，形状为(batch_size, num_classes)
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
#在forward中规定顺序。而Sequential中顺序是固定
# 如：
# nn.Sequential(
#     nn.Linear(10, 20),    #10是输入特征维度，20是隐藏层神经元数量
#     nn.ReLU(),            #ReLU激活函数
#     nn.Linear(20, 10)     #20是隐藏层神经元数量，10是输出类别数
# )
# 而Sequential中顺序是固定的

# 定义模型、损失函数和优化器
input_size = 2  # 输入特征维度
hidden_size = 10  # 隐藏层神经元数量
num_classes = 2  # 分类类别数

model = SimpleNN(input_size, hidden_size, num_classes)  # 实例化模型
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器，学习率为0.01

# -----------------------------
# 训练模型
num_epochs = 100
batch_size = 64

# 将训练数据封装成Dataset和DataLoader
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   #shuffle=True表示打乱顺序

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader): #enumerate用于遍历可迭代对象，i表示索引，(inputs, labels)表示数据和标签
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 20 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))   #.format用于格式化字符串，{:.4f}表示保留4位小数

# -----------------------------
# 评估模型在测试集上的性能
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    print('模型在测试集上的准确率: {:.2f}%'.format(100 * correct / total))

# -----------------------------
print('-' * 50)
print('实施协变量偏移检测器')

# 将训练集和测试集的数据合并，创建一个新的二分类任务
# 标签为0表示来自训练集，标签为1表示来自测试集

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((np.zeros(len(X_train)), np.ones(len(X_test))))

# 将数据转换为Tensor
X_combined_tensor = torch.from_numpy(X_combined).float()
y_combined_tensor = torch.from_numpy(y_combined).long()

# 构建用于检测协变量偏移的分类器
shift_detector = SimpleNN(input_size, hidden_size, 2)  # 输出类别为2

criterion_shift = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer_shift = optim.Adam(shift_detector.parameters(), lr=0.01)  # 优化器

# 将数据封装成Dataset和DataLoader
combined_dataset = TensorDataset(X_combined_tensor, y_combined_tensor)
combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

# 训练检测器
num_epochs_shift = 20
for epoch in range(num_epochs_shift):
    for i, (inputs, labels) in enumerate(combined_loader):
        outputs = shift_detector(inputs)
        loss = criterion_shift(outputs, labels)

        optimizer_shift.zero_grad()
        loss.backward()
        optimizer_shift.step()

    if (epoch+1) % 5 == 0:
        print('Shift Detector Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs_shift, loss.item()))

# -----------------------------
# 评估协变量偏移检测器的性能
with torch.no_grad():
    outputs = shift_detector(X_combined_tensor)
    _, predicted = torch.max(outputs.data, 1)
    total = y_combined_tensor.size(0)
    correct = (predicted == y_combined_tensor).sum().item()
    print('协变量偏移检测器的准确率: {:.2f}%'.format(100 * correct / total))

# 如果检测器的准确率远高于50%，则说明存在协变量偏移

# -----------------------------
print('-' * 50)
print('实施协变量偏移纠正')

# 获取训练样本在协变量偏移检测器下的logits
with torch.no_grad():
    h_x = shift_detector(X_train_tensor)  # 输出为(logits)
    logits = h_x  # logits，未经过softmax

    # 计算重要性权重beta_i = p(z=1|x_i)/p(z=0|x_i)
    prob = nn.functional.softmax(logits, dim=1) # 输出为概率
    p_z1_given_x = prob[:,1]    # p(z=1|x_i)
    p_z0_given_x = prob[:,0]    # p(z=0|x_i)

    # 计算密度比p(x)/q(x) = p(z=1|x)/p(z=0|x)
    density_ratio = p_z1_given_x / p_z0_given_x

    # 为了避免极端的权重，我们可以设置一个上限c
    c = 10  # 常数c
    beta = torch.min(density_ratio, torch.tensor(c))

    # 将beta转换为numpy数组
    beta = beta.numpy()

# -----------------------------
# 重新初始化模型、损失函数和优化器
model_corrected = SimpleNN(input_size, hidden_size, num_classes)
criterion_corrected = nn.CrossEntropyLoss(reduction='none')  # 设置reduction='none'，以获得每个样本的损失
optimizer_corrected = optim.Adam(model_corrected.parameters(), lr=0.01)

# 使用加权经验风险最小化进行训练
num_epochs_corrected = 100
batch_size = 64

train_dataset_corrected = TensorDataset(X_train_tensor, y_train_tensor, torch.from_numpy(beta).float())
train_loader_corrected = DataLoader(train_dataset_corrected, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs_corrected):
    for i, (inputs, labels, weights) in enumerate(train_loader_corrected):
        outputs = model_corrected(inputs)
        losses = criterion_corrected(outputs, labels)  # 每个样本的损失
        loss = torch.mean(losses * weights)  # 加权平均损失

        optimizer_corrected.zero_grad()
        loss.backward()
        optimizer_corrected.step()

    if (epoch+1) % 20 == 0:
        print('Corrected Model Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs_corrected, loss.item()))

# -----------------------------
# 评估修正后的模型在测试集上的性能
with torch.no_grad():
    outputs = model_corrected(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    total = y_test_tensor.size(0)    # 总样本数
    correct = (predicted == y_test_tensor).sum().item()    # 正确预测的样本数
    print('修正后的模型在测试集上的准确率: {:.2f}%'.format(100 * correct / total))

# -----------------------------
# 总结：
# 在本代码示例中，我们演示了协变量偏移的检测和纠正方法。
# 我们使用了以下函数和类：
# - generate_data(mean, cov, num_samples, label): 用于生成二维正态分布的数据。
# - SimpleNN(nn.Module): 定义了一个简单的两层神经网络，用于分类任务。
# - torch.utils.data.TensorDataset: 用于将数据封装成Dataset对象。
# - torch.utils.data.DataLoader: 用于创建可迭代的数据加载器。
# - nn.CrossEntropyLoss(reduction='none'): 损失函数，设置reduction='none'以获得每个样本的损失值。
# - optimizer.zero_grad(), loss.backward(), optimizer.step(): 用于训练神经网络的标准步骤。

# 我们首先训练了一个分类器，观察到由于协变量偏移，模型在测试集上的性能下降。
# 然后，我们通过构建一个协变量偏移检测器，检测到了训练集和测试集分布的差异。
# 接着，我们使用检测器的输出计算了重要性权重，对训练过程进行了纠正。
# 最终，修正后的模型在测试集上的性能得到了提升。

# 练习4：
# 除了分布偏移之外，还有其他因素会影响经验风险接近真实风险的程度：
# - 训练数据的有限性：样本数量不足会导致经验风险估计不准确。
# - 过拟合：模型在训练数据上表现良好，但在测试数据上表现不佳。
# - 模型假设与真实数据不匹配：模型的假设（如线性、非线性）可能不符合真实数据的分布。
# - 噪声和异常值：数据中的噪声和异常值会影响经验风险的估计。

# 总之，为了使经验风险更好地近似真实风险，我们需要确保训练数据的充足性、模型的适用性，以及处理好数据中的噪声和异常值。

