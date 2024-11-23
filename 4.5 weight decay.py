# -*- coding: utf-8 -*-
"""
4.5. 权重衰减（Weight Decay）示例代码

此代码示例演示了如何在PyTorch中实现权重衰减，以解决过拟合问题。
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 配置中文字体
rcParams['font.family'] = 'Microsoft YaHei'

print("4.5. 权重衰减")

# 分割线
print("=" * 50)

# 设置随机数种子，保证可重复性
torch.manual_seed(0)

# 生成高维线性回归数据集
def generate_synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的数据

    参数：
    w (Tensor): 权重向量
    b (float): 偏置
    num_examples (int): 样本数量

    返回：
    features (Tensor): 特征矩阵
    labels (Tensor): 标签向量
    """
    features = torch.randn(num_examples, w.shape[0])
    labels = torch.matmul(features, w) + b
    labels += torch.normal(0, 0.01, size=labels.shape)  # 添加噪声
    return features, labels.reshape(-1, 1)

# 定义真实参数
num_inputs = 200
true_w = torch.ones(num_inputs, 1) * 0.01
true_b = 0.05

# 生成训练和测试数据集
n_train, n_test = 20, 100  # 训练样本数和测试样本数
features_train, labels_train = generate_synthetic_data(true_w, true_b, n_train)
features_test, labels_test = generate_synthetic_data(true_w, true_b, n_test)

# 数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """构造PyTorch数据迭代器

    参数：
    data_arrays (tuple): 包含特征和标签的元组
    batch_size (int): 批量大小
    is_train (bool): 是否打乱数据

    返回：
    data_iter (DataLoader): 数据迭代器
    """
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 5
train_iter = load_array((features_train, labels_train), batch_size)
test_iter = load_array((features_test, labels_test), batch_size, is_train=False)

# 定义模型
def linreg(X, w, b):
    """线性回归模型

    参数：
    X (Tensor): 输入特征
    w (Tensor): 权重
    b (Tensor): 偏置

    返回：
    Tensor: 模型输出
    """
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """平方损失函数

    参数：
    y_hat (Tensor): 预测值
    y (Tensor): 真实值

    返回：
    Tensor: 损失
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法（小批量随机梯度下降）
def sgd(params, lr, batch_size):    #sgd是小批量随机梯度下降的缩写
    """小批量随机梯度下降优化算法

    参数：
    params (list): 模型参数
    lr (float): 学习率
    batch_size (int): 批量大小
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size   #更新参数
            param.grad.zero_()  #清空梯度
# 更新参数时为什么要清空梯度？
#     在神经网络训练中，梯度是累积的。每次反向传播时，梯度值会累加到之前计算的梯度上。如果不清空梯度，下一次的梯度更新就会基于之前累积的梯度，这会导致更新方向不准确，进而影响模型的训练效果。因此，在每次参数更新后，我们需要清空梯度，以确保下一次的梯度计算是从零开始的，从而得到正确的更新方向。
# lr * param.grad / batch_size有什么意义？
#     这个表达式是随机梯度下降（SGD）算法中参数更新的核心部分。
#     lr（学习率）是一个超参数，用于控制参数更新的步长。较大的学习率可能导致训练不稳定，而较小的学习率可能导致训练速度缓慢。
#     param.grad 是当前参数的梯度值，它指示了为了最小化损失函数，参数应该调整的方向和幅度。
#     batch_size 是小批量的大小。在随机梯度下降中，我们不是使用整个数据集来计算梯度，而是使用一个小批量。由于我们只使用了一部分数据来计算梯度，因此需要对梯度进行归一化，以确保更新步长不会因为批量大小的变化而变化。除以 batch_size 就是一种常用的归一化方法。
# 综上所述，lr * param.grad / batch_size 这个表达式计算了参数应该更新的方向和幅度。通过减去这个值，我们实现了参数的更新，从而逐步优化模型。

# param.grad / batch_size 确实是一种归一化的方法，用于调整梯度的大小，使其不依赖于批量（batch）的大小。这样做有助于确保无论批量大小如何，参数更新的步长都保持在一个相对稳定的范围内。
# 在神经网络训练中，我们通常使用小批量梯度下降（Mini-batch Gradient Descent）或其变种来优化模型的参数。这意味着在每次迭代中，我们只使用数据集的一个小子集（即一个小批量）来计算损失函数和梯度。由于不同批量的数据可能具有不同的特性，因此它们产生的梯度也可能有所不同。
# 如果我们直接使用这些梯度来更新参数，那么更新步长将会受到批量大小的影响。具体来说，较大的批量可能会产生较大的梯度，从而导致参数更新步长过大；而较小的批量则可能产生较小的梯度，导致参数更新步长过小。这都不是我们想要的结果，因为我们希望参数更新的步长能够相对稳定，以便更好地控制训练过程。
# 为了解决这个问题，我们可以将梯度除以批量大小来进行归一化。这样做可以消除批量大小对梯度大小的影响，使得参数更新的步长更加稳定和一致。因此，param.grad / batch_size 是一种常用的归一化方法，用于确保参数更新的稳定性和有效性。

# 定义L2范数惩罚项
def l2_penalty(w):
    """计算L2范数惩罚项

    参数：
    w (Tensor): 权重向量

    返回：
    Tensor: L2惩罚项
    """
    return torch.sum(w.pow(2)) / 2  #L2范数惩罚项

# 训练函数
def train(lambd):
    """训练模型并绘制训练和测试损失曲线

    参数：
    lambd (float): 权重衰减（L2正则化）系数
    """
    # 初始化参数
    w = torch.normal(0, 0.01, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 定义学习率和训练周期
    lr = 0.003
    num_epochs = 100

    # 用于绘图的数据
    train_loss = []
    test_loss = []

    # 训练过程
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 计算模型输出
            y_hat = linreg(X, w, b)
            # 计算损失（包含L2惩罚项）
            loss = squared_loss(y_hat, y) + lambd * l2_penalty(w)   #损失函数中加入了L2惩罚项
            # 反向传播
            loss.sum().backward()
            # 更新参数
            sgd([w, b], lr, batch_size)  #使用sgd更新参数
        # 记录损失
        with torch.no_grad():
            train_loss.append(squared_loss(linreg(features_train, w, b), labels_train).mean().item())
            test_loss.append(squared_loss(linreg(features_test, w, b), labels_test).mean().item())
        # 每10个周期打印一次结果
        if (epoch + 1) % 10 == 0:
            print(f'周期 {epoch + 1}, 训练损失 {train_loss[-1]:f}, 测试损失 {test_loss[-1]:f}')

    # 在深度学习中，损失函数（loss function）是用来衡量模型预测值与真实值之间差距的函数，而优化算法（如梯度下降）则是通过调整模型参数来最小化这个损失函数。当你向损失函数中加入L2惩罚项（也称为权重衰减）时，你实际上是在改变损失函数的形状，这会影响梯度计算的结果。
    #
    # 具体来说，在你的例子中：
    # loss = squared_loss(y_hat, y) + lambd * l2_penalty(w)  # 损失函数中加入了L2惩罚项
    # # 反向传播
    # loss.sum().backward()
    #
    # squared_loss(y_hat, y)表示预测值y_hat与真实值y之间的平方损失，而lambd * l2_penalty(w)是L2惩罚项，其中lambd是控制惩罚项强度的超参数，l2_penalty(w)通常是模型权重w的平方和。
    # 当你调用loss.sum().backward()时，PyTorch会自动计算损失函数相对于模型参数的梯度。这里的梯度计算会考虑整个损失函数，包括L2惩罚项。因此，加入L2惩罚项确实会影响梯度计算的对象与内容。
    #
    # 对象：梯度计算的对象是模型参数，即那些需要通过优化算法进行调整的变量。在这个例子中，w
    # 是模型参数之一，而L2惩罚项是直接作用于w的。因此，当计算梯度时，会同时考虑平方损失项和L2惩罚项对w的影响。
    # 内容：梯度计算的内容是损失函数相对于每个模型参数的偏导数。这些偏导数指示了为了最小化损失函数，应该如何调整每个参数。由于L2惩罚项增加了权重值的成本，因此它会倾向于推动权重向零靠近（但不一定完全为零），这有助于防止模型过拟合。这种影响会体现在计算出的梯度中，使得在更新参数时不仅考虑减少预测误差，还要考虑减少权重的大小。

    # 绘制损失曲线

    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, num_epochs + 1), train_loss, label='训练损失')
    plt.semilogy(range(1, num_epochs + 1), test_loss, label='测试损失')
    plt.xlabel('周期')
    plt.ylabel('损失')
    #图表标题为  使用权重衰减（lambda=lambd）的训练和测试损失曲线
    plt.title(f'使用权重衰减（lambda={lambd}）的训练和测试损失曲线')
    plt.legend()
    plt.show()

    # 输出权重的L2范数
    print('w的L2范数：', torch.norm(w).item())

# 分割线
print("-" * 50)
print("不使用权重衰减（lambda=0）")
train(lambd=0)

# 分割线
print("-" * 50)
print("使用权重衰减（lambda=3）")
train(lambd=3)

# 分割线
print("=" * 50)
print("使用PyTorch内置函数实现权重衰减")

def train_concise(wd):
    """使用PyTorch内置的优化器参数实现权重衰减

    参数：
    wd (float): 权重衰减（L2正则化）系数
    """
    # 定义模型
    net = nn.Linear(num_inputs, 1)
    # 初始化参数
    nn.init.normal_(net.weight, mean=0, std=0.01)   #初始化net的weight参数
    nn.init.zeros_(net.bias)    #初始化net的bias参数

    # 定义损失函数
    loss = nn.MSELoss()

    # 定义优化器，传入权重衰减参数weight_decay
    optimizer = torch.optim.SGD([   #SDG是随机梯度下降的缩写
        {'params': net.weight, 'weight_decay': wd},
        {'params': net.bias}
    ], lr=0.003)

    # 在PyTorch的torch.optim.SGD优化器中，weight_decay参数用于指定权重衰减（也称为L2正则化）的系数。权重衰减是一种正则化技术，用于防止模型在训练过程中过拟合。通过在优化过程中向损失函数添加一个与权重参数平方成比例的项，权重衰减鼓励模型使用更小的权重值，这通常有助于提升模型的泛化能力。
    # 对于net.weight参数组，你明确指定了一个weight_decay值wd。这意味着在每次更新net.weight时，优化器都会考虑权重衰减。具体来说，每次梯度下降更新时，net.weight的梯度会被加上一个与net.weight本身成比例的项（比例系数为wd），然后再进行更新。这样，net.weight中的较大值会受到更大的惩罚，从而鼓励模型使用更小的权重。
    #
    # wd参数的值决定了权重衰减的强度。较大的wd值会导致更强的正则化效果，即权重在更新过程中会更快地趋向于零。相反，较小的wd值则会导致较弱的正则化效果。选择合适的wd值通常需要一些实验和调整，以便在防止过拟合和保持模型性能之间找到平衡。
    # 需要注意的是，在你的代码片段中，net.bias参数组没有指定weight_decay，因此偏置项在更新时不会受到权重衰减的影响。这是因为通常认为偏置项对于模型的复杂度贡献较小，所以不需要对其进行正则化。

    # {'params': net.weight, 'weight_decay': wd}：
    # 'params': net.weight指定了这一组参数是网络的权重（net.weight）。这里的net是一个PyTorch神经网络模型，而net.weight通常指的是模型中所有权重参数的集合（但具体取决于net的实现，一般需要通过net.parameters()来获取所有参数）。如果net.weight是特意定义的只包含权重的参数组，那这里的用法就是正确的。否则，通常的做法是使用net.parameters()或单独筛选出权重参数。
    # 'weight_decay': wd指定了权重衰减（weightdecay）的系数，用于正则化。权重衰减是一种防止模型过拟合的技术，它通过在优化过程中对权重参数施加惩罚来实现。wd是权重衰减系数的具体值，需要在代码的其他部分定义。
    #
    # {'params': net.bias}：
    # 'params': net.bias指定了这一组参数是网络的偏置项（net.bias）。和net.weight类似，net.bias通常指的是模型中所有偏置参数的集合（具体也取决于net的实现）。如果net是标准的PyTorch神经网络层或模型，通常不会直接有net.bias这样的属性，除非被特意定义。一般情况下，偏置项会和权重一起通过net.parameters()获取。这两组字典作为列表的元素传递给torch.optim.SGD，意味着优化器将分别对权重（可能带有权重衰减）和偏置进行不同的优化处理。

    # 你可以通过遍历net.parameters()来查看这些参数：
    # for param in net.parameters():
    #     print(param.shape)
    # #这段代码会输出每个参数的形状，对于你的线性层，
    # 输出应该是
    # torch.Size([1, num_inputs])  # 权重的形状
    # torch.Size([1])  # 偏置的形状
    #
    # 如果你想单独获取权重或偏置，你可以这样做：
    # # 获取权重
    # weight = list(net.parameters())[0]
    # # 获取偏置
    # bias = list(net.parameters())[1]

    num_epochs = 100

    # 用于绘图的数据
    train_loss = []
    test_loss = []

    # 训练过程
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()    #使用optimizer.step()更新模型参数,包含weight decay
        # 记录损失
        with torch.no_grad():
            train_loss.append(loss(net(features_train), labels_train).item())
            test_loss.append(loss(net(features_test), labels_test).item())
        # 每10个周期打印一次结果
        if (epoch + 1) % 10 == 0:
            print(f'周期 {epoch + 1}, 训练损失 {train_loss[-1]:f}, 测试损失 {test_loss[-1]:f}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, num_epochs + 1), train_loss, label='训练损失')    #semilogy是绘制对数坐标图的函数
    plt.semilogy(range(1, num_epochs + 1), test_loss, label='测试损失')    #semilogy是绘制对数坐标图的函数
    #对数坐标图
        # 对数坐标图是一种在坐标轴上使用对数刻度的图表。在这种图表中，坐标轴上的刻度不是等距的，而是按照对数比例分布的。这种图表特别适用于展示数量级差异很大的数据，因为对数刻度可以更好地展示这种差异。
        # 在你提供的代码片段中，plt.semilogy()函数是Matplotlib库中的一个函数，用于绘制y轴为对数刻度的图表。这意味着，在生成的图表中，y轴上的刻度将按照对数比例分布，而x轴则保持为普通的线性刻度。
        # plt.semilogy(range(1, num_epochs + 1), train_loss, label='训练损失')        这行代码的作用是绘制一个图表，其中x轴表示训练轮次（从1到num_epochs），y轴表示训练损失，并且y轴使用对数刻度。
        # plt.semilogy(range(1, num_epochs + 1), test_loss, label='测试损失')     这行代码的作用是绘制另一个图表，展示测试损失随训练轮次的变化，并且y轴也使用对数刻度。
    plt.xlabel('周期')    #xlabel是设置x轴标签
    plt.ylabel('损失')    #ylabel是设置y轴标签
    #图表标题为  使用权重衰减（weight_decay=wd）的训练和测试损失曲线
    plt.title(f'使用权重衰减（weight_decay={wd}）的训练和测试损失曲线')
    plt.legend()    #显示图例
    plt.show()

    # 输出权重的L2范数
    print('w的L2范数：', net.weight.norm().item())

    #.norm()是计算矩阵的Frobenius范数的函数，.item()是将张量转换为标量的函数，也就是L2范数。因为默认params=2，所以是L2范数。
    #L1范数计算: torch.norm(net.weight, p=1).item()

# 分割线
print("-" * 50)
print("不使用权重衰减（weight_decay=0）")
train_concise(0)

# 分割线
print("-" * 50)
print("使用权重衰减（weight_decay=3）")
train_concise(3)

# 分割线
print("=" * 50)
print("练习题解答")

# 1. 使用不同的lambda值进行实验，绘制训练和测试精度关于lambda的函数。

lambda_values = [0, 0.1, 1, 3, 5, 10]
train_losses = []    #训练损失
test_losses = []     #测试损失
norms = []           #权重范数

for lambd in lambda_values:
    # 初始化参数
    w = torch.normal(0, 0.01, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 定义学习率和训练周期
    lr = 0.003
    num_epochs = 100

    # 训练过程
    for epoch in range(num_epochs):
        for X, y in train_iter:
            y_hat = linreg(X, w, b)
            loss = squared_loss(y_hat, y) + lambd * l2_penalty(w)
            loss.sum().backward()
            sgd([w, b], lr, batch_size)

    # 记录损失和权重范数
    with torch.no_grad():
        train_loss = squared_loss(linreg(features_train, w, b), labels_train).mean().item()
        test_loss = squared_loss(linreg(features_test, w, b), labels_test).mean().item()
        w_norm = torch.norm(w).item()    #计算权重范数
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        norms.append(w_norm)    #记录权重范数

# 绘制损失关于lambda的曲线
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, train_losses, label='训练损失')
plt.plot(lambda_values, test_losses, label='测试损失')
plt.xlabel('lambda')
plt.ylabel('损失')
plt.legend()
plt.show()

print("观察到当lambda增大时，训练损失逐渐增大，而测试损失先减小后增大，存在一个最佳的lambda使测试损失最小。")

# 2. 使用验证集来找到最佳值lambda。它真的是最优值吗？这有关系吗？

# 通常，我们可以将训练数据再划分出一部分作为验证集，用于选择最佳的lambda值。
# 由于样本量较小，这里仅作示意。

# 3. 如果我们使用sum_i |w_i|作为惩罚（L1正则化），更新方程会是什么样子？

# 对于L1正则化，损失函数中增加了lambda * sum_i |w_i|，其梯度为lambda * sign(w_i)
# 因此，更新方程为：w_i = w_i - lr * (dL/dw_i + lambda * sign(w_i))

# 4. 我们知道||w||^2 = w^T w。能找到类似的矩阵方程吗？

# 对于矩阵W，其Frobenius范数||W||_F^2 = trace(W^T W)

# 5. 除了权重衰减、增加训练数据、使用适当复杂度的模型之外，还能想出其他什么方法来处理过拟合？

# 答：可以使用 dropout、数据增强、早停（early stopping）、集成方法等来处理过拟合。

# 6. 在贝叶斯统计中，我们使用先验和似然的乘积，通过公式P(w|x) ∝ P(x|w)P(w)得到后验。如何得到带正则化的P(w)？

# 答：带正则化的P(w)可以被视为对参数w的先验分布。例如，L2正则化相当于对w施加零均值的高斯先验。

# 分割线
print("=" * 50)

# 总结
"""
在本代码示例中，我们使用了以下函数和方法：

1. generate_synthetic_data(w, b, num_examples)
   - 生成模拟数据集的函数，参数包括权重w、偏置b和样本数量num_examples。

2. load_array(data_arrays, batch_size, is_train=True)
   - 构造数据迭代器的函数，参数包括数据元组data_arrays、批量大小batch_size和是否打乱数据is_train。

3. linreg(X, w, b)
   - 定义线性回归模型的函数，参数包括输入特征X、权重w和偏置b。

4. squared_loss(y_hat, y)
   - 定义平方损失函数，参数包括预测值y_hat和真实值y。

5. sgd(params, lr, batch_size)
   - 实现小批量随机梯度下降优化算法的函数，参数包括模型参数params、学习率lr和批量大小batch_size。

6. l2_penalty(w)
   - 计算L2范数惩罚项的函数，参数是权重向量w。

7. train(lambd)
   - 训练模型的函数，参数是权重衰减系数lambd。

8. train_concise(wd)
   - 使用PyTorch内置优化器实现权重衰减的函数，参数是权重衰减系数wd。

这些函数展示了如何在PyTorch中从零开始和使用内置函数实现权重衰减，以解决过拟合问题。
"""


# https://www.bilibili.com/video/BV1gf4y1c7Gg