# 4.4. 模型选择、欠拟合和过拟合
print("4.4. 模型选择、欠拟合和过拟合")
print("4.4.4. 多项式回归")
print('-' * 50)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
#微软雅黑
rcParams['font.family'] = 'Microsoft YaHei'


# 为了可重复性，设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# 多项式的最大阶数
max_degree = 20

# 训练和测试样本数量
n_train, n_test = 100, 100

# 用于生成数据的真实权重（只有前4个是非零的）
true_w = np.zeros(max_degree)   # 初始化真实权重，所有权重初始化为0，shape为 (max_degree,)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) # 设置前4个权重为 [5, 1.2, -3.4, 5.6]

# 生成特征 x ~ N(0, 1)
features = np.random.normal(size=(n_train + n_test, 1)) # 生成特征,np.random.normal 生成正态分布的随机数，均值为0，标准差为1
#如果想要生成均值为1，标准差为2的正态分布随机数，可以使用 np.random.normal(loc=1, scale=2, size=(n_train + n_test, 1)) loc全称location，scale全称scale,loc是均值，scale是标准差
np.random.shuffle(features) #打乱数据顺序

# 生成多项式特征，最高到 max_degree 次方
poly_features = np.power(features, np.arange(max_degree))   #np.power(features, np.arange(max_degree)) 计算 features 的幂次方，np.arange(max_degree) 生成从 0 到 max_degree-1 的数组,也就是计算 features 的 0 次方到 max_degree-1 次方
#poly_features的shape: (n_train+n_test, max_degree)
# 使用阶乘进行标准化
for i in range(max_degree):    # range(n) 生成从 0 到 n-1 的数组
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n) = (n-1)!
#［i:j, k:l］表示取第i到j行，第k到l列的元素
# 使用 true_w 生成标签，并添加高斯噪声
#features np.power前:    [[x1], [x2], [x3],...,[x(n_train + n_test)]]
#np.power后：[[x1^0,x1^1,x1^2,x1^3……x^(max_degree-1)], [x2^0,x2^1,x2^2,x2^3……x^(max_degree-1)], [x3^0,x3^1,x3^2,x3^3……x^(max_degree-1)],...,[x(n_train + n_test)^0,x(n_train + n_test)^1,x(n_train + n_test)^2,x(n_train + n_test)^3……x^(max_degree-1)]]
#math.gamma后:   [[x1^0/gamma(1),x1^1/gamma(2),x1^2/gamma(3),x1^3/gamma(4)……x^(max_degree-1)/gamma(max_degree)], [x2^0/gamma(1),x2^1/gamma(2),x2^2/gamma(3),x2^3/gamma(4)……x^(max_degree-1)/gamma(max_degree)], [x3^0/gamma(1),x3^1/gamma(2),x3^2/gamma(3),x3^3/gamma(4)……x^(max_degree-1)/gamma(max_degree)],...,[x(n_train + n_test)^0/gamma(1),x(n_train + n_test)^1/gamma(2),x(n_train + n_test)^2/gamma(3),x(n_train + n_test)^3/gamma(4)……x^(max_degree-1)/gamma(max_degree)]]

#shape  poly_features: (n_train + n_test, max_degree)   true_w: (max_degree,)   labels: (n_train + n_test,)
labels = np.dot(poly_features, true_w)  #这里执行点积操作原因：poly_features 的每一行是一个样本，每一列是一个特征，true_w 是一个权重向量，poly_features 和 true_w 的点积就是 poly_features 的每一行的线性组合，也就是 poly_features 的每一行的线性回归
#np.dot(poly_features, true_w) 计算 poly_features 和 true_w 的点积，也就是计算 poly_features 的每一行和 true_w 的点积，也就是计算 poly_features 的每一行的线性组合，也就是计算 poly_features 的每一行的线性回归
labels += np.random.normal(scale=0.1, size=labels.shape)    # 加上高斯噪声，scale=0.1 表示标准差为0.1，size=labels.shape 表示生成的噪声的形状和 labels 的形状相同

#假设features = [[1], [2], [3]]   shape:(3, 1)
# np.arange(max_degree) = [0, 1, 2, 3, 4]   shape:(max_degree,)=(5,)
# 则poly_features = np.dot(features, true_w) =
# 对于每一个 features[i]，我们会计算它的 0 次幂、1 次幂、2 次幂、3 次幂、4 次幂：
# 对于 features[0] = 1：
# 1^0 = 1
# 1^1 = 1
# 1^2 = 1
# 1^3 = 1
# 1^4 = 1
# 对于 features[1] = 2：
# 2^0 = 1
# 2^1 = 2
# 2^2 = 4
# 2^3 = 8
# 2^4 = 16
# 对于 features[2] = 3：
# 3^0 = 1
# 3^1 = 3
# 3^2 = 9
# 3^3 = 27
# 3^4 = 81
# 因此，poly_features 将会是：
# poly_features = np.array([[1, 1, 1, 1, 1],
#                           [1, 2, 4, 8, 16],
#                           [1, 3, 9, 27, 81]])
#shape of poly_features: (3,5)
# A 是 (m, n)，B 是 (n, p) → 点积结果是 (m, p)。
# A 是 (m, n)，B 是 (n,) → 点积结果是 (m,)。
# A 是 (n,)，B 是 (n,) → 点积结果是标量。

# 将 numpy 数组转换为 torch 张量
features = torch.tensor(features, dtype=torch.float32)
poly_features = torch.tensor(poly_features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# 前两个样本
print("前两个样本:")
print("features:", features[:2])
print("poly_features:", poly_features[:2])
print("labels:", labels[:2])    #labels[:2]输出shanpe为：(2,),因为labels是一个一维数组，它的shape为 (n_train + n_test,)
print('-' * 50)
#实际意义：
#features: 表示两个样本的特征，每个样本只有一个特征，这个特征是一个随机数，服从正态分布，均值为0，标准差为1
#poly_features: 表示两个样本的多项式特征，每个样本有 max_degree 个特征，这些特征是 features 的幂次方，从 0 次方到 max_degree-1 次方，并且每个特征都除以了相应的阶乘
#labels: 表示两个样本的标签，每个样本的标签是 poly_features 的每一行和 true_w 的点积，也就是 poly_features 的每一行的线性组合，也就是 poly_features 的每一行的线性回归，并且每个标签都加上了高斯噪声，服从正态分布，均值为0，标准差为0.1

# 定义评估损失的函数
def evaluate_loss(net, data_iter, loss_fn):
    """在给定数据集上评估模型的损失。

    参数:
    net (torch.nn.Module): 神经网络模型。
    data_iter (DataLoader): 数据加载器。
    loss_fn: 损失函数。

    返回:
    float: 数据集上的平均损失。
    """
    total_loss = 0.0    # 损失和
    n_samples = 0     # 样本数量
    with torch.no_grad():
        for X, y in data_iter:  #X是特征，y是标签,标签的形状为 (n,)，我们需要将其转换为 (n, 1) 的形状，以便于损失函数计算
            # 注意这里的 y.view(-1, 1) 是为了将 y 转换为 (n, 1) 的形状，因为损失函数要求输入的标签形状为 (n, 1)
            y_pred = net(X) #shape: (n, 1)
            loss = loss_fn(y_pred, y.view(-1, 1))   #loss_fn(y_pred, y.view(-1, 1)) 计算预测值和真实值的损失，y.view(-1, 1) 将 y 转换为 (n, 1) 的形状，因为损失函数要求输入的标签形状为 (n, 1)
            total_loss += loss.item() * y.shape[0]    #total_loss += loss.item() * y.shape[0] 计算损失的总和，loss.item() 是将损失转换为标量，y.shape[0] 是当前批次的样本数量
            n_samples += y.shape[0]    #n_samples += y.shape[0] 是计算样本数量的总和，y.shape[0] 是当前批次的样本数量
            #y.shape[0]表示y的行数，也就是当前批次的样本数量
    return total_loss / n_samples    #返回数据集上的平均损失


# 训练函数  功能包括：训练模型、评估模型、绘制损失曲线、打印学习到的权重
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    """训练模型并绘制训练和测试损失。

    参数:
    train_features (Tensor): 训练特征。
    test_features (Tensor): 测试特征。
    train_labels (Tensor): 训练标签。
    test_labels (Tensor): 测试标签。
    num_epochs (int): 训练的轮数。
    """

    #一般而言，训练data的shape[0]是data中数据的行数，也就是样本数量，shape[1]是data中数据的列数，也就是特征数量,输入维度

    # 定义损失函数
    loss_fn = nn.MSELoss()    #nn.MSELoss() 是均方误差损失函数，它的输入是预测值和真实值，输出是它们之间的误差

    # 定义模型
    net = nn.Linear(train_features.shape[1], 1, bias=False)    #nn.Linear是一个线性层，它的输入维度是train_features.shape[1]，输出维度是1,bias=False表示不使用偏置

    # 初始化权重
    for param in net.parameters():    #net.parameters() 是net的所有参数，包括权重和偏置
        # 初始化权重为正态分布，均值为0，标准差为0.01
        nn.init.normal_(param, mean=0, std=0.01)    #nn.init.normal_(param, mean=0, std=0.01) 是初始化权重为正态分布，均值为0，标准差为0.01
        #因为这里net的bias=False，所以这里只初始化权重，不初始化偏置
        #若bias=True，则还需要初始化偏置，初始化偏置为常数，值为0
        #若想让偏置等于1，则可以写成 nn.init.constant_(param, 1)

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.01)    #optim.SGD是随机梯度下降优化器，它的输入是模型的参数和学习率，输出是优化器

    # 准备数据加载器
    batch_size = min(10, train_labels.shape[0])    #batch_size = min(10, train_labels.shape[0]) 是为了防止 batch_size 超过数据的样本数量，这里取 batch_size 为 10 和数据的样本数量中较小的那个
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)    #TensorDataset是一个数据集，它将数据和标签打包成一个数据集，以便于批量处理,这是个迭代器
    train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)    #DataLoader是一个数据加载器，它将数据和标签打包成一个数据集，以便于批量处理

    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 存储损失
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):   #for epoch in range(num_epochs) 是训练的轮次，num_epochs 是训练的轮次
        net.train()    #net.train() 是将模型设置为训练模式，这样可以启用一些特殊的层，比如 Dropout 和 BatchNorm
        for X, y in train_iter:
            optimizer.zero_grad()    #optimizer.zero_grad() 是将优化器的梯度清零，这样可以防止梯度累积
            y_pred = net(X)    #y_pred = net(X) 是将特征 X 输入到模型 net 中，得到预测值 y_pred
            loss = loss_fn(y_pred, y.view(-1, 1))    #loss = loss_fn(y_pred, y.view(-1, 1)) 计算预测值和真实值的损失，y.view(-1, 1) 将 y 转换为 (n, 1) 的形状，因为损失函数要求输入的标签形状为 (n, 1)
            loss.backward()    #loss.backward() 是计算梯度，这样可以通过梯度下降法更新权重
            optimizer.step()    #optimizer.step() 是更新权重，这样可以通过梯度下降法更新权重

        # 在当前轮次评估损失
        net.eval()    #net.eval() 是将模型设置为评估模式，这样可以禁用一些特殊的层，比如 Dropout 和 BatchNorm
        train_loss = evaluate_loss(net, train_iter, loss_fn)    #train_loss = evaluate_loss(net, train_iter, loss_fn) 计算训练集上的平均损失
        test_loss = evaluate_loss(net, test_iter, loss_fn)    #test_loss = evaluate_loss(net, test_iter, loss_fn) 计算测试集上的平均损失
        train_losses.append(train_loss)    #train_losses.append(train_loss) 是将训练集上的平均损失添加到列表 train_losses 中
        test_losses.append(test_loss)    #test_losses.append(test_loss) 是将测试集上的平均损失添加到列表 test_losses 中

        # 可选地打印进度
        if epoch == 0 or (epoch + 1) % 20 == 0:    #epoch == 0 or (epoch + 1) % 20 == 0 是为了每隔 20 轮次打印一次进度
            print(f'轮次 {epoch + 1}, 训练损失: {train_loss:.6f}, 测试损失: {test_loss:.6f}')

    # 打印学习到的权重
    w = net.weight.data.numpy()    #net.weight.data.numpy() 是将模型的权重转换为 numpy 数组
    print('学习到的权重:', w)

    # 绘制损失曲线
    plt.figure()
    plt.semilogy(range(1, num_epochs + 1), train_losses, label='训练损失')  #semilogy 是 y 轴对数刻度，x 轴是轮次，y 轴是损失
    plt.semilogy(range(1, num_epochs + 1), test_losses, label='测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()    #legend 是图例，label 是图例的标签
    plt.show()


# 划分数据集
train_poly_features = poly_features[:n_train]   #取前n_train个样本作为训练集
test_poly_features = poly_features[n_test:]    #取后n_test个样本作为测试集

train_labels = labels[:n_train]   #取前n_train个样本作为训练集的标签
test_labels = labels[n_test:]     #取后n_test个样本作为测试集的标签

# 拟合三阶多项式（正常情况）
print("拟合三阶多项式（正常情况）")
print('-' * 50)
train(train_poly_features[:, :4], test_poly_features[:, :4], train_labels, test_labels)
print('-' * 50)

# 拟合线性模型（欠拟合）
print("拟合线性模型（欠拟合）")
print('-' * 50)
train(train_poly_features[:, :2], test_poly_features[:, :2], train_labels, test_labels)
print('-' * 50)

# 拟合20阶多项式（过拟合）
print("拟合20阶多项式（过拟合）")
print('-' * 50)
train(train_poly_features, test_poly_features, train_labels, test_labels, num_epochs=1500)
print('-' * 50)
'''
这三个 train 函数调用中的区别主要在于输入的特征数据的维度不同，这直接影响模型的复杂度以及拟合的效果。让我们逐步分析每个训练过程的差异和原因：

拟合三阶多项式（正常情况）
train(train_poly_features[:, :4], test_poly_features[:, :4], train_labels, test_labels)
特征选择：这里使用的是前三个多项式特征（[:, :4]）。通常，我们在构建多项式特征时，可能会通过某种方式构造不同阶数的特征（比如二次、三次、四次等），并选择一个合适的阶数进行训练。
模型复杂度：三阶多项式通常在一定程度上能够捕捉到数据的非线性关系，但不会过于复杂，因此它通常能在测试集上有一个良好的泛化能力。

拟合线性模型（欠拟合）
train(train_poly_features[:, :2], test_poly_features[:, :2], train_labels, test_labels)
特征选择：这里只使用了前两个多项式特征（[:, :2]）。这些特征通常是低阶的，代表了数据中的较简单模式。
模型复杂度：由于只使用了较低阶的特征，模型的表达能力比较弱，可能无法充分拟合训练数据中的复杂关系，容易产生欠拟合（underfitting）。也就是说，模型无法捕捉到数据中的非线性关系，因此在训练集和测试集上都会表现不佳。

拟合20阶多项式（过拟合）
train(train_poly_features, test_poly_features, train_labels, test_labels, num_epochs=1500)
特征选择：这里使用了所有的多项式特征（train_poly_features 和 test_poly_features），这些特征可能包括非常高阶的项，例如20阶的多项式特征。
模型复杂度：20阶的多项式模型非常复杂，能够拟合训练数据中的几乎所有细节，包括噪声。这种复杂的模型往往会在训练集上表现很好，但在测试集上表现较差，因为它过度拟合了训练集的细节，无法很好地泛化到新的数据，这就是过拟合（overfitting）的表现。

总结：区别和联系
输入特征的维度不同：在不同的训练过程中，输入特征的维度不同，导致了模型的复杂度也不同。三阶多项式有适度的复杂度，线性模型特征较少，复杂度较低，20阶多项式则拥有非常高的复杂度。
模型的拟合效果不同：
三阶多项式模型一般能够在训练集和测试集上取得较好的平衡，避免欠拟合和过拟合。
线性模型（低阶特征）可能会欠拟合，无法捕捉到数据中的复杂关系。
20阶多项式模型容易过拟合，虽然在训练集上表现很好，但无法很好地泛化到测试集，导致测试集的表现很差。
训练数据的维度和模型表现的关系
维度较低的特征（如线性模型的2阶特征）往往容易欠拟合，因为它们无法捕捉到数据中的复杂关系。
适度复杂的模型（如三阶多项式特征）通常能较好地拟合数据，取得较好的效果。
高维度的特征（如20阶多项式特征）通常容易过拟合，尤其是当数据量较小时，模型容易学习到训练数据中的噪声而不是数据的真实规律。
'''
# 练习1：这个多项式回归问题可以准确地解出吗？
print("练习1：使用线性代数精确解出多项式回归问题")
print('-' * 50)


def solve_exactly(features, labels):    #solve_exactly(features, labels) 是使用最小二乘法精确求解权重
    """使用最小二乘法精确求解权重。

    参数:
    features (Tensor): 特征矩阵。
    labels (Tensor): 标签向量。

    返回:
    numpy.ndarray: 精确求解的权重。
    """
    # 将张量转换为 numpy 数组
    X = features.numpy()
    y = labels.numpy()
    # 解正规方程 X^T X w = X^T y
    w = np.linalg.lstsq(X, y, rcond=None)[0]    #np.linalg.lstsq(X, y, rcond=None)[0] 是解正规方程 X^T X w = X^T y，rcond=None 是为了防止警告信息，rcond=None 表示使用默认的容差值
    return w


# 对三阶多项式精确求解权重
w_exact = solve_exactly(train_poly_features[:, :4], train_labels)
print("精确解出的权重:", w_exact)
print("真实权重:", true_w[:4])
print('-' * 50)

# 练习2：考虑多项式的模型选择
print("练习2：多项式的模型选择")
print('-' * 50)

degrees = list(range(1, max_degree + 1))
train_losses = []
test_losses = []

for degree in degrees:
    # 选择当前阶数的特征
    train_feats = train_poly_features[:, :degree]
    test_feats = test_poly_features[:, :degree]

    # 定义模型
    net = nn.Linear(train_feats.shape[1], 1, bias=False)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)

    # 定义优化器和损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # 准备数据加载器
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_feats, test_labels)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 100
    # 训练循环
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            y_pred = net(X)
            loss = loss_fn(y_pred, y.view(-1, 1))
            loss.backward()
            optimizer.step()
    # 评估损失
    net.eval()
    train_loss = evaluate_loss(net, train_iter, loss_fn)
    test_loss = evaluate_loss(net, test_iter, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f'阶数 {degree}, 训练损失: {train_loss:.6f}, 测试损失: {test_loss:.6f}')

# 绘制训练和测试损失与阶数的关系图
plt.figure()
plt.plot(degrees, train_losses, label='训练损失')
plt.plot(degrees, test_losses, label='测试损失')
plt.xlabel('多项式的阶数')
plt.ylabel('损失')
plt.legend()
plt.show()
print('-' * 50)

# 练习3：生成同样的图，作为数据量的函数
print("练习3：绘制损失与训练数据量的关系图")
print('-' * 50)

data_sizes = [5, 10, 20, 50, 70, 100]
train_losses = []
test_losses = []
degree = max_degree  # 使用20阶多项式

for n_train_samples in data_sizes:
    # 获取当前的数据量
    train_feats = train_poly_features[:n_train_samples, :degree]
    train_labs = train_labels[:n_train_samples]
    test_feats = test_poly_features[:, :degree]

    # 定义模型
    net = nn.Linear(train_feats.shape[1], 1, bias=False)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)

    # 定义优化器和损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # 准备数据加载器
    batch_size = min(10, train_labs.shape[0])
    dataset = torch.utils.data.TensorDataset(train_feats, train_labs)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_feats, test_labels)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 100
    # 训练循环
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            y_pred = net(X)
            loss = loss_fn(y_pred, y.view(-1, 1))
            loss.backward()
            optimizer.step()
    # 评估损失
    net.eval()
    train_loss = evaluate_loss(net, train_iter, loss_fn)
    test_loss = evaluate_loss(net, test_iter, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f'训练样本数: {n_train_samples}, 训练损失: {train_loss:.6f}, 测试损失: {test_loss:.6f}')

# 绘制损失与数据量的关系图
plt.figure()
plt.plot(data_sizes, train_losses, label='训练损失')
plt.plot(data_sizes, test_losses, label='测试损失')
plt.xlabel('训练样本数量')
plt.ylabel('损失')
plt.legend()
plt.show()
print('-' * 50)

# 练习4：使用未标准化的多项式特征进行训练
print("练习4：使用未标准化的多项式特征进行训练")
print('-' * 50)

# 生成未标准化的多项式特征
poly_features_no_norm = np.power(features.numpy(), np.arange(max_degree))
poly_features_no_norm = torch.tensor(poly_features_no_norm, dtype=torch.float32)

# 划分数据集
train_poly_features_no_norm = poly_features_no_norm[:n_train]
test_poly_features_no_norm = poly_features_no_norm[n_train:]

# 训练模型
print("使用未标准化的多项式特征进行训练（阶数=20）")
train(train_poly_features_no_norm, test_poly_features_no_norm, train_labels, test_labels, num_epochs=100)
print('-' * 50)

# 对特征进行标准化以解决问题
print("对未标准化的多项式特征进行标准化")
mean = train_poly_features_no_norm.mean(dim=0, keepdim=True)
std = train_poly_features_no_norm.std(dim=0, keepdim=True)
train_poly_features_std = (train_poly_features_no_norm - mean) / std
test_poly_features_std = (test_poly_features_no_norm - mean) / std

print("使用标准化后的未标准化多项式特征进行训练（阶数=20）")
train(train_poly_features_std, test_poly_features_std, train_labels, test_labels, num_epochs=100)
print('-' * 50)

# 在练习4中，我们生成了未标准化的多项式特征，即直接计算𝑥^(𝑖)而不除以𝑖!
# 由于特征𝑥是从标准正态分布中采样的，值可能在 -3 到 3 之间。当我们计算高次幂（如 20 次方）时，这些值会变得非常大或非常小（对于负数还会有符号交替），导致数值溢出或下溢。
#
# 举个例子：
# 如果 𝑥=3那么
# 𝑥^(20)=3^(20)≈3.5×10^(9)
# 如果𝑥=−3那么
# 𝑥^(20)=(−3)^(20)≈3.5×10^(9)
#
# 如此大的数值会导致在计算过程中出现溢出，权重更新时产生 inf 或 nan，最终导致模型无法学习到有效的参数。
# 即使在之后对这些未标准化的特征进行标准化（减去均值，除以标准差），由于原始数据中存在极端值，标准化后的数据仍然可能存在数值问题。

# 函数总结:
# - evaluate_loss(net, data_iter, loss_fn): 在数据集上评估模型的损失。
#   - net: 神经网络模型。
#   - data_iter: 数据加载器。
#   - loss_fn: 损失函数。
#
# - train(train_features, test_features, train_labels, test_labels, num_epochs=400):
#   训练模型并绘制训练和测试损失。
#   - train_features: 训练特征张量。
#   - test_features: 测试特征张量。
#   - train_labels: 训练标签张量。
#   - test_labels: 测试标签张量。
#   - num_epochs: 训练的轮数。
#
# - solve_exactly(features, labels): 使用最小二乘法精确求解权重。
#   - features: 特征矩阵张量。
#   - labels: 标签张量。
