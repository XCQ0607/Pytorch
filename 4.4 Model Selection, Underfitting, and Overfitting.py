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
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

# 生成特征 x ~ N(0, 1)
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)

# 生成多项式特征，最高到 max_degree 次方
poly_features = np.power(features, np.arange(max_degree))
# 使用阶乘进行标准化
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n) = (n-1)!

# 使用 true_w 生成标签，并添加高斯噪声
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# 将 numpy 数组转换为 torch 张量
features = torch.tensor(features, dtype=torch.float32)
poly_features = torch.tensor(poly_features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# 前两个样本
print("前两个样本:")
print("features:", features[:2])
print("poly_features:", poly_features[:2])
print("labels:", labels[:2])
print('-' * 50)


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
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            y_pred = net(X)
            loss = loss_fn(y_pred, y.view(-1, 1))
            total_loss += loss.item() * y.shape[0]
            n_samples += y.shape[0]
    return total_loss / n_samples


# 训练函数
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    """训练模型并绘制训练和测试损失。

    参数:
    train_features (Tensor): 训练特征。
    test_features (Tensor): 测试特征。
    train_labels (Tensor): 训练标签。
    test_labels (Tensor): 测试标签。
    num_epochs (int): 训练的轮数。
    """
    # 定义损失函数
    loss_fn = nn.MSELoss()

    # 定义模型
    net = nn.Linear(train_features.shape[1], 1, bias=False)

    # 初始化权重
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # 准备数据加载器
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 存储损失
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            y_pred = net(X)
            loss = loss_fn(y_pred, y.view(-1, 1))
            loss.backward()
            optimizer.step()

        # 在当前轮次评估损失
        net.eval()
        train_loss = evaluate_loss(net, train_iter, loss_fn)
        test_loss = evaluate_loss(net, test_iter, loss_fn)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # 可选地打印进度
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f'轮次 {epoch + 1}, 训练损失: {train_loss:.6f}, 测试损失: {test_loss:.6f}')

    # 打印学习到的权重
    w = net.weight.data.numpy()
    print('学习到的权重:', w)

    # 绘制损失曲线
    plt.figure()
    plt.semilogy(range(1, num_epochs + 1), train_losses, label='训练损失')
    plt.semilogy(range(1, num_epochs + 1), test_losses, label='测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.show()


# 划分数据集
train_poly_features = poly_features[:n_train]
test_poly_features = poly_features[n_train:]

train_labels = labels[:n_train]
test_labels = labels[n_train:]

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

# 练习1：这个多项式回归问题可以准确地解出吗？
print("练习1：使用线性代数精确解出多项式回归问题")
print('-' * 50)


def solve_exactly(features, labels):
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
    w = np.linalg.lstsq(X, y, rcond=None)[0]
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
