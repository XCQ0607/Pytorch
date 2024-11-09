# -*- coding: utf-8 -*-

# 打印章节标题
print("2.6. 概率")

# 导入必要的包
import torch
from torch.distributions import multinomial
import matplotlib.pyplot as plt
from matplotlib import rcParams # 导入 Matplotlib 的 rcParams 模块

# 设置默认字体为支持中文的字体（例如：SimHei黑体）
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他支持中文的字体

# 分割线
print("-" * 50)

# 示例1：从多项式分布中采样
print("示例1：从多项式分布中采样")

# 定义公平骰子的概率分布
# 创建一个长度为6的张量，表示骰子6个面的概率均为1/6
fair_probs = torch.ones(6) / 6

# multinomial.Multinomial()函数用于创建一个多项式分布
# 参数total_count表示一次实验中抽取的样本总数
# 参数probs表示每个事件的概率分布
# sample()方法用于从该分布中采样

# 抛掷一次骰子，抽取1个样本
single_roll = multinomial.Multinomial(1, fair_probs).sample()
print("抛掷一次骰子，结果：", single_roll)

# 抛掷10次骰子，抽取10个样本
multiple_rolls = multinomial.Multinomial(10, fair_probs).sample()
print("抛掷10次骰子，结果：", multiple_rolls)

# 分割线
print("-" * 50)

# 示例2：验证大数定律
print("示例2：验证大数定律")

# 进行500组实验，每组投掷10次骰子
num_experiments = 500
num_trials = 10

# 初始化计数器，用于统计每个面的出现次数
counts = torch.zeros(6)

for i in range(num_experiments):
    # 从多项式分布中采样
    rolls = multinomial.Multinomial(num_trials, fair_probs).sample()
    # 累加每个面的出现次数
    counts += rolls

# 计算每个面的相对频率
estimated_probs = counts / (num_experiments * num_trials)
print("每个面的估计概率：", estimated_probs)

# 绘制估计概率的收敛图像
cumulative_counts = torch.zeros(num_experiments, 6)
for i in range(num_experiments):
    rolls = multinomial.Multinomial(num_trials, fair_probs).sample()
    if i == 0:
        cumulative_counts[i] = rolls
    else:
        cumulative_counts[i] = cumulative_counts[i - 1] + rolls

# 计算累计估计概率
cumulative_estimates = cumulative_counts / ((torch.arange(num_experiments).view(-1, 1) + 1) * num_trials)

# 绘制图形
for i in range(6):
    plt.plot(cumulative_estimates[:, i].numpy(), label=f"P(die={i+1})")
plt.axhline(y=1/6, color='black', linestyle='dashed')
plt.xlabel('实验组数')
plt.ylabel('估计概率')
plt.legend()
plt.show()

# 分割线
print("-" * 50)

# 示例3：贝叶斯定理应用
print("示例3：贝叶斯定理应用")

# 定义条件概率
# P(D=1|H=1) = 1
# P(D=1|H=0) = 0.01
# P(H=1) = 0.0015

# 定义变量
P_H1 = 0.0015  # 患病概率
P_H0 = 1 - P_H1  # 未患病概率
P_D1_H1 = 1  # 患病且测试阳性的概率
P_D1_H0 = 0.01  # 未患病但测试阳性的概率

# 计算边缘概率P(D=1)
P_D1 = P_D1_H1 * P_H1 + P_D1_H0 * P_H0
print("P(D=1) =", P_D1)

# 计算后验概率P(H=1|D=1)
P_H1_D1 = (P_D1_H1 * P_H1) / P_D1
print("P(H=1|D=1) =", P_H1_D1)

# 分割线
print("-" * 50)

# 示例4：计算期望和方差
print("示例4：计算期望和方差")

# 定义一个随机变量的概率分布
values = torch.tensor([1, 2, 3, 4, 5, 6])
probs = torch.ones(6) / 6

# 计算期望E[X]
expectation = torch.sum(values * probs)
print("期望E[X] =", expectation.item())

# 计算方差Var[X]
variance = torch.sum((values - expectation) ** 2 * probs)
print("方差Var[X] =", variance.item())

# 分割线
print("-" * 50)

# 总结：
# 本代码示例中使用了以下函数：
# 1. torch.distributions.multinomial.Multinomial(total_count=1, probs=None, logits=None)
#    用于创建多项式分布。
#    参数：
#    - total_count：实验总次数，默认为1。
#    - probs：事件的概率分布，张量类型。
#    - logits：事件概率的对数，probs和logits二选一。
#    示例：
#    multinomial.Multinomial(10, probs=fair_probs)

# 2. sample(sample_shape=torch.Size())
#    从指定的分布中采样。
#    参数：
#    - sample_shape：采样的形状，默认为空。
#    示例：
#    multinomial.Multinomial(10, fair_probs).sample()

# 3. torch.sum(input, dim=None)
#    对张量进行求和。
#    参数：
#    - input：输入张量。
#    - dim：进行求和的维度。
#    示例：
#    torch.sum(values * probs)

# 4. torch.arange(start=0, end, step=1)
#    生成从start到end的等差数列张量。
#    示例：
#    torch.arange(num_experiments)

# 5. plt.plot()
#    绘制二维线条图。
#    示例：
#    plt.plot(cumulative_estimates[:, i].numpy())

# 6. plt.show()
#    显示图形。

