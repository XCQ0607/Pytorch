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
#single_roll是一个长度为6的张量，每个元素表示对应面出现的次数
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
    # 这里的count是一个长度为6的张量，每个元素表示对应面出现的次数
    counts += rolls

# 计算每个面的相对频率
estimated_probs = counts / (num_experiments * num_trials)
print("每个面的估计概率：", estimated_probs)

# 绘制估计概率的收敛图像
cumulative_counts = torch.zeros(num_experiments, 6) # 初始化累计估计概率的张量，这个代码会生成一个500x6的张量，每个元素表示对应面出现的次数
for i in range(num_experiments):
    rolls = multinomial.Multinomial(num_trials, fair_probs).sample()    #做一次实验，抽取10次骰子
    if i == 0:
        cumulative_counts[i] = rolls
    else:
        cumulative_counts[i] = cumulative_counts[i - 1] + rolls

# 计算累计估计概率
cumulative_estimates = cumulative_counts / ((torch.arange(num_experiments).view(-1, 1) + 1) * num_trials)

#view(-1, 1)中的1表示在改变张量形状时，第二维（列）的大小为1。view函数用于重塑张量，而-1是一个占位符，它告诉PyTorch自动计算该维度的大小，以保持元素总数不变。
# cumulative_counts 的形状是 [num_experiments, num_faces]（假设我们在模拟一个num_faces面的骰子），那么上述表达式中的除法操作将会广播（broadcast），使得每个实验的累积计数都被其对应的权重（由实验编号和试验数量决定）所除。
# 这种加权方式可以调整每个实验数据点对最终结果的影响程度。
#torch.arange(num_experiments).view(-1, 1) + 1
#假设num_experiments为10，那它生成
# [
#  [1],
#  [2],
#  [3],
#  [4],
#  [5],
#  [6],
#  [7],
#  [8],
#  [9],
#  [10]
# ]
# 如果原始张量是[0, 1, 2, 3, 4]，那么view(-1, 1)之后的结果将是：
# [
#  [0],
#  [1],
#  [2],
#  [3],
#  [4]
# ]
# + 1:
# 这个操作将上述二维张量中的每个元素加1。所以，上述张量将变为：
# [
#  [1],
#  [2],
#  [3],
#  [4],
#  [5]
# ]


# 绘制图形
for i in range(6):
    # 绘制每个面的估计概率的曲线[:, i]表示取第i列，也就是第i个面的估计概率的曲线
    plt.plot(cumulative_estimates[:, i].numpy(), label=f"P(die={i+1})")
    # 表示从cumulative_estimates张量中取所有行（:表示所有行）和第i列的数据。这里的i代表骰子的某一个面（假设我们在模拟掷骰子的实验），循环将遍历并绘制每个面的估计概率曲线。
    # cumulative_estimates的形状应该是[num_experiments, num_faces]，其中num_faces是骰子面的数量（在这个例子中是6）

# 绘制黑色虚线，代表真实概率，类型为虚线
plt.axhline(y=1/6, color='black', linestyle='dashed')
plt.xlabel('实验组数')
plt.ylabel('估计概率')
plt.title('估计概率的收敛图像')
# 设置图例，也就是每个面的估计概率的曲线
plt.legend()
plt.show()

# 通过多次实验来估计掷骰子每个面出现的概率，并希望将这些概率的收敛过程可视化。确实，如果您只是简单地想表示每个面在n次实验中出现的频率，那么您可以直接使用每个面出现的次数除以总试验次数来得到估计的概率，而无需进行加权。
# 然而，加权可能在某些特定情境下是有意义的：
# 非均匀采样：如果实验不是均匀进行的，比如早期的实验次数较少，而随着时间的推移实验次数增加，那么加权可以帮助调整这种不平衡，使得每个实验阶段的贡献更加均衡。
# 强调近期数据：在某些在线学习或流数据处理的场景中，可能更希望强调最近的数据点。通过给近期的实验数据更高的权重，可以让模型更快地适应数据分布的变化。
# 减少初始化偏差：在实验的初始阶段，由于数据点较少，估计的概率可能会有较大的偏差。通过给后续的实验数据更高的权重，可以减少这种初始化偏差的影响。
# 但在您描述的掷骰子实验中，如果实验是均匀进行的，且没有特别的理由去强调某些实验阶段的数据，那么确实可以直接使用频率来估计概率，而不需要额外的加权。

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

# 计算边缘概率P(D=1)  即为P(D=1) = P(D=1|H=1) * P(H=1) + P(D=1|H=0) * P(H=0)
P_D1 = P_D1_H1 * P_H1 + P_D1_H0 * P_H0
#阳性概率
print("P(D=1) =", P_D1)

# 计算后验概率P(H=1|D=1) 即为P(H=1|D=1) = P(D=1|H=1) * P(H=1) / P(D=1)
P_H1_D1 = (P_D1_H1 * P_H1) / P_D1
# 测试结果为阳性的条件下，患者真正患病的概率。
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

# 计算方差Var[X]    即E[(X - E[X])^2]
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

