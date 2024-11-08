# 2.3. 线性代数
from numpy.ma.core import product

print("2.3. 线性代数")

# 导入PyTorch库
import torch

#种子设置
torch.manual_seed(0)


# 2.3.1. 标量
print("\n2.3.1. 标量")

# 创建标量 x 和 y
x = torch.tensor(3.0)
y = torch.tensor(2.0)

# 展示标量的基本算术运算
print("标量加法示例:")
print(f"x + y = {x + y}")

print("标量乘法示例:")
print(f"x * y = {x * y}")

print("标量除法示例:")
print(f"x / y = {x / y}")

print("标量指数运算示例:")
print(f"x ** y = {x ** y}")

# 2.3.2. 向量
print("\n2.3.2. 向量")

# 使用 torch.arange() 创建一维张量（向量）
# torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
# 参数说明：
# - start（可选）：起始值，默认0
# - end（必选）：结束值（不包含该值）
# - step（可选）：步长，默认1
x = torch.arange(4)
print("向量 x:")
print(x)

# 访问向量的某个元素
index = 3
print(f"向量 x 的第 {index} 个元素（从0开始计数）:")
print(x[index])

# 2.3.2.1. 长度、维度和形状
print("\n2.3.2.1. 长度、维度和形状")

# 获取向量的长度
length = len(x)
print(f"向量 x 的长度为: {length}")

# 获取向量的形状
shape = x.shape
print(f"向量 x 的形状为: {shape}")

# 2.3.3. 矩阵
print("\n2.3.3. 矩阵")

# 创建一个 5 行 4 列的矩阵
# 使用 torch.arange() 和 reshape()
A = torch.arange(20).reshape(5, 4)
print("矩阵 A:")
print(A)

# 访问矩阵的某个元素
row = 2
col = 3
print(f"矩阵 A 第 {row+1} 行，第 {col+1} 列的元素:")
print(A[row, col])

# 矩阵转置
print("矩阵 A 的转置:")
print(A.T)

# 创建一个对称矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print("对称矩阵 B:")
print(B)

# 验证矩阵是否对称
print("验证矩阵 B 是否对称 (B == B.T):")
print(B == B.T)

# 2.3.4. 张量
print("\n2.3.4. 张量")

# 创建一个形状为 (2, 3, 4) 的张量
X = torch.arange(24).reshape(2, 3, 4)
print("张量 X:")
print(X)

# 2.3.5. 张量算法的基本性质
print("\n2.3.5. 张量算法的基本性质")

# 按元素运算
print("矩阵 A 和 A 的按元素加法 (A + A):")
print(A + A)

# 按元素乘法（Hadamard积）
print("矩阵 A 和 A 的按元素乘法 (A * A):")
print(A * A)

# 标量与张量的运算
a = 2
print(f"标量 a = {a}")
print("标量与张量相加 (a + X):")
print(a + X)
print("标量与张量相乘 (a * X):")
print(a * X)

# 2.3.6. 降维
print("\n2.3.6. 降维")

# 求和
print("向量 x:")
print(x)
sum_x = x.sum()
print(f"向量 x 的元素和: {sum_x}")

print("矩阵 A:")
print(A)
sum_A = A.sum()
print(f"矩阵 A 的元素和: {sum_A}")

# 指定维度求和
sum_axis0 = A.sum(axis=0)
print("矩阵 A 在轴 0 上的求和 (按列求和):")
print(sum_axis0)
print(f"结果形状: {sum_axis0.shape}")

sum_axis1 = A.sum(axis=1)
print("矩阵 A 在轴 1 上的求和 (按行求和):")
print(sum_axis1)
print(f"结果形状: {sum_axis1.shape}")

# 求平均值
A = A.float()
#PyTorch 要求输入必须是浮点类型或复数类型。
mean_A = A.mean()
print(f"矩阵 A 的平均值: {mean_A}")

mean_axis0 = A.mean(axis=0)
print("矩阵 A 在轴 0 上的平均值:")
print(mean_axis0)

# 非降维求和
# keepdim全称是keep dimensions，即保持维度不变。
sum_A_keepdim = A.sum(axis=1, keepdims=True)
print("矩阵 A 在轴 1 上的求和，保持维度:")
print(sum_A_keepdim)
print(f"结果形状: {sum_A_keepdim.shape}")

# 利用广播机制
# 广播机制指的是在进行运算时，PyTorch 会自动将较小的张量广播到较大的张量的形状，以便进行逐元素运算。
print("矩阵 A 除以其每行的元素和:")
print(A / sum_A_keepdim)
# A.shape: (5,4)
# sum_A_keepdim.shape: (5,1)
# A中每个轴1上的值除以sum_A_keepdim中对应轴1上的值
# 广播方向：从较小的张量的最后一个维度开始，向较大的张量的最后一个维度对齐，然后进行逐元素运算。

# 累积求和
#cumsum是cumulative sum的缩写，即累积求和。
#累计求和计算方法是将向量中的元素依次相加，得到一个新的向量，其中每个元素都是前面所有元素的和。
cumsum_A = A.cumsum(axis=1)
print("矩阵 A 在轴 0 上的累积求和:")
print(cumsum_A)

#原A
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15],
#         [16, 17, 18, 19]])
#累计求和sum axis=0 外层累计求和（向下）
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  6.,  8., 10.],
#         [12., 15., 18., 21.],
#         [24., 28., 32., 36.],
#         [40., 45., 50., 55.]])
#累计求和sum axis=1 内层累计求和（向右）
# tensor([[ 0.,  1.,  3.,  6.],
#         [ 4.,  9., 15., 22.],
#         [ 8., 17., 27., 38.],
#         [12., 25., 39., 54.],
#         [16., 33., 51., 70.]])



# 2.3.7. 点积（Dot Product）
print("\n2.3.7. 点积（Dot Product）")

# 创建两个向量
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print("向量 x:")
print(x)
print("向量 y:")
print(y)

# 确保向量 x 和 y 的形状相同
assert x.shape == y.shape, "向量 x 和 y 的形状不匹配"
# 计算点积
dot_product = torch.dot(x, y)
print(f"向量 x 和 y 的点积: {dot_product}")
# 计算方法torch.dot(x, y)=x.T@y     @：矩阵乘法
#=x0y0+x1y1+x2y2+x3y3

# 手动计算点积
manual_dot = (x * y).sum()
print(f"手动计算的点积: {manual_dot}")
#x*y与matmul(x,y)区别：
# x*y：对应元素相乘，结果是一个新的张量，形状x,y相同
# matmul(x,y)：矩阵乘法，要求x和y的维度匹配，结果是一个新的张量，形状与x和y的维度匹配

# 2.3.8. 矩阵-向量积
print("\n2.3.8. 矩阵-向量积")

# 矩阵 A 和向量 x
print("矩阵 A:")
print(A)
print("向量 x:")
print(x)

# 计算矩阵-向量积
# torch.mv(input, vec, *, out=None) -> Tensor
# 参数说明：
# - input（必选）：矩阵，形状为 (m, n)
# - vec（必选）：向量，形状为 (n)
# 返回值：结果向量，形状为 (m)
mv_result = torch.mv(A, x)
print("矩阵 A 和向量 x 的矩阵-向量积 (A * x):")
print(mv_result)
#结果：tensor([ 14.,  38.,  62.,  86., 110.])

# 矩阵 A:
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.],
#         [16., 17., 18., 19.]])
# 向量 x:
# tensor([0., 1., 2., 3.])
'''
0    1    2    3                        0         1         2        3      14
4    5    6    7                        4         5         6        7      38
8    9   10   11  x [0   1   2   3] =0x 8   + 1x  9   + 2x 10  + 3x 11  =   62
12  13   14   15                        12       13        14       15      86
16  17   18   19                        16       17        18       19      110
'''

# 2.3.9. 矩阵-矩阵乘法
print("\n2.3.9. 矩阵-矩阵乘法")
#矩阵A
print("矩阵 A:")
print(A)

# 创建矩阵 B，形状为 (4, 3)
B = torch.arange(12, dtype=torch.float32).reshape(4, 3)
print("矩阵 B:")
print(B)


# 矩阵 A:
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.],
#         [16., 17., 18., 19.]])
# 矩阵 B:
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.],
#         [ 6.,  7.,  8.],
#         [ 9., 10., 11.]])
#
#                         [ 0.,   1.,   2. ]
#                         [ 3.,   4.,   5. ]
#                         [ 6.,   7.,   8. ]
#                         [ 9.,   10.,  11.]
#
# [ 0.,  1.,  2.,  3.]    [ 42.,  48.,  54.]
# [ 4.,  5.,  6.,  7.]    [114., 136., 158.]
# [ 8.,  9., 10., 11.]    [186., 224., 262.]
# [12., 13., 14., 15.]    [258., 312., 366.]
# [16., 17., 18., 19.]    [330., 400., 470.]
#42=0*0+1*3+2*6+3*9

# 计算矩阵-矩阵乘法
# torch.mm(input, mat2, *, out=None) -> Tensor
# 参数说明：
# - input（必选）：矩阵，形状为 (m, n)
# - mat2（必选）：矩阵，形状为 (n, p)
# 返回值：结果矩阵，形状为 (m, p)
mm_result = torch.mm(A, B)
print("矩阵 A 和矩阵 B 的矩阵-矩阵乘法 (A * B):")
print(mm_result)

# 2.3.10. 范数
print("\n2.3.10. 范数")
#范数指的是一个向量或矩阵中所有元素的绝对值的总和。

# 向量的 L2 范数
u = torch.tensor([3.0, -4.0])
l2_norm = torch.norm(u)
print(f"向量 u 的 L2 范数: {l2_norm}")

# 向量的 L1 范数
l1_norm = torch.abs(u).sum()
print(f"向量 u 的 L1 范数: {l1_norm}")

# 范数计算
# 范数指的是一个向量或矩阵中所有元素的绝对值的总和。
# p值默认为2
# print(f"L1范数: {torch.norm(data, p=1)}")  # L1范数指的是一个向量中所有元素的绝对值之和。
# print(f"L2范数: {torch.norm(data, p=2)}")  # L2范数指的是一个向量中所有元素的平方和的平方根。
# print(f"无穷范数: {torch.norm(data, p=float('inf'))}")  # 无穷范数指的是一个向量中所有元素绝对值的最大值。
# float('inf')指的是正无穷大。类似的，float('-inf')指的是负无穷大。
# P参数：
# p=1, 则范数为向量中所有元素绝对值的和。
# p=2, 则范数为向量中所有元素平方和的平方根。
# p=3, 则范数为向量中所有元素立方和的立方根。
# p=4, 则范数为向量中所有元素四次方和的四次方根。
# p=∞, 则范数为向量中所有元素绝对值的最大值。


# 矩阵的 Frobenius 范数
# 矩阵的 Frobenius 范数指的是矩阵中所有元素的平方和的平方根。也就是L2范数。
fro_norm = torch.norm(torch.ones((4, 9)))
print(f"矩阵的 Frobenius 范数: {fro_norm}")

# 对于高维张量的范数计算
#randn()函数用于生成服从标准正态分布的张量。即：从均值为0，标准差为1的正态分布中采样得到的张量。
#randint()函数用于生成服从均匀分布的张量。即：从指定范围中采样得到的张量。如果不指定范围，则默认为[0, 10)。指定范围调用实例：torch.randint(0, 20, (3, 4))
#rand()函数用于生成服从均匀分布的张量。即：从指定范围中采样得到的张量。如果不指定范围，则默认为[0, 1)。指定范围调用实例：torch.rand(3, 4)（小数）
# 指定范围 [a, b)
a = 5
b = 10
# 使用 torch.rand() 生成 0 到 1 之间的随机数，然后映射到 [a, b)
Q = a + (b - a) * torch.rand((2, 3, 4))  # 形状为 (2, 3, 4) 的张量，取值范围为 [a, b)
X = torch.randn(2, 3, 4)    # 形状为 (2, 3, 4) 的张量
Y = torch.randint(0, 10, (2  ,3, 4))    # 形状为 (3, 4) 的张量, 取值范围为[0, 10)
Z = torch.rand(2 , 3 , 4)    # 形状为 (3, 4) 的张量, 取值范围为[0, 1)
print("张量 X:")
print(X)
print("张量 Y:")
print(Y)
print("张量 Z:")
print(Z)
print("张量 Q:")
print(Q)

# 计算张量的范数
# torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None) -> Tensor
# 参数说明：
# - input（必选）：输入张量
# - p（可选）：范数类型，默认 'fro'（Frobenius 范数）
#   - 对于向量：p 可以是数值，如 1，2，inf 等
#   - 对于矩阵：p 可以是 'fro'（Frobenius 范数）或核范数 'nuc'
# - dim（可选）：要计算范数的维度
# - keepdim（可选）：是否保持维度
tensor_norm = torch.norm(X)
print(f"张量 X 的范数（Frobenius 范数）: {tensor_norm}")

# 计算张量在特定维度上的范数
tensor_norm_dim = torch.norm(X, dim=0,keepdim=True) #shape: (1,3,4)
#tensor_norm_dim = torch.norm(X, dim=0) #shape: (3,4)
print("张量 X 在维度 0 上的范数:")
print(tensor_norm_dim)

# 张量 X:
# tensor([[[-0.0245, -0.4406, -0.6549, -0.3274],
#          [-0.3368,  1.4618, -0.1700, -0.8595],
#          [-0.1285,  1.6010,  1.8653,  0.5617]],
#
#         [[ 1.4788,  1.7060,  1.9647,  2.7830],
#          [-1.0943, -0.4549, -0.2399, -3.4315],
#          [-0.1998,  1.4026,  1.3948,  0.9239]]])
# 张量 X 在维度 0 上的范数:
# tensor([[1.4790, 1.7620, 2.0710, 2.8022],
#         [1.1449, 1.5309, 0.2940, 3.5375],
#         [0.2376, 2.1284, 2.3291, 1.0813]])
#
# 1.4790=sqrt((-0.0245)^2+(1.4788)^2)
# *sqrt：开平方



# 矩阵×矩阵的基础性质
# 第一部分，在满足矩阵可乘的条件下：
# 第1条：A×B≠B×A，指A×B不一定等于B×A
# -- 补充：当A×B=B×A时，称AB可交换
# -- 补充：当A×B，称：矩阵B右乘A，也称：矩阵A左乘B
# 第2条：若A×B=0，是无法推出A或B=0的
# 第3条：若A×B=A×C，且A≠0，无法推出B=C（任何矩阵都不可约）
# 第4条：零矩阵与任何矩阵相乘，都等于一个零矩阵
# 第二部分，综合运算的性质：
# 结合律：(A×B)×C=A×(B×C)
# 分配率1：(A+B)×C=A×C+B×C，C×(A+B)=C×A+C×B
# 分配率2：k(A×B)=(k×A)×B=A×(k×B)，k为常数
# 特别注意：无论可用哪种运算律变形，变形后都有：
# 依然按照矩阵原式中，矩阵从左至右的顺位，依次进行计算
# 例：(A×B)×C=A×(B×C)，不能为：(A×B)×C=(B×C)×A
# 即：破坏了原式中矩阵A、B，从左至右的顺序
# 例：(A+B)×C=A×C+B×C，能为(修正)：(A+B)×C=B×C+A×C
# 因为矩阵加法满足交换律
# 例：k(A×B)=(k×A)×B，k为常数，不能为：k(A×B)=(K×B)×A
# 即：还是破坏了原式中矩阵A、B，从左至右的顺序

# A.T*B.T=(B*A).T
