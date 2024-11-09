# 2.4. 微积分
from six import print_

print("2.4. 微积分")

# 导入必要的库
import torch
from torch import autograd  # 导入自动微分模块
import numpy as np  # 导入 numpy 库
import matplotlib.pyplot as plt # 导入绘图库
from matplotlib import rcParams # 导入 Matplotlib 的 rcParams 模块

# 设置默认字体为支持中文的字体（例如：SimHei黑体）
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他支持中文的字体

# 设置在Jupyter中显示图形为SVG格式
# 这可以使图形更清晰
from matplotlib_inline import backend_inline
# 这将使所有的绘图都显示为 SVG 格式
backend_inline.set_matplotlib_formats('svg')

# 2.4.1. 导数和微分
print("\n2.4.1. 导数和微分")

# 定义一个函数 f(x) = 3x^2 - 4x
def f(x):
    return 3 * x ** 2 - 4 * x

# 定义一个函数来计算数值导数的近似值
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

# 设置初始的 h 值
h = 0.1

print("数值导数近似示例:")
# 计算并打印不同 h 值下的数值导数近似值
for i in range(5):
    approx = numerical_lim(f, 1, h)
    print(f"h = {h:.5f}, 数值导数近似值 = {approx:.5f}")
    h *= 0.1

# 使用 PyTorch 的自动微分计算导数
print("\n使用 PyTorch 自动微分计算导数:")

# 需要使用 requires_grad=True 来告诉 PyTorch 需要计算梯度
x = torch.tensor([1.0,2.0,3.0], requires_grad=True)
y = 3 * x ** 2 - 4 * x
# 计算梯度
# 梯度指的是 函数在某一点的变化率，是函数的斜率
#这行代码使得x.grad 保存了 y 对 x 的导数值，x.grad中的grad是x的一个属性
y.backward(gradient=torch.ones_like(y))    #backward() 是 PyTorch 中的一个函数，用于计算梯度
# x.grad 保存了 y 对 x 的导数值
for i in range(len(x)):
    print(f"在 x = {x[i].item()} 处，y的值是y = {y[i]}，y 对 x 的导数值为: {x.grad[i].item()}")
    #x.grad 保存了 y 对 x 的导数值

#不需要y.backward就能输出y的函数值

# torch.ones_like(y)：
# 这个函数会创建一个与 y 形状相同（在这个例子中是一个一维张量，长度为 3）的新张量。
# 新张量的所有元素都会被填充为 1。
# 因此，torch.ones_like(y) 的输出结果将是 [1, 1, 1]。
# torch.copy(y)：
# 这个函数会创建 y 的一个副本，即一个与 y 形状和数据都相同的新张量。
# 因此，torch.copy(y) 的输出结果将是 [1, 2, 3]，与原始的 y 相同。

# torch.ones_like(input, , dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.contiguous_format)
# 参数:
# input: 一个已存在的张量，torch.ones_like 会基于这个张量的形状来创建新的全1张量。
# dtype, layout, device, requires_grad, memory_format: 这些都是可选参数，用于指定新张量的数据类型、内存布局、设备位置、是否需要梯度和内存格式。
# 作用: torch.ones_like 创建一个与输入张量 input 形状完全相同的新张量，并且新张量中的所有元素都被初始化为 1。
# torch.ones(size, , dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 参数:
# size: 一个整数序列，定义了输出张量的形状。例如，(2, 3) 会创建一个 2x3 的二维张量。
# dtype, layout, device, requires_grad: 与 torch.ones_like 中的可选参数类似，用于指定新张量的属性。
# 作用: torch.ones 根据指定的形状 size 创建一个新的全1张量。与 torch.ones_like 不同，torch.ones 需要明确指定张量的形状，而不是基于一个已存在的张量。
# 主要区别:
# torch.ones_like 是基于一个现有的张量来创建形状相同的新全1张量，而 torch.ones 需要你显式地指定新张量的形状。

# 绘制函数及其在 x=1 处的切线
print("\n绘制函数及其在 x=1 处的切线:")

print("------测试-----")
# 假设 x_vals 是一个 PyTorch 张量
x_vals_tensor = torch.arange(0, 3, 0.1)
# x_vals_tensor = torch.tensor([1.0, 2.0, 3.0])     # 创建一个包含三个元素的张量
# y_vals_tensor = 3 * x_vals_tensor ** 2 - 4 * x_vals_tensor
y_vals_tensor = f(x_vals_tensor)

# 将张量转换为 NumPy 数组
x_vals = x_vals_tensor.numpy()
y_vals = y_vals_tensor.numpy()

# 现在可以使用这些 NumPy 数组进行绘图
plt.plot(x_vals, y_vals, label='f(x) = 3x^2 - 4x')
plt.xlabel('x') # x 轴标签
plt.ylabel('f(x)')  # y 轴标签
plt.legend()    # 添加图例
plt.grid()  # 添加网格
plt.show()     # 显示图形
print("--------------")

# 定义 x 的范围
#np.arange(start, stop, step)，这里不使用tensor.arange()  arange()返回的是一个numpy数组，而不是一个tensor
#在后续绘图中，numpy数组的元素可以直接用于绘图
x_vals = np.arange(0, 3, 0.1)
# 计算函数值
y_vals = f(x_vals)
# 计算切线 y = 2x - 3
tangent_line = 2 * x_vals - 3

# plt.plot() 函数是 Matplotlib 库中用于绘制二维图形的主要函数。在这个函数中，x_vals 和 y_vals 参数通常期望是 Python 的可迭代对象，比如列表（list）或者 NumPy 数组（numpy.ndarray），它们分别代表图上点的 x 坐标和 y 坐标。
# 直接使用可能不行：如果你尝试直接将一个 PyTorch 或 TensorFlow 的 tensor 传递给 plt.plot()，可能会遇到错误，因为 Matplotlib 不一定能够直接处理这些特定框架的张量类型。
# 转换为 NumPy 数组：不过，你可以很容易地将这些张量转换为 NumPy 数组，然后使用这些数组进行绘图。例如，在 PyTorch 中，你可以使用 .numpy() 方法来转换张量：
# x_vals = x_vals_tensor.numpy()
# y_vals = y_vals_tensor.numpy()

#x_vals是x轴的坐标的范围以及步长，y_vals是函数表达式，label是图例的标签
# 创建图形
plt.figure()
plt.plot(x_vals, y_vals, label='f(x) = 3x^2 - 4x')
plt.plot(x_vals, tangent_line, '--', label='切线 (x=1)')  #'--'表示虚线
# '-' ：实线
# '--'：虚线
# '-.'：点划线（虚线与点交替）
# ':'：点线（由点组成的线）
# 'None' 或 ''：不绘制线条（通常与标记一起使用，如只绘制点）
plt.xlabel('x') # x 轴标签
plt.ylabel('f(x)')  # y 轴标签
plt.legend()    # 添加图例
plt.grid()  # 添加网格
plt.show()

# 2.4.2. 偏导数
print("\n2.4.2. 偏导数")

# 定义一个多元函数 z = x^2 + y^2 - 1
def func(x, y):
    return x ** 2 + y ** 2 - 1

# 使用 PyTorch 计算偏导数
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = func(x, y)

# 计算梯度
z.backward()

print("函数：z = x^2 + y^2")
print(f"原函数函数值{z.item()}")
print(f"在 x = {x.item()}, y = {y.item()} 处，z 对 x 的偏导数为: {x.grad.item()}")
print(f"在 x = {x.item()}, y = {y.item()} 处，z 对 y 的偏导数为: {y.grad.item()}")

#绘图x^2+y^2-1=0的图像
# 定义 x 和 y 范围
#linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
#start: 起始值，默认为 0。
#stop: 结束值，默认为 1。
#num: 生成的样本数量，默认为 50。
#endpoint: 是否包含 stop，默认为 True。
#retstep: 是否返回步长，默认为 False。
#dtype: 输出数组的数据类型，默认为 None。
#axis: 沿着哪个轴计算步长，默认为 0。0即为行，1为列

# # 使用默认参数（retstep=False）
# arr = np.linspace(0, 10, 5)
# print("Array without step:", arr)
# # 设置 retstep=True
# arr, step = np.linspace(0, 10, 5, retstep=True)
# print("Array with step:", arr)
# print("Step size:", step)
# 在 NumPy 中，linspace 函数返回的本身就是一个数组（NumPy 的 ndarray 对象）。当你设置 retstep=True 时，linspace 会返回一个元组，其中包含两个元素：第一个是生成的数值数组，第二个是步长值。

x_vals = np.linspace(-1.5, 1.5, 300)  # 缩小范围以更好地显示圆
y_vals = np.linspace(-1.5, 1.5, 300)

# np.arange 的输出长度取决于起始值、停止值和步长。如果步长不能整除起始值和停止值之间的差，则最后一个元素可能会略低于停止值。
# np.linspace 的输出长度始终等于指定的样本数量 num，且样本在起始值和停止值之间均匀分布。
# torch.arange相当于np.arange()
# torch.linspace相当于np.linspace()
# tensor = torch.linspace(0, 10, 5)
# print(tensor)
# 输出：tensor([0., 2.5, 5., 7.5, 10.])
# numpy = np.linspace(0, 10, 5)
# print(numpy)
# 输出：[ 0.   2.5  5.   7.5 10. ]
#相比两个输出结果
# 观察到np.linspace的输出看起来“不规则”主要是因为NumPy数组在打印时默认使用了一种不同的对齐方式，特别是当数组中的元素是浮点数时。NumPy会尝试以一种在视觉上更易于解析的方式来对齐小数点，但这并不总是会导致每个数字都占据相同的空间量。
# PyTorch张量在打印时通常使用一种更统一的格式，其中每个元素都占据相同的空间量，无论其值如何。这就是为什么torch.linspace的输出看起来更加“规则”

# 使用 np.meshgrid 生成二维网格
X, Y = np.meshgrid(x_vals, y_vals)
#在Matplotlib中，库本身不直接提供一个专门的“隐函数绘图”功能，但你可以通过创建一个网格（meshgrid）并使用条件语句或逻辑来绘制满足隐函数条件的点。

# 计算网格点上的 Z 值
Z = func(X , Y)

# 绘制等高线图，只显示 z = 0 的等高线
plt.figure()
plt.contour(X, Y, Z, levels=[0], colors='r')  # 绘制 z = 0 的等高线，即 x^2 + y^2 = 1
#colors='r' 表示绘制的等高线为红色
#'b' 蓝色 'y' 黄色 'g' 绿色 'c' 青色'm' 品红色 'k' 黑色 'w' 白色
#coontour是等高线的意X是变量，Y是变量
#plot是绘制曲线的意思
# plt.xlim(-1.5, 1.5) # 设置 x 轴范围
# plt.ylim(-1.5, 1.5) # 设置 y 轴范围
plt.xlabel('x')
plt.ylabel('y')
plt.title('Circle plot of x^2 + y^2 = 1')
plt.axis('equal')  # 确保 x 和 y 轴的比例相同，以正确显示圆形
#axis是坐标轴的意思，axis('equal') 表示将 x 轴和 y 轴的比例设置为相等，以确保图形在 x 轴和 y 轴上的比例一致，从而更好地显示圆形。
plt.grid(True)  # 添加网格
plt.show()


# 2.4.3. 梯度
print("\n2.4.3. 梯度")

# 定义一个多元函数 f(x1, x2) = 3(x1)^2 + 5e^{x2}
def f(x):   # 定义一个多元函数
    # x 是一个包含两个元素的张量
    x1 = x[0]
    x2 = x[1]
    #torch中也有数学运算函数如：torch.exp() e开方函数   torch.log() 自然对数函数(以e为底)  torch.sqrt() 平方根函数 torch.pow(x1, x2) 开方函数 指数是x2      torch.pow(x, 1/3) 三次方根
    #计算log2(3)为 torch.log2(torch.tensor(3.0))相当于torch.log2(3.0) 这两种方式在功能上是等效的，都会返回相同的结果。第二种方式（直接传递标量）在语法上更简洁，因为它避免了显式创建张量的步骤。
    #torch.exp() 函数用于计算自然指数（e）的幂，即 e^x。
    #(torch.e + 1) ** 2   #计算e+1的2次方
    #调用常数(pytorch)
    #e:torch.e
    #pi:torch.pi
    #调用常数(math)
    #e:math.e
    #pi:math.pi
    return 3 * x1 ** 2 + 5 * torch.exp(x2)

# 使用 PyTorch 计算梯度
x = torch.tensor([1.0, 0.0], requires_grad=True)
y = f(x)
y.backward()

print("函数：f(x1, x2) = 3(x1)^2 + 5e^(x2)")
print(f"函数 f(x1, x2) = 3(x1)^2 + 5e^(x2) 在 x1 = {x[0].item()}, x2 = {x[1].item()} 处的梯度为: {x.grad}")

#绘图
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_vals, y_vals)
X_tensor = torch.from_numpy(X)  #等价为X_tensor = torch.tensor(X, dtype=torch.float32)。但是这样效率低
Y_tensor = torch.from_numpy(Y)
#numpy转换为张量(tensor)用.from_numpy(<numpy>)函数，张量转换为numpy数组用<tensor>.numpy()函数
# torch.from_numpy() 函数被用来将 numpy 数组转换为 PyTorch 张量，torch.stack() 函数则用来将两个张量堆叠在一起。注意，dim=0 表示在第一个维度（即批处理维度）上堆叠张量。
Z = f(torch.stack((X_tensor, Y_tensor), dim=0))
#将 X 和 Y（它们是 numpy 数组）转换为一个 PyTorch 张量并传递给 f 函数
# Z = f(torch.tensor([X, Y], dtype=torch.float32))

# Z = f(torch.from_numpy(X).unsqueeze(1))
# 在这个例子中，unsqueeze(1) 是用来增加一个新的维度，使得 X 可以被视为一个网格。
# 因此，具体如何转换取决于 f 函数的设计和你的具体需求。

plt.figure()
plt.contour(X, Y, Z, colors='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour plot of f(x1, x2) = 3(x1)^2 + 5e^(x2)')
plt.axis('equal')   # 确保 x 和 y 轴的比例相同
plt.grid(True)
plt.show()

# 2.4.4. 链式法则
print("\n2.4.4. 链式法则")

# 定义复合函数 y = f(u), u = g(x)
# 设 f(u) = u^2, g(x) = 2x + 1
# 即（2x + 1)^2 = 4x^2 + 4x + 1
def g(x):
    return 2 * x + 1

def f(u):
    return u ** 2

# 使用 PyTorch 计算复合函数的导数
x = torch.tensor(1.0, requires_grad=True)
u = g(x)    #内层函数
y = f(u)    #外层函数
y.backward()

print(f"复合函数 y = [2x + 1]^2 在 x = {x.item()} 处的导数为: {x.grad.item()}")

# 手动计算链式法则的结果
# dy/dx = dy/du * du/dx = 2u * 2 = 4u
# 当 x = 1 时，u = 2 * 1 + 1 = 3，因此 dy/dx = 4 * 3 = 12
u_value = u.detach().item() # 分离 u，使其不再是计算图的一部分，使其不再参与梯度计算
manual_grad = 4 * u_value   #u_value=3=2x1+1=3
print(f"手动计算的导数值为: {manual_grad}")

# 2.4.5. 小结
print("\n2.4.5. 小结")

# 梳理一下本章用到的函数和方法：
# 1. torch.tensor(data, requires_grad=True): 创建一个可计算梯度的张量
#    - data（必选）：数据，可以是数值、列表、数组等
#    - requires_grad（可选）：是否需要计算梯度，默认为 False
# 2. tensor.backward(): 对张量进行反向传播，计算梯度
#    - grad_tensors（可选）：外部梯度
# 3. tensor.grad: 获取张量的梯度
# 4. torch.exp(tensor): 计算张量的指数
# 5. 绘图函数 plt.plot(): 绘制函数曲线
#    - x（必选）：x 轴数据
#    - y（必选）：y 轴数据
#    - format_string（可选）：控制曲线的格式，如颜色、线型等
# 6. autograd 自动微分机制：PyTorch 中用于自动计算梯度的模块

# 本章中，我们使用 PyTorch 的自动微分功能，计算了标量函数和多元函数的导数和梯度。
# 我们还展示了如何使用链式法则计算复合函数的导数。

# 最后，使用 Matplotlib 库绘制了函数及其切线的图像，帮助理解导数的几何意义。
