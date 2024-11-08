# 2.4. 微积分
print("2.4. 微积分")

# 导入必要的库
import torch
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置默认字体为支持中文的字体（例如：SimHei黑体）
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他支持中文的字体

# 设置在Jupyter中显示图形为SVG格式
# 这可以使图形更清晰
from matplotlib_inline import backend_inline
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
x = torch.tensor(1.0, requires_grad=True)
y = 3 * x ** 2 - 4 * x
# 计算梯度
y.backward()
# x.grad 保存了 y 对 x 的导数值
print(f"在 x = {x.item()} 处，y 对 x 的导数值为: {x.grad.item()}")

# 绘制函数及其在 x=1 处的切线
print("\n绘制函数及其在 x=1 处的切线:")

# 定义 x 的范围
x_vals = np.arange(0, 3, 0.1)
# 计算函数值
y_vals = f(x_vals)
# 计算切线 y = 2x - 3
tangent_line = 2 * x_vals - 3

# 创建图形
plt.figure()
plt.plot(x_vals, y_vals, label='f(x) = 3x^2 - 4x')
plt.plot(x_vals, tangent_line, '--', label='切线 (x=1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()

# 2.4.2. 偏导数
print("\n2.4.2. 偏导数")

# 定义一个多元函数 z = x^2 + y^2
def func(x, y):
    return x ** 2 + y ** 2

# 使用 PyTorch 计算偏导数
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = func(x, y)

# 计算梯度
z.backward()

print(f"在 x = {x.item()}, y = {y.item()} 处，z 对 x 的偏导数为: {x.grad.item()}")
print(f"在 x = {x.item()}, y = {y.item()} 处，z 对 y 的偏导数为: {y.grad.item()}")

# 2.4.3. 梯度
print("\n2.4.3. 梯度")

# 定义一个多元函数 f(x1, x2) = 3x1^2 + 5e^{x2}
def f(x):
    # x 是一个包含两个元素的张量
    x1 = x[0]
    x2 = x[1]
    return 3 * x1 ** 2 + 5 * torch.exp(x2)

# 使用 PyTorch 计算梯度
x = torch.tensor([1.0, 0.0], requires_grad=True)
y = f(x)
y.backward()

print(f"函数 f(x1, x2) = 3x1^2 + 5e^x2 在 x1 = {x[0].item()}, x2 = {x[1].item()} 处的梯度为: {x.grad}")

# 2.4.4. 链式法则
print("\n2.4.4. 链式法则")

# 定义复合函数 y = f(u), u = g(x)
# 设 f(u) = u^2, g(x) = 2x + 1
def g(x):
    return 2 * x + 1

def f(u):
    return u ** 2

# 使用 PyTorch 计算复合函数的导数
x = torch.tensor(1.0, requires_grad=True)
u = g(x)
y = f(u)
y.backward()

print(f"复合函数 y = [2x + 1]^2 在 x = {x.item()} 处的导数为: {x.grad.item()}")

# 手动计算链式法则的结果
# dy/dx = dy/du * du/dx = 2u * 2 = 4u
# 当 x = 1 时，u = 2 * 1 + 1 = 3，因此 dy/dx = 4 * 3 = 12
u_value = u.detach().item()
manual_grad = 4 * u_value
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
