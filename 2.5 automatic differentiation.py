print("2.5. 自动微分")

# 导入PyTorch库
import torch

# ==========================
# 1. 创建需要求导的张量
# ==========================

# 创建一个张量 x，要求对其计算梯度
# requires_grad=True 表示需要对 x 计算梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print("初始张量 x:", x)

# ==========================
# 2. 定义复杂的函数 y
# ==========================

# 定义一个复杂的函数 y，包括多种张量操作
# y = x^3 + 2*x^2 + x
y = x**3 + 2 * x**2 + x
print("函数 y = x^3 + 2*x^2 + x:", y)

# ==========================
# 3. 计算标量输出 scalar_y
# ==========================

# 将 y 的所有元素求和，得到一个标量 scalar_y
scalar_y = y.sum()
print("标量输出 scalar_y (y 的和):", scalar_y)

# ==========================
# 4. 进行反向传播，计算梯度
# ==========================

# 对 scalar_y 进行反向传播，计算 x 的梯度
# 由于我们计划多次调用 backward()，需要设置 retain_graph=True 保留计算图
scalar_y.backward(retain_graph=True)
print("第一次反向传播后 x 的梯度 x.grad:", x.grad)

# ==========================
# 5. 多次反向传播的情况
# ==========================

# 再次对 scalar_y 进行反向传播
# 如果不清零梯度，新的梯度会累加到原来的梯度上
scalar_y.backward(retain_graph=True)
print("第二次反向传播后 x 的梯度 x.grad:", x.grad)

# 为了避免梯度累加，在每次反向传播前清零梯度
x.grad.zero_()
scalar_y.backward(retain_graph=True)
print("清零梯度后再次反向传播 x.grad:", x.grad)


# 错误原因： 当您第一次调用 scalar_y.backward() 时，PyTorch 会默认释放计算图。如果您随后再次调用 backward()，会因为计算图已经被释放而报错。
# 解决方法： 在第一次调用 backward() 时，设置 retain_graph=True，以保留计算图，允许多次反向传播。
# 注意事项：
#
# retain_graph=True： 在需要对同一个计算图进行多次反向传播时，需要保留计算图。
# 梯度累加： 在每次反向传播前，最好使用 x.grad.zero_() 将梯度清零，避免梯度累加导致结果不符合预期。


# ==========================
# 6. 计算二阶导数
# ==========================

# 为了计算二阶导数，需要对一阶梯度再次求导
# 首先，保留计算图，并获取一阶梯度
x.grad.zero_()
scalar_y.backward(create_graph=True)
first_order_grad = x.grad.clone()
print("一阶梯度 first_order_grad:", first_order_grad)

# 对一阶梯度求和，得到一个标量，然后对 x 进行反向传播，计算二阶导数
# 由于一阶梯度是一个向量，需要将其与一个同形状的张量相乘后求和，得到标量
grad_outputs = torch.ones_like(x)
first_order_grad.backward(grad_outputs)
second_order_grad = x.grad.clone()
print("二阶导数 second_order_grad:", second_order_grad)

# ==========================
# 7. 分离计算图
# ==========================

# 有时需要将某些计算从计算图中分离出来，使用 detach() 方法
x.grad.zero_()
y_detached = y.detach()  # detach() 返回一个新的张量，与 y 有相同的数据，但无梯度关系
z = y_detached * x
scalar_z = z.sum()
scalar_z.backward()
print("使用 detach() 后的梯度 x.grad:", x.grad)

# 注意：由于 y_detached 与 x 无梯度关系，因此 z 对 x 的梯度只来自于 x，而不包括 y 对 x 的梯度

# ==========================
# 8. 控制流中的梯度计算
# ==========================

# 定义一个包含控制流的函数
def control_flow(x):
    y = x
    for _ in range(5):
        if y.norm() < 1000:
            y = y * 2
    if y.sum() > 0:
        out = y
    else:
        out = 100 * y
    return out

# 创建一个标量张量 a，要求梯度
a = torch.tensor([2.0], requires_grad=True)
print("控制流输入 a:", a)

# 计算控制流函数的输出
b = control_flow(a)
print("控制流输出 b:", b)

# 对输出 b 进行反向传播，计算 a 的梯度
b.backward()
print("控制流中 a 的梯度 a.grad:", a.grad)

# 如果将 a 改为随机向量或矩阵
a = torch.randn((2, 2), requires_grad=True)
print("新的控制流输入 a:", a)
b = control_flow(a)
print("新的控制流输出 b:", b)
b.backward(torch.ones_like(a))
print("新的控制流中 a 的梯度 a.grad:", a.grad)

# ==========================
# 9. 梳理本章用到的函数
# ==========================

# 本章用到的主要函数和方法：
# - torch.tensor(data, requires_grad=True): 创建一个需要梯度的张量
# - 张量操作（如加法、乘法、幂等）：用于构建计算图
# - backward(retain_graph=False, create_graph=False, grad_tensors=None): 反向传播，计算梯度
#   - retain_graph（可选）：是否保留计算图，默认 False
#   - create_graph（可选）：是否构建用于高阶导数的计算图，默认 False
#   - grad_tensors（可选）：外部梯度张量，与张量形状相同，默认 None
# - zero_(): 将张量的数据清零，常用于梯度清零
# - clone(): 复制张量，生成与原始张量数据相同但不共享的张量
# - detach(): 返回一个新的张量，从当前计算图中分离
# - torch.ones_like(tensor): 生成一个与指定张量形状相同的全 1 张量
# - requires_grad_(True/False): 原地修改张量的 requires_grad 属性

