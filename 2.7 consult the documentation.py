import torch
import torch.nn as nn
import torch.optim as optim

print("2.7. 查阅文档")

# 2.7.1. 查找模块中的所有函数和类
print("\n2.7.1. 查找模块中的所有函数和类")
print("torch.nn模块中的所有属性:")
nn_attributes = dir(nn)
print(", ".join(attr for attr in nn_attributes if not attr.startswith("_")))

# 2.7.2. 查找特定函数和类的用法
print("\n2.7.2. 查找特定函数和类的用法")
print("torch.ones函数的帮助文档:")
help(torch.ones)

# 2.7.3. 创建张量的示例
print("\n2.7.3. 创建张量的示例")

# torch.ones示例
print("torch.ones函数示例:")
ones_tensor = torch.ones(3, 4, dtype=torch.float32)
print(ones_tensor)

# torch.zeros示例
print("\ntorch.zeros函数示例:")
zeros_tensor = torch.zeros(2, 3, 5)
print(zeros_tensor)

# torch.randn示例
print("\ntorch.randn函数示例:")
random_tensor = torch.randn(2, 3)
print(random_tensor)

# 2.7.4. 张量操作示例
print("\n2.7.4. 张量操作示例")

# 张量加法
print("张量加法示例:")
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.add(a, b)
print(f"a + b = {c}")

# 张量乘法
print("\n张量乘法示例:")
matrix1 = torch.randn(2, 3)
matrix2 = torch.randn(3, 2)
result = torch.mm(matrix1, matrix2)
print(f"矩阵乘法结果:\n{result}")

# 2.7.5. 神经网络模块示例
print("\n2.7.5. 神经网络模块示例")


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = SimpleNet()
print("简单神经网络结构:")
print(net)

# 2.7.6. 优化器示例
print("\n2.7.6. 优化器示例")
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
print("SGD优化器参数:")
print(optimizer.state_dict())

# 2.7.7. 损失函数示例
print("\n2.7.7. 损失函数示例")
loss_fn = nn.CrossEntropyLoss()
input = torch.randn(3, 2)
target = torch.empty(3, dtype=torch.long).random_(2)
loss = loss_fn(input, target)
print(f"输入: {input}")
print(f"目标: {target}")
print(f"损失: {loss.item()}")

"""
本章使用的主要函数和类:
1. dir(): 列出模块的所有属性
2. help(): 显示函数或类的帮助文档
3. torch.ones(): 创建全1张量
4. torch.zeros(): 创建全0张量
5. torch.randn(): 创建随机正态分布张量
6. torch.add(): 张量加法
7. torch.mm(): 矩阵乘法
8. nn.Module: 神经网络模块基类
9. nn.Linear: 线性层
10. optim.SGD: 随机梯度下降优化器
11. nn.CrossEntropyLoss: 交叉熵损失函数
"""
