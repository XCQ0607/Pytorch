###########################################
# 5.5. 读写文件
###########################################
print("5.5. 读写文件")

# 本代码示例将比上述文档中的示例更加复杂、深入。
# 我们将展示如何使用PyTorch的save和load函数来读写张量、模型参数，甚至部分参数字典。
# 同时，我们将引入更多的控制台输出，以便清晰了解代码执行过程。
# 此外，我们还会在代码中展示如何用不同参数来调用这些函数，并回答文档中的问题。

# 环境准备与导入
#-----------------------------------------
import torch
from torch import nn
from torch.nn import functional as F
import os

###########################################
# 目录与子目录展示
###########################################
# 假设我们在当前工作目录下进行操作
# 使用os模块打印当前工作目录，并列出文件，以便清晰了解存储情况
print("\n当前工作目录:", os.getcwd())
print("当前目录下文件列表:", os.listdir('.'))


###########################################
# 5.5.1. 加载和保存张量的基本用法
###########################################
print("\n-------------------------------------")
print("5.5.1. 加载和保存张量示例")
print("-------------------------------------")

# 创建一个简单的张量并保存到文件中
x = torch.arange(4)    #创建一个张量,tensor([0, 1, 2, 3])
# torch.save函数示例:
# torch.save(obj, f, *, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)
# 参数说明：
# obj: 要保存的对象（如张量、字典、列表、模型状态字典等）
# f: 文件名(字符串)或文件对象
# pickle_module: 可选，默认为pickle，用于序列化的模块
# pickle_protocol: pickle协议版本
# _use_new_zipfile_serialization: 是否使用新的压缩文件格式序列化（默认True）
torch.save(x, './data/x-file.pt')
print("已将张量 x 保存至 './data/x-file.pt' 文件")
#保存在当前目录的data目录下


# 加载张量
# torch.load函数示例:
# torch.load(f, map_location=None, pickle_module=pickle, **pickle_load_args)
# 参数说明：
# f: 文件名(字符串)或文件对象
# map_location: 可选，用于将存储在特定设备上的张量映射到另一设备，如 map_location='cpu'
# pickle_module: 可选，序列化加载模块
x2 = torch.load('./data/x-file.pt', map_location=torch.device('cpu'), weights_only=True)
print("从文件中加载的张量 x2:", x2)


###########################################
# 保存和加载多种数据结构
###########################################
print("\n-------------------------------------")
print("保存和加载多个张量的示例")
print("-------------------------------------")

y = torch.zeros(4)    #创建一个张量,tensor([0., 0., 0., 0.])
torch.save([x, y], './data/x-files.pt')
x2_list, y2_list = torch.load('./data/x-files.pt',pickle_module=None)    # 从文件中加载的张量 x2: tensor([0., 0., 0., 0.])
#pickle_module=None 表示不使用pickle模块，直接加载张量
print("从文件中加载的列表 [x2_list, y2_list]:", x2_list, y2_list)

print("\n-------------------------------------")
print("保存和加载字典对象的示例")
print("-------------------------------------")

mydict = {'x': x, 'y': y}
torch.save(mydict, './data/mydict.pt')
mydict2 = torch.load('./data/mydict.pt')
print("从文件中加载的字典 mydict2:", mydict2)


###########################################
# 5.5.2. 加载和保存模型参数
###########################################
print("\n-------------------------------------")
print("5.5.2. 加载和保存模型参数示例")
print("-------------------------------------")

# 定义一个多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 继承自nn.Module, 继承父类的属性和方法
        self.hidden = nn.Linear(20, 256)    # 隐藏层
        self.output = nn.Linear(256, 10)    # 输出层
        #Linear层的权重和偏置参数是随机初始化的(默认是正态分布,均值为0,方差为1)
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

# 初始化模型并进行一次前向传播
net = MLP()
X = torch.randn(size=(2, 20))    # 随机生成一个2x20的张量
Y = net(X)
print("模型输出 Y:", Y)

# 保存模型参数（state_dict是一个包含模型所有参数的字典）
# state_dict通常包括weight, bias等参数项
torch.save(net.state_dict(), './data/mlp.params')

print("已将模型参数保存至 'mlp.params' 文件")


###########################################
# 从文件中加载模型参数
###########################################
print("\n-------------------------------------")
print("从文件中加载模型参数示例")
print("-------------------------------------")

clone = MLP()    # 创建一个新的MLP实例
clone.load_state_dict(torch.load('./data/mlp.params'))    # 从文件中加载模型参数到模型中
clone.eval()    # 设置为评估模式（可选）

Y_clone = clone(X)  # 使用加载参数的clone模型进行前向传播
print("利用加载参数的clone模型输出 Y_clone:", Y_clone)
print("Y_clone与Y是否相等:", Y_clone.eq(Y))

# 注：我们在这里对比Y与Y_clone，相同输入下参数相同则输出应一致。


###########################################
# 实际应用问题演示
###########################################
print("\n-------------------------------------")
print("实际应用问题演示")
print("-------------------------------------")

# 1. 即使不将训练好的模型部署到不同的设备，保存模型参数也有实际好处：
#    比如长时间训练后，为了防止断电或中断，可以定期保存模型参数，以便在中途意外停止后可以恢复训练，
#    不用从头开始。这节省了大量计算资源和时间。
#    在下面的例子中，我们假设训练了很久后保存中间参数，然后再载入进行继续训练或推理。
intermediate_params_file = './data/mlp_intermediate.params' # 假设这是中间状态文件
torch.save(net.state_dict(), intermediate_params_file)
print("已将中间训练状态参数保存至:", intermediate_params_file)

# 在实际运行中如果中断，这里可以使用如下代码恢复：
net_recovery = MLP()
net_recovery.load_state_dict(torch.load(intermediate_params_file))
print("从中间状态文件中恢复模型参数成功。")


# 2. 假设我们只想复用网络的一部分，将其合并到不同的网络架构中：
#    比如在一个新的模型中使用之前网络的前两层（这里我们就简单复用Linear层参数）
#    我们可以从state_dict中选择性的加载参数。

print("\n-------------------------------------")
print("部分参数复用示例")
print("-------------------------------------")

# 新的网络架构（假设我们想用旧模型的hidden层参数，但修改output层结构）
class NewMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)   # 与原模型相同结构的层
        self.new_output = nn.Linear(256, 5) # 改变输出维度为5
    def forward(self, x):
        return self.new_output(F.relu(self.hidden(x)))

new_net = NewMLP()
old_state_dict = torch.load('./data/mlp.params')

# 我们只拷贝hidden层的参数
partial_dict = {k: v for k, v in old_state_dict.items() if "hidden" in k}   # 只提取hidden层的参数
print("从旧模型参数中提取的子字典:", partial_dict.keys())
# 字典推导式（dictionary comprehension）
# old_state_dict.items() 遍历 old_state_dict 字典中的所有项（键值对）。
# 对于 old_state_dict 中的每一项（即每一对键值对），代码检查键（k）是否包含字符串 "hidden"。
# 如果键包含 "hidden"，则将该键值对（k: v）添加到新字典 partial_dict 中。
# 最终，partial_dict 只包含原字典中键含有 "hidden" 的那些项。

# 使用load_state_dict的strict=False来允许只加载部分参数
new_net.load_state_dict(partial_dict, strict=False) #strict=False参数允许状态字典中的键与模型参数不完全匹配
print("已将旧模型的hidden层参数加载到新模型中")
X_new = torch.randn(size=(2, 20))
Y_new = new_net(X_new)
print("新模型输出 Y_new:", Y_new)


# 3. 如何同时保存网络架构和参数？
#    若要同时保存网络架构与参数，需要对架构加上限制，通常做法是：
#    - 将模型的定义代码（如类定义）保存在一个独立的Python文件中。
#    - 保存参数时，仍使用torch.save(net.state_dict(), 'xxx.params')。
#    - 加载时，先导入定义模型架构的代码，再实例化相同架构的模型对象，再调用load_state_dict()。
#
#    如果想“一键”保存模型和结构，可以考虑使用 `torch.save(net, 'model.pth')` 但这要求：
#    模型类定义必须在加载代码中可用（包括相同的类名、相同的代码结构）；
#    否则会报错，因为pickle无法加载未知结构的类对象。
#
#    示例（请注意这种方式有局限性，不一定适用于所有场景）：
print("\n-------------------------------------")
print("保存整个模型(架构+参数)的示例（需谨慎使用）")
print("-------------------------------------")

# torch.save可以直接保存模型对象，但必须在加载时有相同的类定义可用
torch.save(net, './data/full_model.pth')
print("已保存整个模型(包含架构和参数)到 'full_model.pth'")

# 加载此模型
loaded_net = torch.load('./data/full_model.pth')    #前提: 加载时必须有相同的类定义可用
#也就是这里的MLP类定义必须可用
#Class MLP(nn.Module):
#    def __init__(self):
#       ...
#    def forward(self, x):
#     ...

print("已从 'full_model.pth' 中加载整个模型架构和参数")
Y_loaded = loaded_net(X)
print("加载整个模型后，模型输出与之前一致:", Y_loaded)


###########################################
# 总结与参数说明
###########################################

# 在本代码示例中，我们使用了以下函数：
# 1. torch.save(obj, f, ...)
#    - 必选参数：
#      obj: 要保存的Python对象（张量、状态字典、模型等）
#      f: 字符串文件名或文件对象
#    - 可选参数：
#      pickle_module (默认pickle)
#      pickle_protocol (pickling协议版本)
#      _use_new_zipfile_serialization (bool，默认为True，使用新的zipfile格式)
#
#    用途：将PyTorch对象序列化并写入到磁盘文件中，以便日后加载。
#
# 2. torch.load(f, ...)
#    - 必选参数：
#      f: 字符串文件名或文件对象
#    - 可选参数：
#      map_location: 指定加载到的设备，如'cpu'或{'cuda:0':'cpu'}
#      pickle_module (默认pickle)
#    用途：从磁盘文件中反序列化读取对象（张量、状态字典、模型等）。
#
# 此外我们还展示了：
# - 如何保存和加载单个张量、多张量列表、字典。
# - 如何保存和加载模型参数(state_dict)。
# - 如何加载部分参数到新网络中，方便参数复用。
# - 如何保存模型架构与参数（需保证架构定义可用）。
#
# 问题回答：
# 1. 即使不需要将经过训练的模型部署到不同设备上，存储模型参数的好处：
#    - 防止在长时间训练中意外中断后丢失进度，可以从中间状态恢复继续训练。
#    - 可用于模型版本控制，便于后续分析对比不同版本参数的性能。
#
# 2. 如果只想复用网络的一部分，可以使用加载旧模型state_dict后，根据key筛选需要的部分参数，使用strict=False加载到新模型。这允许新的模型架构中参数字典与旧模型的参数字典不完全匹配，从而只更新所需的部分。
#
# 3. 同时保存架构和参数的方法：
#    - 最佳实践仍是保存参数，再加载时定义好相同架构并导入参数。
#    - 如果必须一起保存，可用torch.save(model, 'xxx.pth')，但必须在加载时提供相同的模型类定义。
#    - 限制是架构必须可用(已定义)，且模型类结构代码不变，否则无法正常加载。

print("\n-------------------------------------")
print("代码执行完毕，已演示所有功能和问题回答。")
print("-------------------------------------")

# .pth文件通常用于存储完整的模型状态，包括模型的结构和参数。它允许你直接加载一个完整的模型实例。
# .params文件（尽管这不是PyTorch官方推荐的文件扩展名，但在这个上下文中我们假设它用于存储状态字典）通常只包含模型的参数，而不包含模型的结构信息。你需要先创建一个模型实例，然后将这些参数加载到这个实例上。


# 使用torch.load直接加载模型实例：
# new_net = torch.load('./data/mlp.pth')
# 这种方法会加载一个完整的模型实例，包括模型的结构和参数。这要求加载时环境中存在与保存模型时相同的类定义。.pth文件通常用于存储完整的模型状态，包括模型的结构和参数。
#
# 加载状态字典并应用到新模型实例：
# class NewMLP():
#     def __init__(self):
#         # 初始化模型结构
#         pass
#
# new_net = NewMLP()
# old_state_dict = torch.load('./data/mlp.params')
# new_net.load_state_dict(old_state_dict, strict=False)
#
# 这两种方法并不完全等效。使用torch.load直接加载.pth文件会得到一个完整的模型实例，而加载.params文件（假设它包含状态字典）并应用到新模型实例上则要求你先定义模型结构。如果.params文件实际上包含了完整的模型状态（这很少见，因为通常我们会区分状态字典和完整模型状态），并且你的环境中有正确的类定义，那么理论上你可以通过第二种方法得到一个与第一种方法相同的模型实例。但在实践中，.params更可能只包含参数，因此你需要确保NewMLP类的定义与保存模型时的类定义兼容。