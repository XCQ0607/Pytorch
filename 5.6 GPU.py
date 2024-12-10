# ============================================================
# 5.6. GPU
# ============================================================
print("5.6. GPU")

# ------------------------------------------------------------
# 本示例将使用 PyTorch 来演示在 CPU 和 GPU 上进行计算、存储张量、
# 模型的参数存储与读取、多 GPU 计算加速、以及对比 CPU/GPU 性能的差异。
#
# 本代码会分阶段逐步展示以下内容：
# 1. 查询可用的设备 (CPU / GPU)
# 2. 张量在 CPU 与 GPU 之间的存储与移动
# 3. 基于 GPU 的简单矩阵操作与性能测量（大规模与小规模）
# 4. 模型参数在 GPU 上的初始化与存储、读取
# 5. 同时使用多个 GPU 来并行进行运算并测量性能
#
# 此外，在代码中会用 print 分割线明确区分不同的演示阶段。
#
# 在注释中将介绍一些函数及常用参数:
#
# torch相关常用函数与参数介绍：
# 1. torch.device(type, index=0)
#    参数介绍：
#      - type: 字符串，"cpu" 或 "cuda"。
#      - index: 整数型，代表第几个GPU（如有多个）。默认为0，即第一块GPU。
#    功能：
#      用于指定一个设备对象，将张量或模型放置到该设备上。
#
# 2. torch.cuda.is_available()
#    功能：
#      返回布尔值，表示是否可以使用GPU(CUDA)。
#
# 3. tensor.to(device)
#    参数介绍：
#      - device: torch.device类型或者字符串("cpu"/"cuda")，将此张量拷贝或移动到该设备。
#    功能：
#      将张量复制/移动到指定设备中。如果本身已经在这个设备上，将不做额外拷贝。
#
# 4. model.to(device)
#    参数介绍：
#      - device: 同上，将模型的所有参数移动到指定设备中。
#    功能：
#      将模型中的所有参数(权重、偏置)移动到指定设备。
#
# 5. torch.randn(size, device=None)
#    参数介绍：
#      - size: 元组，指定生成张量的形状，如(2, 3)代表2行3列。
#      - device: 同上，将生成的张量直接创建在该设备上。
#    功能：
#      生成一个服从标准正态分布N(0,1)的随机张量。
#
# 6. torch.ones(size, device=None)
#    功能同上，生成全为1的张量。
#
# 7. 张量操作(如加法)需要在同一设备上进行，否则会报错。
#
# 8. 模型初始化:
#    我们以一个简单的线性层为例(torch.nn.Linear)：
#    nn.Linear(in_features, out_features)
#    参数介绍：
#      - in_features: 输入特征数
#      - out_features: 输出特征数
#    功能：
#      定义一个线性层的可训练参数 W 和 b，通常在CPU上初始化，通过model.to(device)移动。
#
# 9. 保存与加载模型参数:
#    torch.save(obj, path) 和 torch.load(path)
#    参数介绍：
#      - obj: 要保存的对象（如model的state_dict()）
#      - path: 文件路径字符串
#    功能：
#      保存和加载模型参数字典。可在CPU或GPU上操作，但注意加载后需手动to(device)。
#
# 在最后，我们会在注释中总结本代码用到的函数与参数。
#
# 完成文档中的部分练习：
# - 测试CPU与GPU对大矩阵乘法的加速比。
# - 在GPU上对模型进行读写操作。
# - 对1000个100x100矩阵乘法计时并记录结果。
# - 尝试同时在两个GPU进行计算与在单个GPU上顺序计算的耗时比较。
#
# 注意：本代码假设运行环境有可用GPU。例如在Google Colab中执行。
# 如果本地无GPU，请自行根据条件调试。
#
# ------------------------------------------------------------
import torch
import torch.nn as nn
import time

# ------------------------------------------------------------
print("\n" + "="*60)
print("演示1: 查询可用设备与定义辅助函数")
print("="*60)

def try_gpu(i=0):
    """如果存在第i个GPU，则返回torch.device('cuda', i)，否则返回cpu()。
    参数：
      i(int): 指定第i个GPU,从0开始。如i=1即代表第二块GPU。
    返回值：
      torch.device对象，表示可用设备。
    """
    if torch.cuda.is_available() and i < torch.cuda.device_count():
            return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU列表，如果没有GPU，则返回[torch.device('cpu')]。
    返回值：
      list[torch.device]: 所有可用GPU的列表，如[device('cuda:0'), device('cuda:1'), ...]，若无GPU则[device('cpu')]。
    """
    if torch.cuda.is_available():
        return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return [torch.device('cpu')]

cpu_device = torch.device('cpu')    # 定义CPU设备
gpu0 = try_gpu(0)
gpu1 = try_gpu(1)
all_gpus = try_all_gpus()

print("CPU设备:", cpu_device)
print("GPU设备0:", gpu0)
print("GPU设备1(如有):", gpu1)
print("所有可用的GPU设备:", all_gpus)

# ------------------------------------------------------------
print("\n" + "="*60)
print("演示2: 张量在CPU和GPU上的创建与移动")
print("="*60)

# 在CPU上创建张量
x_cpu = torch.tensor([1, 2, 3])
print("x_cpu:", x_cpu, "所在设备:", x_cpu.device)

# 在第一个GPU上创建张量
x_gpu0 = torch.ones((2,3), device=gpu0)
print("x_gpu0:", x_gpu0, "所在设备:", x_gpu0.device)

# 若有第二个GPU，则在上面创建一个随机张量
if torch.cuda.device_count() > 1:
    x_gpu1 = torch.rand((2,3), device=gpu1)
    print("x_gpu1:", x_gpu1, "所在设备:", x_gpu1.device)

# 将CPU张量拷贝到gpu0
x_cpu_to_gpu0 = x_cpu.to(gpu0)
print("x_cpu_to_gpu0:", x_cpu_to_gpu0, "所在设备:", x_cpu_to_gpu0.device)

# 如有GPU1，将gpu0上的张量移动到gpu1
if torch.cuda.device_count() > 1:
    x_gpu0_to_gpu1 = x_gpu0.to(gpu1)
    print("x_gpu0_to_gpu1:", x_gpu0_to_gpu1, "所在设备:", x_gpu0_to_gpu1.device)

# ------------------------------------------------------------
print("\n" + "="*60)
print("演示3: 张量计算——确保在同一设备上操作")
print("="*60)

# 如果我们要对两个张量相加，他们需要在同一个设备上
if torch.cuda.device_count() > 1:
    # 在gpu1上进行相加
    sum_result = x_gpu0_to_gpu1 + x_gpu1
    print("在gpu1上对张量相加结果:", sum_result)
else:
    # 若只有cpu，直接在cpu上加
    sum_result = x_cpu + x_cpu
    print("在CPU上对张量相加结果:", sum_result)

# ------------------------------------------------------------
print("\n" + "="*60)
print("演示4: CPU vs GPU 性能测试 (大矩阵乘法)")
print("="*60)

# 定义一个函数来进行大矩阵乘法，并测量用时
def benchmark_matrix_multiplication(device, num_trials=10, matrix_size=(1000,1000)):
    """
    在给定设备上进行多次大矩阵乘法，并返回平均耗时。
    参数：
      device: torch.device 设备
      num_trials(int): 重复试验次数
      matrix_size(tuple): 矩阵大小，如(1000,1000)
    返回值：
      平均耗时(秒)
    """
    A = torch.randn(matrix_size, device=device)
    B = torch.randn(matrix_size, device=device)
    # 预热(让GPU先运行一次不计时，避免初次调用慢)
    _ = torch.mm(A, B)
    torch.cuda.synchronize() if device.type == 'cuda' else None # 确保所有GPU操作完成后再计时，这里的device是torch.device类型
    #torch.cuda.synchronize()是一个PyTorch函数，它的作用是阻塞当前线程，直到所有之前的CUDA操作完成。这在某些情况下非常有用，比如当你想要确保所有的GPU操作都已完成，然后再进行下一步操作（比如计时或进行下一步计算）时。

    start = time.time()
    for _ in range(num_trials):
        C = torch.mm(A, B)
    # 等待GPU运算完成再计时（如果是GPU）
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()
    avg_time = (end - start) / num_trials
    return avg_time

# 测试在CPU上和GPU上进行1000x1000矩阵乘法的耗时
cpu_time = benchmark_matrix_multiplication(cpu_device)
print(f"CPU上进行1000x1000矩阵乘法平均耗时: {cpu_time:.6f} 秒")

if torch.cuda.is_available():
    gpu_time = benchmark_matrix_multiplication(gpu0)
    print(f"GPU0上进行1000x1000矩阵乘法平均耗时: {gpu_time:.6f} 秒")

# ------------------------------------------------------------
print("\n" + "="*60)
print("演示5: 模型参数在GPU上的存储与读写")
print("="*60)

# 定义一个简单的线性模型
model = nn.Linear(100, 10)  # 100维输入 -> 10维输出
print("模型参数初始化设备:", next(model.parameters()).device)    #next(model.parameters())返回的是一个迭代器，迭代器中的第一个元素是模型的第一个参数，即权重矩阵W,第二个参数是偏置向量b。

# 将模型放到GPU上(如果有GPU)
model = model.to(gpu0 if torch.cuda.is_available() else cpu_device)  # 移动模型到GPU
print("模型已移动到:", next(model.parameters()).device)

# 模拟保存与加载模型参数
model_path = ".data/temp_model.pth"
torch.save(model.state_dict(), model_path)  # 保存参数到文件
print("模型参数已保存到文件:", model_path)

# 加载参数到CPU
model_loaded = nn.Linear(100, 10)
model_loaded.load_state_dict(torch.load(model_path, map_location='cpu'))    #map_location='cpu'表示将模型加载到CPU上
print("加载到CPU后的模型参数设备:", next(model_loaded.parameters()).device)

# 将加载后的模型移动回GPU(如可用)
if torch.cuda.is_available():
    model_loaded = model_loaded.to(gpu0)
    print("加载后的模型已移动到:", next(model_loaded.parameters()).device)

# ------------------------------------------------------------
print("\n" + "="*60)
print("演示6: 计算1000个100x100矩阵乘法的时间与Frobenius范数") #范数是第二范数
print("="*60)

def compute_1000_matmul_and_log(device):
    """
    在指定设备上计算1000个100x100矩阵与其自身的乘法。
    同时记录每次运算的结果矩阵Frobenius范数，并在计算过程中打印日志。
    为了避免频繁的CPU-GPU数据传输，我们先把Frobenius范数收集在GPU上，然后一次性打印。
    """
    # 创建数据
    X = torch.randn((100, 100), device=device)
    norms = []  # 用于存储范数的列表
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for i in range(1000):
        Y = torch.mm(X, X)
        # 直接在GPU上计算范数
        # Frobenius范数即 sqrt( sum of squares )，等价于Y.norm('fro')
        fnorm = Y.norm(p='fro')
        # 为减少IO开销和传输，暂不立即打印，先存下来
        norms.append(fnorm)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()

    # 现在统一转到CPU来打印（减少重复传输）
    norms_cpu = [n.item() for n in norms]  # 将GPU上的标量移动到CPU
    # 打印示例日志（仅打印前5个结果，避免过多输出）
    print("前5个乘法结果矩阵Frobenius范数:", norms_cpu[:5])
    print(f"在{device}设备上完成1000次100x100乘法总耗时: {end - start:.6f} 秒")

compute_1000_matmul_and_log(cpu_device)
if torch.cuda.is_available():
    compute_1000_matmul_and_log(gpu0)

# ------------------------------------------------------------
print("\n" + "="*60)
print("演示7: 同时在两个GPU上进行矩阵运算与在单GPU上顺序执行的对比(如有两个GPU)")
print("="*60)

if torch.cuda.device_count() > 1:
    # 在两个GPU上同时执行两个1000x1000矩阵乘法
    # 定义一个测试函数
    def matrix_mul_on_device(device):
        A = torch.randn((1000,1000), device=device)
        B = torch.randn((1000,1000), device=device)
        _ = torch.mm(A, B)  # 预热
        torch.cuda.synchronize()
        start = time.time()
        C = torch.mm(A, B)
        torch.cuda.synchronize()
        end = time.time()
        return end - start

    # 并行执行需要多线程或异步操作，但简化起见，我们用torch.cuda.Stream来演示一下简单的并发
    stream0 = torch.cuda.Stream(device=gpu0)
    stream1 = torch.cuda.Stream(device=gpu1)

    # 在stream0上运行GPU0的计算
    with torch.cuda.stream(stream0):
        time_gpu0 = matrix_mul_on_device(gpu0)

    # 在stream1上运行GPU1的计算
    with torch.cuda.stream(stream1):
        time_gpu1 = matrix_mul_on_device(gpu1)

    # 等待流执行完毕
    stream0.synchronize()
    stream1.synchronize()

    # 并行耗时不会简单相加，这里只是单次执行，一般应多次测量再比较平均值
    print(f"两个GPU并行执行单次1000x1000矩阵乘法的时间: GPU0={time_gpu0:.6f}s, GPU1={time_gpu1:.6f}s")

    # 对比在单个GPU上顺序执行两个矩阵乘法的耗时
    def sequential_two_mul_on_one_gpu(device):
        A = torch.randn((1000,1000), device=device)
        B = torch.randn((1000,1000), device=device)
        C = torch.mm(A, B)  # 第一次
        D = torch.mm(A, B)  # 第二次
        torch.cuda.synchronize()
        return

    start = time.time()
    sequential_two_mul_on_one_gpu(gpu0)
    end = time.time()
    seq_time = end - start
    print(f"在单GPU上顺序执行两次1000x1000矩阵乘法耗时: {seq_time:.6f}s")

# ------------------------------------------------------------
print("\n" + "="*60)
print("总结本代码示例中使用到的函数与参数")
print("="*60)
# 在本示例中使用到的函数与参数包括（仅列举重点）：
# 1. torch.device(type, index=0): 用于指定设备。type='cpu'或'cuda'，index指定GPU号。
# 2. torch.cuda.is_available(): 判断当前环境是否支持CUDA(GPU)。
# 3. 张量创建函数：torch.tensor, torch.randn, torch.ones等，可使用device参数指定存储设备。
# 4. 张量移动：tensor.to(device) 将张量移动到指定设备。若已在该设备上则不移动。
# 5. 模型移动：model.to(device) 将模型参数移动到指定设备。
# 6. 矩阵乘法：torch.mm(A,B) 计算两个矩阵的乘积，需保证A,B在同一设备。
# 7. 模型参数保存与加载：torch.save(model.state_dict(), path) 与 torch.load(path)。
# 8. 时间测量：time.time() 获取当前时间戳，用于计算耗时。
# 9. torch.cuda.synchronize(): 在使用GPU计时时，需调用此函数确保所有GPU操作完成。
# 10. torch.norm(p='fro'): 用于计算张量的Frobenius范数。
#
# 参数介绍已在代码注释处给出示例。调用示例如上文展示。
#
# 对文档中提出的练习进行了响应：
# - 尝试一个计算量更大的任务(1000x1000矩阵相乘)测试CPU/GPU速度差异。
# - 展示了如何在GPU上读写模型参数(存储到文件并加载)。
# - 计算了1000个100x100矩阵相乘的时间和Frobenius范数，并记录日志。
# - 尝试了在两个GPU上并行执行与在一个GPU上顺序执行的时间对比(如有两个GPU)。
#
# 整个示例演示了GPU在深度学习中的基本使用、数据移动和性能测量方法。
