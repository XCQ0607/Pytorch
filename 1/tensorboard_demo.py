import torch
import torch.nn as nn   #构建神经网络模型
import numpy as np          # 用于生成随机数据
from torch.utils.tensorboard import SummaryWriter   # 用于生成TensorBoard日志文件
import datetime # 用于生成当前时间
import torchvision  # 用于加载数据集
from torchvision import transforms  # 用于数据预处理
import matplotlib.pyplot as plt   # 用于绘制图像

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 创建 SummaryWriter 实例
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')    #.strftime() 格式化日期时间为字符串
writer = SummaryWriter(log_dir=f'runs/demo2_{current_time}')   # 实例化SummaryWriter类，参数为日志文件保存路径
#SummaryWriter传入参数有：1. log_dir(str) 日志文件保存路径, 默认为runs/。2. 注释(str)，默认为空。3. flush_secs(int)，默认为10。4. max_queue(int)，默认为10。5. filename_suffix(str)，默认为空。

# 1. add_scalar: 记录标量值
for i in range(100):
    #add_scalar(tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False)
    # tag(str) 标量的名称（title），全局步长(global_step)(x轴，训练次数），标量值(scalar_value)(y轴，值)
    # 可选参数：walltime(float)，new_style(bool)，double_precision(bool),walltime是时间戳，new_style(bool)是新的样式，double_precision(bool)是双精度
    #add_scalr是一个函数，用于记录标量值
    writer.add_scalar('Loss/train', torch.rand(1), global_step=i)
    writer.add_scalar('Accuracy/train', torch.rand(1) * 100, global_step=i)

# 2. add_scalars: 同时记录多个相关的标量值
for i in range(100):
    writer.add_scalars('Loss', {
        # train是训练集，val是验证集，test是测试集
        'train': torch.rand(1), #torch.rand是生成随机数，参数为形状，这里是1000个随机数
        'val': torch.rand(1),
        'test': torch.rand(1)
    }, global_step=i)
#global_step是全局步长，定义它的目的是为了让TensorBoard知道哪些数据是同一组数据，以便在同一图表中显示。

# 3. add_histogram: 记录数据分布
for i in range(10):
    x = torch.randn(1000)   # torch.randn是生成正态分布的随机数，参数为形状，这里是1000个随机数
    #这里形状的意思可以理解为生活概念中的一个概念，比如一张图片的大小为64x64，那么形状就是(64,64)，这里的形状就是(1000,)
    writer.add_histogram('distribution', x, global_step=i)

# 4. add_image: 记录图像
image = torch.rand(3, 64, 64)  # 创建随机图像   # 3表示通道数，64x64表示图像大小
#通道数指数指的是图像中每个像素点的颜色通道数，比如RGB图像，通道数就是3，灰度图像，通道数就是1。
#add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
#参数介绍：tag---标签，img_tensor---图像张量，global_step---全局步长，walltime---时间戳，dataformats---数据格式
#如果加了时间戳这个参数，那么这个参数就会被记录到TensorBoard中，这样就可以在TensorBoard中看到这个时间戳对应的图像。
# dataformats---数据格式，CHW表示通道在前，HWC表示通道在后，默认是CHW
#C表示channel，H表示height，W表示width，设置这个作用是为了方便TensorBoard读取图像数据。
writer.add_image('random_image', image, global_step=0)

# 5. add_images: 记录多张图像
images = torch.rand(16, 3, 64, 64)  # 创建16张随机图像
writer.add_images('random_images', images, global_step=0)

# 6. add_figure: 记录matplotlib图像
plt.figure(figsize=(10, 5))   # 创建一个画布, 10表示宽度，5表示高度
plt.plot(torch.randn(100))   # 绘制随机曲线
plt.title('Random Curve')   # 标题
writer.add_figure('matplotlib_plot', plt.gcf(), global_step=0)

# 7. 定义并记录模型结构
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
dummy_input = torch.rand(1, 1, 28, 28)  # MNIST图像大小
writer.add_graph(model, dummy_input)

# 8. add_embedding: 记录高维数据的嵌入向量
embeddings = torch.randn(100, 64)  # 100个64维的向量
metadata = [f'data_{i}' for i in range(100)]
writer.add_embedding(embeddings, metadata=metadata, global_step=0)

# 9. add_text: 记录文本
writer.add_text('experiment_notes', 'This is a TensorBoard demo experiment', global_step=0)

# 10. add_pr_curve: 记录PR曲线
labels = torch.randint(2, (100,))
predictions = torch.rand(100)
writer.add_pr_curve('pr_curve', labels, predictions, global_step=0)

# 11. add_custom_scalars: 创建自定义布局
layout = {
    'Taiwan': {
        'Loss': ['Multiline', ['Loss/train', 'Loss/val']],
        'Accuracy': ['Multiline', ['Accuracy/train', 'Accuracy/val']]
    }
}
writer.add_custom_scalars(layout)

# 12. add_hparams: 记录超参数
hparams = {
    'lr': 0.01,
    'bsize': 32,
    'optimizer': 'Adam'
}
metrics = {
    'hparam/accuracy': torch.rand(1).item() * 100,
    'hparam/loss': torch.rand(1).item()
}
writer.add_hparams(hparams, metrics)

# 关闭writer
writer.close()

print("TensorBoard logs generated. To view them, run:")
print("tensorboard --logdir=runs")
print("Then open http://localhost:6006 in your browser")


# TensorBoard 简介：
# TensorBoard 是一个可视化工具套件，最初是为 TensorFlow 设计的，现在也被 PyTorch 等其他深度学习框架广泛使用。它可以帮助我们：
# 跟踪和可视化损失及准确率等指标
# 可视化模型结构
# 查看权重、偏置等参数的分布
# 投影嵌入向量
# 显示图像、文本等数据
# tensorboard --logdir=runs 命令详解：
# bashCopytensorboard --logdir=runs [其他可选参数]
# 主要参数选项：
# bashCopy# 基础参数
# --logdir=LOGDIR          # 指定日志目录，必需参数
# --host=HOST             # 指定主机名，默认为 localhost
# --port=PORT             # 指定端口号，默认为 6006
# --bind_all              # 允许远程访问
# # 高级参数
# --purge_orphaned_data   # 是否清除孤立数据，默认 True
# --reload_interval=SECS  # 重新加载数据的时间间隔，默认 5 秒
# --samples_per_plugin    # 每个插件显示的最大样本数
# # 安全相关
# --path_prefix=PATH      # URL 路径前缀
# --window_title=TITLE    # 设置浏览器窗口标题
# 示例：
# bashCopy# 基本用法
# tensorboard --logdir=runs
# # 指定端口
# tensorboard --logdir=runs --port=8008
# # 允许远程访问
# tensorboard --logdir=runs --bind_all
# # 监视多个日志目录
# tensorboard --logdir name1:path1,name2:path2
# # 设置刷新间隔
# tensorboard --logdir=runs --reload_interval=10