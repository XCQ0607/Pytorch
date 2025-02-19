import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

from matplotlib import rcParams # 导入 Matplotlib 的 rcParams 模块
# 设置默认字体为支持中文的字体（例如：SimHei黑体）
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他支持中文的字体

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

print("8.1. 序列模型")
print("-" * 50)


# 8.1.1. 生成带噪声的正弦序列数据
def generate_data_from_sin(mean: float = 0.0, std: float = 0.2, size: int = 1000, tau: int = 4, freq: float = 0.01) -> \
Tuple[torch.Tensor, torch.Tensor]:
    """
    生成基于正弦函数的序列数据，添加高斯噪声

    参数:
    - mean: float, 噪声的均值（默认为0.0）
    - std: float, 噪声的标准差（默认为0.2）
    - size: int, 生成序列的长度（默认为1000）
    - tau: int, 历史窗口大小（默认为4）
    - freq: float, 正弦函数的频率系数（默认为0.01）

    返回:
    - time_steps: torch.Tensor, 时间步
    - values: torch.Tensor, 对应的序列值
    """
    print("generate_data_from_sin函数示例: 生成带噪声的正弦序列数据")

    time_steps = torch.arange(0, size, 1)   # 时间步
    base_signal = torch.sin(freq * time_steps)  # 正弦信号
    noise = torch.randn(size) * std + mean  # 高斯噪声
    values = base_signal + noise  # 正弦信号与噪声的和

    print(f"生成了{size}个数据点，前5个值: {values[:5].tolist()}")

    return time_steps, values


print("8.1.1.1. 自回归模型")
print("-" * 50)


# 8.1.1.2. 获取特征和标签
def get_features_and_labels(values: torch.Tensor, size: int, tau: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将序列数据转换为特征-标签对，适用于监督学习

    参数:
    - values: torch.Tensor, 序列数据
    - size: int, 数据点总数
    - tau: int, 历史窗口大小(特征数量)

    返回:
    - features: torch.Tensor, 特征，形状为 [size-tau, tau]
    - labels: torch.Tensor, 标签，形状为 [size-tau, 1]
    """
    print("get_features_and_labels函数示例: 将序列转换为特征-标签对")

    features = []
    labels = []

    for i in range(size - tau):
        features.append(values[i:i + tau])
        labels.append(values[i + tau])

    features = torch.stack(features)
    labels = torch.tensor(labels).reshape(-1, 1)

    print(f"转换后特征形状: {features.shape}, 标签形状: {labels.shape}")
    print(f"特征示例(前两行): {features[:2].tolist()}")
    print(f"标签示例(前两行): {labels[:2].tolist()}")

    return features, labels


# 8.1.1.3. 获取数据加载器
def get_dataloader(features: torch.Tensor, labels: torch.Tensor, train_size: int, batch_size: int,
                   shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练集和验证集的DataLoader

    参数:
    - features: torch.Tensor, 特征数据
    - labels: torch.Tensor, 标签数据
    - train_size: int, 训练集大小
    - batch_size: int, 批次大小
    - shuffle: bool, 是否打乱训练数据

    返回:
    - train_loader: DataLoader, 训练数据加载器
    - valid_loader: DataLoader, 验证数据加载器
    """
    print("get_dataloader函数示例: 创建数据加载器")

    # 分割训练集和验证集
    train_features = features[:train_size]
    train_labels = labels[:train_size]

    valid_features = features[train_size:]
    valid_labels = labels[train_size:]

    # 创建Dataset
    train_dataset = TensorDataset(train_features, train_labels)
    valid_dataset = TensorDataset(valid_features, valid_labels)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(valid_dataset)}")
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(valid_loader)}")

    return train_loader, valid_loader


# 自回归模型示例
class SimpleAutoregressive(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 10, output_size: int = 1):
        super(SimpleAutoregressive, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 训练函数
def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, criterion, optimizer,
                epochs: int = 5, writer: Optional[SummaryWriter] = None):
    print("训练自回归模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item() * inputs.size(0)

        valid_loss /= len(valid_loader.dataset)

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)

        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')


# 数据生成与处理
sample_size = 1000  # 样本大小
tau = 4  # 历史窗口大小
time_steps, values = generate_data_from_sin(mean=0.0, std=0.2, size=sample_size, tau=tau)
features, labels = get_features_and_labels(values=values, size=sample_size, tau=tau)
train_loader, valid_loader = get_dataloader(features, labels, train_size=int(0.8 * sample_size), batch_size=16)

# 模型训练
ar_model = SimpleAutoregressive(input_size=tau, hidden_size=32)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(ar_model.parameters(), lr=0.01)

# TensorBoard
writer = SummaryWriter(log_dir='./logs/autoregressive_model')
train_model(ar_model, train_loader, valid_loader, loss_fn, optimizer, epochs=5, writer=writer)


# 预测函数
def multi_step_predict(model: nn.Module, features: torch.Tensor, step_begin: int, step_num: int) -> torch.Tensor:
    """
    进行多步预测

    参数:
    - model: nn.Module, 训练好的模型
    - features: torch.Tensor, 特征数据
    - step_begin: int, 开始预测的位置索引
    - step_num: int, 预测步数

    返回:
    - predictions: torch.Tensor, 预测结果
    """
    print("multi_step_predict函数示例: 进行多步预测")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 获取初始历史窗口
    data = features[step_begin].clone().unsqueeze(0).to(device)  # [1, tau]
    result = []

    for i in range(step_num):
        # 单步预测
        with torch.no_grad():
            predicted = model(data).detach()  # [1, 1] when batch_size=1

        # 记录预测结果
        result.append(predicted.squeeze().cpu())  # 存到列表

        # 更新窗口：移除最旧的值，添加预测值作为新的历史值
        # 注意此处不再 unsqueeze(0)，直接拼接
        data = torch.cat((data[:, 1:], predicted), dim=1)  # 拼接后形状仍是 [1, tau]

    predictions = torch.stack(result)  # [step_num]
    return predictions



# 可视化
step001_predi = ar_model(features).detach().squeeze()
step500_predi = multi_step_predict(ar_model, features, step_begin=400, step_num=500)

plt.figure(figsize=(12, 6))
plt.plot(time_steps, values, label='原始带噪声正弦波', lw=0.8, ls='-', color='#2E7CEE', alpha=0.7)
plt.plot(time_steps[:sample_size - tau], step001_predi, label='单步预测', lw=1.0, ls='-.', color='#FCC526')
plt.plot(time_steps[400:400 + 500], step500_predi, label='多步预测(500步)', lw=1.0, ls='-.', color='#E53E31')
plt.xlabel('时间步')
plt.ylabel('值')
plt.title('自回归模型的单步和多步预测')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("./logs/autoregressive_model/prediction_comparison.png")
plt.close()

# 保存到TensorBoard
writer.add_figure('Predictions/Comparison', plt.gcf())

print("-" * 50)
print("TensorBoard logs generated. To view them, run:")
print("tensorboard --logdir=logs")
print("Then open http://localhost:6006 in your browser")

# 总结
"""
1. `generate_data_from_sin`函数用于生成带噪声的正弦序列，接受的参数包括：
    - mean: 噪声均值
    - std: 噪声标准差
    - size: 数据序列长度
    - tau: 历史窗口大小
    - freq: 正弦频率
2. `get_features_and_labels`函数将数据转换为监督学习所需的特征-标签对，接受的参数有：
    - values: 输入的原始序列
    - size: 序列的长度
    - tau: 历史窗口大小
3. `get_dataloader`函数用于创建训练集和验证集的DataLoader，接受的参数有：
    - features: 特征数据
    - labels: 标签数据
    - train_size: 训练集大小
    - batch_size: 批次大小
4. `train_model`函数用于训练自回归模型，接受的参数有：
    - model: 训练的模型
    - train_loader: 训练数据加载器
    - valid_loader: 验证数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - epochs: 训练轮数
    - writer: TensorBoard记录器（可选）
5. `multi_step_predict`函数用于多步预测，接受的参数有：
    - model: 训练好的模型
    - features: 输入的特征数据
    - step_begin: 预测开始的索引
    - step_num: 预测的步数
"""
