import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


# NiN模型定义
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


def create_nin_model(num_classes=10):
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, num_classes, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    return net


# 数据加载
def load_data_fashion_mnist(batch_size, resize=None):
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform)

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter


# 训练函数
def train_model(net, train_iter, test_iter, num_epochs, lr, device, writer):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)  # 使用Adam优化器

    net.to(device)

    for epoch in range(num_epochs):
        net.train()
        train_loss, train_correct = 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (output.argmax(1) == y).sum().item()

        train_acc = train_correct / len(train_iter.dataset)

        # 将训练损失和准确率写入TensorBoard
        writer.add_scalar('Loss/train', train_loss / len(train_iter), epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_iter)}, Train Accuracy: {train_acc:.4f}")

        net.eval()
        test_correct = 0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                output = net(X)
                test_correct += (output.argmax(1) == y).sum().item()

        test_acc = test_correct / len(test_iter.dataset)

        # 将测试准确率写入TensorBoard
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        print(f"Test Accuracy: {test_acc:.4f}")


# 模型训练配置
batch_size = 128
num_epochs = 10
lr = 0.001  # 更低的学习率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化TensorBoard的SummaryWriter
writer = SummaryWriter(log_dir='./logs/nin_model')

# 加载数据并创建模型
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
model = create_nin_model(num_classes=10)

# 训练模型
train_model(model, train_iter, test_iter, num_epochs, lr, device, writer)

# 关闭TensorBoard writer
writer.close()

print("TensorBoard logs generated. To view them, run:")
print("tensorboard --logdir=logs")
print("Then open http://localhost:6006 in your browser")
