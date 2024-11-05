import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

# 使用时间戳创建唯一的运行目录
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
writer = SummaryWriter(log_dir=f'runs/experiment_{current_time}')

# 使用 tag 的命名约定来组织中文显示
writer.add_scalar('训练/损失', torch.rand(1), 0)
writer.add_scalar('训练/准确率', torch.rand(1), 0)
writer.add_scalar('验证/损失', torch.rand(1), 0)
writer.add_scalar('验证/准确率', torch.rand(1), 0)

# 使用中文注释
writer.add_text('实验说明', '''
# 实验配置
- 学习率：0.001
- 批次大小：32
- 优化器：Adam
- 轮次：100
''', 0)

# 创建自定义布局
layout = {
    '训练指标': {
        '损失曲线': ['Multiline', ['训练/损失', '验证/损失']],
        '准确率曲线': ['Multiline', ['训练/准确率', '验证/准确率']]
    }
}
writer.add_custom_scalars(layout)

writer.close()