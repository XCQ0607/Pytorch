print("4.10. 实战Kaggle比赛：预测房价")

# 导入必要的包
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import os

# ---------------------- 分割线 ----------------------

print("读取训练和测试数据集...")

# 读取训练和测试数据集
train_data_url = 'http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv'
test_data_url = 'http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv'

train_data = pd.read_csv(train_data_url)      # 读取训练数据集 对象:DataFrame对象
test_data = pd.read_csv(test_data_url)        # 读取测试数据集

print("训练数据集的形状：", train_data.shape)
print("测试数据集的形状：", test_data.shape)

# ---------------------- 分割线 ----------------------

print("训练数据集的前4条样本：")
print(train_data.iloc[0:4, [0,1,2,3,-3,-2,-1]]) #0:4表示从第0行开始，到第4行结束，[0,1,2,3,-3,-2,-1]表示从第1列到第4列，第-3列到第-1列
print(train_data['SalePrice'].head())   # 显示前4行的SalePrice列

# iloc 是 Pandas 库中的一个函数，用于基于整数位置索引 DataFrame 中的行和列。与 loc 函数不同，loc 是基于标签（即行索引和列名）来索引数据的，而 iloc 则完全基于整数位置。
'''
loc 是 Pandas 库中用于基于标签索引数据的函数。与 iloc 不同，loc 允许你通过行索引（或称为行标签）和列名来访问 DataFrame 中的数据。下面是一些 loc 的使用示例：
示例 DataFrame
首先，我们创建一个简单的 DataFrame 作为示例：
Python
import pandas as pd
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)
print(df)
输出：
      Name  Age           City
0    Alice   25       New York
1      Bob   30  San Francisco
2  Charlie   35    Los Angeles
3    David   40        Chicago

使用 loc 访问数据
选择单行：
print(df.loc[0])
输出：
Name      Alice
Age          25
City    New York
Name: 0, dtype: object

选择多行：
print(df.loc[0:2])
注意：这里使用的是标签切片，与基于位置的切片不同，它包含结束标签。
输出：
      Name  Age           City
0    Alice   25       New York
1      Bob   30  San Francisco
2  Charlie   35    Los Angeles

选择单列：
print(df.loc[:, 'Age'])
输出：
0    25
1    30
2    35
3    40
Name: Age, dtype: int64

选择多列：
print(df.loc[:, ['Name', 'Age']])
输出：
      Name  Age
0    Alice   25
1      Bob   30
2  Charlie   35
3    David   40

选择特定行和列：
print(df.loc[0:2, ['Name', 'Age']])
输出：
      Name  Age
0    Alice   25
1      Bob   30
2  Charlie   35

使用条件选择行：
print(df.loc[df['Age'] > 30])
输出：
      Name  Age           City
2  Charlie   35    Los Angeles
3    David   40        Chicago
'''

#显示表的lable
print(train_data.columns)
#行数
print(train_data.shape[0])
#列数
print(train_data.shape[1])


# ---------------------- 分割线 ----------------------

print("数据预处理...")

# 删除Id列并保存测试集的Id
train_data.drop('Id', axis=1, inplace=True)
test_ids = test_data['Id']          # 保存测试集的Id
test_data.drop('Id', axis=1, inplace=True)

# 这行代码是使用 Pandas 库中的 drop 函数来删除 DataFrame 中的一列或多列。具体到这行代码：
# train_data.drop('Id', axis=1, inplace=True)
# 它的意思是：
# train_data：这是一个 DataFrame 对象，代表你要操作的数据集。
# .drop()：这是 Pandas 库中用于删除行或列的方法。
# 'Id'：这是你要删除的列名。在这个例子中，你想要删除名为 'Id' 的列。
# axis=1：这个参数指定了操作的轴。axis=0 表示操作将沿着行的方向进行（即删除行），而 axis=1 表示操作将沿着列的方向进行（即删除列）。在这个例子中，axis=1 表示你要删除的是列。
# inplace=True：这个参数决定了操作是否直接在原 DataFrame train_data 上进行。当 inplace=True 时，train_data 会被直接修改，不会返回新的 DataFrame 对象。如果不设置 inplace=True（或设置为 False），则 drop 方法不会修改原 DataFrame，而是返回一个新的已删除指定行或列的 DataFrame 对象。


# 合并训练和测试数据集，方便统一预处理
all_features = pd.concat([train_data.iloc[:, :-1], test_data], ignore_index=True)

# 这行代码使用了 Pandas 库中的 concat 函数来合并两个或多个 DataFrame 对象。具体到这行代码：
# all_features = pd.concat([train_data.iloc[:, :-1], test_data], ignore_index=True)
# 它的意思是：
# pd.concat(...)：这是 Pandas 库中用于合并 DataFrame 对象的函数。它可以沿着一个轴（默认是行方向）将多个 DataFrame 对象合并成一个新的 DataFrame。
# [train_data.iloc[:, :-1], test_data]：这是一个列表，包含了要合并的 DataFrame 对象。
# train_data.iloc[:, :-1]：这部分代码使用 iloc 选择了 train_data DataFrame 中的所有行（: 表示选择所有行）和除了最后一列之外的所有列（:-1 表示选择到倒数第二列为止）。这样做的目的通常是为了排除作为目标变量或不需要的列（比如在这个例子中可能是 'Id' 列或 'Target' 列）。
# test_data：这是另一个 DataFrame 对象，它将被与 train_data 的选定部分合并。
# ignore_index=True：这个参数告诉 concat 函数在合并后的新 DataFrame 中忽略原有的索引，并创建一个新的整数索引。如果不设置这个参数（或设置为 False），则合并后的 DataFrame 会保留原有 DataFrame 的索引，这可能会导致索引重复或混乱。

#eg
# train_data.iloc[:, 1:-1]：
# 选择 train_data 的所有行，但只选取 从第二列到倒数第二列（1:-1），即 不包括第一列 和 不包括最后一列。


# 为了帮助你理解 ignore_index=True 在 pd.concat 中的作用，下面是一个简单的可视化示例。
# 示例数据：
# 假设我们有两个简单的数据框 train_data 和 test_data：
# import pandas as pd
# # 创建示例数据
# train_data = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [4, 5, 6],
#     'C': [7, 8, 9]
# })
#
# test_data = pd.DataFrame({
#     'A': [10, 11],
#     'B': [12, 13],
#     'C': [14, 15]
# })
# print("train_data:")
# print(train_data)
# print("\ntest_data:")
# print(test_data)
# 输出：
# train_data:
#    A  B  C
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9
# test_data:
#     A   B   C
# 0  10  12  14
# 1  11  13  15
#
# 使用 pd.concat 合并数据：
# all_features = pd.concat([train_data.iloc[:, :-1], test_data], ignore_index=True)
# print("\nall_features (ignore_index=True):")
# print(all_features)
# 结果：
# all_features (ignore_index=True):
#     A   B
# 0   1   4
# 1   2   5
# 2   3   6
# 3  10  12
# 4  11  13
# 解释：
# train_data.iloc[:, :-1] 选择了 train_data 的前两列（去掉了最后一列 C），结果是：
#    A  B
# 0  1  4
# 1  2  5
# 2  3  6
# test_data 直接包含了所有的列：
#    A   B   C
# 0 10  12  14
# 1 11  13  15
# 合并时，使用 ignore_index=True：
# ignore_index=True 会重置索引，从 0 开始重新编号。
# 如果没有 ignore_index=True，原来的索引会保留（即 train_data 的索引是 0, 1, 2，test_data 的索引是 0, 1，这可能导致重复的索引）。
#
# 如果不使用 ignore_index=True：
# all_features_without_ignore = pd.concat([train_data.iloc[:, :-1], test_data])
# print("\nall_features (without ignore_index):")
# print(all_features_without_ignore)
# 输出：
# all_features (without ignore_index):
#     A   B   C
# 0   1   4   7
# 1   2   5   8
# 2   3   6   9
# 0  10  12  14
# 1  11  13  15
# 结果对比：
# 没有使用 ignore_index=True 时，原始索引被保留，导致 train_data 和 test_data 都有相同的索引值（0 和 1）。
# 使用 ignore_index=True 时，索引会被重新编号，从 0 开始，避免了索引重复的问题。
# 总结：
# ignore_index=True 会使得合并后的 DataFrame 重新编排索引，避免出现重复的索引。


# 处理数值特征
numeric_feats = all_features.dtypes[all_features.dtypes != 'object'].index  # 获取数值特征的列名,.index 表示提取列名,.dtypes 表示提取数据类型

print("标准化数值特征...")
# 标准化数值特征，并将缺失值填充为0
all_features[numeric_feats] = all_features[numeric_feats].apply(    #all_features: 包含所有特征的DataFrame,numeric_feats: 包含数值特征的列名
    #lambda x: (x - x.mean()) / x.std() 这是一个匿名函数（lambda函数），用于对每一列（特征）进行标准化操作。
    #x.mean() 计算列的均值，x.std() 计算列的标准差。
    lambda x: (x - x.mean()) / x.std()
)
all_features[numeric_feats] = all_features[numeric_feats].fillna(0) # 将缺失值填充为0
# 直观地说，我们标准化数据有两个原因： 首先，它方便优化。 其次，因为我们不知道哪些特征是相关的， 所以我们不想让惩罚分配给一个特征的系数比分配给其他任何特征的系数更大。

'''
这两行代码的作用是对 all_features 中的数值特征进行标准化，并将缺失值填充为 0。让我们通过一个简单的示例数据来可视化这个过程。

代码解析：
标准化（Standardization）：
all_features[numeric_feats] = all_features[numeric_feats].apply(
    lambda x: (x - x.mean()) / x.std()
)
这行代码的作用是对数值特征（numeric_feats）进行 标准化。标准化操作是通过对每一列（数值特征）进行如下计算来实现的：
               原始值 − 均值
标准化后的值 =  ——————————————
                  标准差
这样处理后，每一列的均值变为 0，标准差变为 1，使得不同特征的数据尺度一致，有助于机器学习模型的训练。

填充缺失值：
all_features[numeric_feats] = all_features[numeric_feats].fillna(0)
这行代码将 numeric_feats 中的 缺失值（NaN）填充为 0。

示例数据：
假设我们有一个包含数值特征和缺失值的简单 DataFrame all_features，以及一个列名列表 numeric_feats 来指定需要标准化的数值特征。
import pandas as pd
import numpy as np
# 创建一个示例 DataFrame
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, 6, 7, 8, 9],
    'C': [10, 11, 12, np.nan, 14]
}
all_features = pd.DataFrame(data)
# 数值特征列名
numeric_feats = ['A', 'B', 'C']
print("原始数据:")
print(all_features)

输出：
原始数据:
     A  B     C
0  1.0  5  10.0
1  2.0  6  11.0
2  NaN  7  12.0
3  4.0  8  NaN
4  5.0  9  14.0

第一步：标准化
在标准化过程中，我们会对每一列进行处理，让每列的均值变为 0，标准差变为 1。假设我们先对列 A、B 和 C 分别进行标准化。
all_features[numeric_feats] = all_features[numeric_feats].apply(
    lambda x: (x - x.mean()) / x.std()
)
print("\n标准化后的数据:")
print(all_features)
标准化过程：
列 A 的均值是 (1 + 2 + 4 + 5) / 4 = 3, 标准差为 1.58。
列 B 的均值是 (5 + 6 + 7 + 8 + 9) / 5 = 7, 标准差为 1.58。
列 C 的均值是 (10 + 11 + 12 + 14) / 4 = 11.75, 标准差为 1.58。

标准化后，数据如下：
标准化后的数据:
          A         B         C
0 -1.264911 -1.264911 -1.118034
1 -0.632456 -0.632456 -0.447214
2       NaN  0.000000  0.000000
3  0.632456  0.632456       NaN
4  1.264911  1.264911  1.118034

第二步：填充缺失值
我们可以看到在标准化过程中，列 A 和 C 仍然存在缺失值（NaN）。这时，我们使用 fillna(0) 方法将缺失值填充为 0。
all_features[numeric_feats] = all_features[numeric_feats].fillna(0)
print("\n填充缺失值后的数据:")
print(all_features)
输出结果：
填充缺失值后的数据:
          A         B         C
0 -1.264911 -1.264911 -1.118034
1 -0.632456 -0.632456 -0.447214
2  0.000000  0.000000  0.000000
3  0.632456  0.632456  0.000000
4  1.264911  1.264911  1.118034

总结：
标准化：通过对每一列进行均值归零、标准差为一的操作，使得数据集中的每个特征具有相同的尺度。
填充缺失值：使用 fillna(0) 将所有缺失值（NaN）填充为 0。

这样做的好处是：
标准化 使得不同特征的值位于同一尺度上，避免了在机器学习模型中某些特征对模型的影响过大。
填充缺失值 确保数据中不会因为缺失值（NaN）导致计算错误或模型训练失败。
'''


# 对类别特征进行独热编码
print("对类别特征进行独热编码...")
all_features = pd.get_dummies(all_features, dummy_na=True)

'''
独热编码（One-Hot Encoding）
独热编码（One-Hot Encoding） 是一种将分类变量转换为数值的常用方法，尤其适用于机器学习模型。它将每个类别特征转换为新的二进制特征（即 0 或 1），每个新特征代表一个类别。

举个例子： 假设有一个特征 MSZoning，它包含几个类别值，例如 "RL", "RM", "C", "FV"。为了将这些类别转换为数值，我们可以使用独热编码，得到四个新的特征：
MSZoning_RL
MSZoning_RM
MSZoning_C
MSZoning_FV
如果某一行的 MSZoning 值是 "RL"，那么该行的 MSZoning_RL 列值为 1，其他列（MSZoning_RM, MSZoning_C, MSZoning_FV）的值为 0。如果 MSZoning 为 "RM"，则 MSZoning_RM 列为 1，其他列为 0，依此类推。
使用 pandas.get_dummies：pandas 提供了 get_dummies 函数来实现这一转换，且可以处理缺失值（NaN）。
示例数据
我们通过一个简单的示例来说明 get_dummies 的操作。
原始数据：
import pandas as pd
# 创建一个示例 DataFrame
data = {
    'MSZoning': ['RL', 'RM', 'RL', 'FV', 'RM'],
    'LotConfig': ['Inside', 'Corner', 'Inside', 'Corner', 'Inside']
}
all_features = pd.DataFrame(data)
print("原始数据:")
print(all_features)

输出：
原始数据:
  MSZoning LotConfig
0       RL     Inside
1       RM     Corner
2       RL     Inside
3       FV     Corner
4       RM     Inside

使用 pd.get_dummies 进行独热编码
# 使用 pd.get_dummies 进行独热编码
all_features_encoded = pd.get_dummies(all_features, dummy_na=True)
print("\n独热编码后的数据:")
print(all_features_encoded)

结果：
独热编码后的数据:
   MSZoning_FV  MSZoning_RL  MSZoning_RM  LotConfig_Corner  LotConfig_Inside  LotConfig_na
0            0            1            0                 0                 1             0
1            0            0            1                 1                 0             0
2            0            1            0                 0                 1             0
3            1            0            0                 1                 0             0
4            0            0            1                 0                 1             0
解释：
MSZoning_FV, MSZoning_RL, MSZoning_RM：
这些是 MSZoning 列的独热编码后的新列。
如果 MSZoning 列的值为 "RL"，则 MSZoning_RL 列为 1，其他列（MSZoning_FV, MSZoning_RM）为 0，依此类推。

LotConfig_Corner, LotConfig_Inside：
同样地，LotConfig 列也被独热编码。
如果 LotConfig 的值是 "Inside"，则 LotConfig_Inside 为 1，LotConfig_Corner 为 0，反之亦然。

LotConfig_na：
由于我们设置了 dummy_na=True，如果有缺失值（NaN），get_dummies 会为缺失值创建一个新的列 LotConfig_na，表示该位置原本是缺失值。
在这个例子中，数据中没有缺失值，所以该列全为 0。


dummy_na=True 的作用：
设置 dummy_na=True 会为缺失值（NaN）创建一个专门的指示符列。在示例数据中，假设某个类别列存在缺失值（如 LotConfig 列有缺失值），dummy_na 就会创建一列 LotConfig_na 来标识这些缺失值。
使用独热编码的好处：
数值化分类数据：许多机器学习模型要求输入特征为数值类型，独热编码将分类变量转换为数值特征，使其可以被模型处理。
避免类别顺序假设：直接将分类数据转换为数值数据（例如用数字代替字符串）可能会误导模型，独热编码通过为每个类别创建一个二进制特征，避免了类别之间的顺序关系假设。
适用于有多个类别的特征：对于具有多个类别的特征（如 MSZoning），独热编码可以将每个类别表示为一个独立的特征，确保模型可以独立地学习每个类别的影响。

总结：
独热编码通过将每个类别变量转换为二进制特征向量，将类别值表示为 0 或 1 的指示符变量。
使用 pandas.get_dummies 可以方便地实现这一转换，并通过 dummy_na=True 处理缺失值。
'''
#这里把一些label的数据不是数字的转化成了新的列致使数据量（lable）变大了

print("预处理后的特征数量：", all_features.shape[1])

# ---检查特征的数据类型
print("检查特征的数据类型...")
object_cols = all_features.columns[all_features.dtypes == 'object']  # 选择数据类型为 'object' 的列
print(f"存在 {len(object_cols)} 列的数据类型为 object，需要处理。")

if len(object_cols) > 0:
    # 打印这些列的名称
    print("以下列的数据类型为 object：")
    print(object_cols.tolist())  # 打印列名列表

    # 处理方式1：删除这些非数值类型的列
    print("删除非数值类型的列...")
    all_features = all_features.drop(columns=object_cols)   # 删除这些列
    print("删除后特征数量：", all_features.shape[1])      # 打印删除后特征数量

# 确保所有特征都是数值类型
print("转换所有特征为 float32 类型...")
all_features = all_features.astype(np.float32)
#---

# 转换为张量
n_train = train_data.shape[0]

# 对标签取对数，缩小价格的差异，便于模型学习
train_labels = torch.tensor(np.log(train_data['SalePrice'].values).reshape(-1, 1), dtype=torch.float32)
# 对标签取对数是机器学习中常用的一个数据预处理技术，尤其是在处理具有很大范围差异（比如价格、收入等）的数值型标签时。通过对标签（如房价）取对数，可以缩小数值范围，减小极端值的影响，使得模型在训练时更容易学习。
#
# 为什么要对标签取对数？
# 许多机器学习算法在训练时都希望输入数据的分布是比较均匀的，尤其是在面对“偏态分布”时（即大多数数据集中在某一范围，但有少数极端值）。例如，房价（SalePrice）通常呈现右偏分布（大多数房价集中在中低价位，而少数高端房产价格非常高）。对标签取对数有以下几个好处：
# 缩小差异：对数变换可以将那些非常大的值压缩到较小的范围，这有助于减少极端值对模型的影响。
# 使数据更符合正态分布：许多机器学习模型（如线性回归、神经网络等）假设数据接近正态分布，对数变换可以帮助数据更符合这一假设。
# 便于模型学习：通过对数缩小价格差异后，模型在训练时的收敛速度可能会更快，预测性能也可能更好。

# 原始数据：
# 原始房价：
# 0     100000
# 1     200000
# 2     150000
# 3     300000
# 4     500000
# 5     800000
# 6    1200000
# Name: SalePrice, dtype: int64
# 如果直接使用这些房价数据训练模型，可能会出现极端值（例如，1200000）对模型产生较大影响。为了缩小这些差异，我们对标签进行对数变换：
# # 对房价标签进行对数变换
# import numpy as np
# train_data['LogSalePrice'] = np.log(train_data['SalePrice'])
# print("\n对数变换后的房价：")
# print(train_data['LogSalePrice'])
#
# 对数变换后的数据：
# 对数变换后的房价：
# 0    11.512925
# 1    12.903681
# 2    11.918390
# 3    12.611537
# 4    13.122364
# 5    13.587219
# 6    14.002074
# Name: LogSalePrice, dtype: float64


# 将特征转换为张量
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

# ---------------------- 分割线 ----------------------

print("定义模型...")

###多多学习这种模型定义方法
def get_net(input_features, hidden_sizes=[256, 128, 64], dropout_rates=[0.2, 0.2, 0.2]):
    '''
    构建一个多层感知机模型
    参数：
    - input_features: 输入特征的数量
    - hidden_sizes: 隐藏层的单元数列表，默认值为[256, 128, 64]
    - dropout_rates: 对应隐藏层的dropout率列表，默认值为[0.2, 0.2, 0.2]
    返回：
    - net: 构建的神经网络模型
    '''
    net = nn.Sequential()     # 使用Sequential容器
    for i in range(len(hidden_sizes)):
        if i == 0:  # 第一层
            net.add_module(f'fc{i}', nn.Linear(input_features, hidden_sizes[i]))    # 输入层到第一个隐藏层
        else:         # 其他隐藏层
            net.add_module(f'fc{i}', nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))   # 隐藏层到下一个隐藏层
        net.add_module(f'relu{i}', nn.ReLU())   # ReLU激活函数
        net.add_module(f'dropout{i}', nn.Dropout(dropout_rates[i]))  # Dropout层，防止过拟合
    # 最后一层，输出层
    net.add_module('output', nn.Linear(hidden_sizes[-1], 1))
    return net

# 定义损失函数和评价指标
loss_fn = nn.MSELoss()      # 均方误差损失函数

def log_rmse(net, features, labels):    # 对数均方根误差
    '''
    计算模型在给定数据上的对数均方根误差
    参数：
    - net: 神经网络模型
    - features: 输入特征
    - labels: 真实标签
    返回：
    - rmse: 对数均方根误差
    '''
    net.eval()   # 进入评估模式
    with torch.no_grad():
        preds = net(features)     # 预测
        # 将预测值小于1的设为1，避免取对数时的负无穷
        preds = torch.clamp(preds, 1, float('inf')) #clamp函数用于将输入张量中的元素限制在指定的范围内，参数:clamp(input, min, max),input是输入张量，min是最小值，max是最大值
        #float('inf')表示正无穷
        rmse = torch.sqrt(loss_fn(torch.log(preds), torch.log(labels)))   # 计算对数均方根误差
        #返回:(|(pred)½-(label)½|)½
    net.train()  # 进入训练模式
    return rmse.item()   # 返回对数均方根误差

# ---------------------- 分割线 ----------------------

print("定义训练函数...")

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    '''
    训练模型并返回训练和验证集上的损失
    参数：
    - net: 神经网络模型
    - train_features: 训练特征
    - train_labels: 训练标签
    - test_features: 验证特征（可选）
    - test_labels: 验证标签（可选）
    - num_epochs: 训练的轮数
    - learning_rate: 学习率
    - weight_decay: 权重衰减系数
    - batch_size: 批量大小
    返回：
    - train_ls: 每个epoch的训练集损失列表
    - test_ls: 每个epoch的验证集损失列表
    '''
    train_ls, test_ls = [], []
    dataset = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss_fn(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
        else:
            test_ls.append(None)
        if (epoch + 1) % 10 == 0 or epoch == 1:
            print(f"epoch {epoch + 1}, training log rmse {train_ls[-1]:.4f}")
    return train_ls, test_ls

# ---------------------- 分割线 ----------------------

print("定义K折交叉验证函数...")

def get_k_fold_data(k, i, X, y):
    '''
    返回第i折交叉验证所需要的训练和验证数据
    参数：
    - k: 折数
    - i: 第i折
    - X: 所有特征
    - y: 所有标签
    返回：
    - X_train: 训练特征
    - y_train: 训练标签
    - X_valid: 验证特征
    - y_valid: 验证标签
    '''
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j+1) * fold_size)
        X_part, y_part = X[idx], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], dim=0)
            y_train = torch.cat([y_train, y_part], dim=0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
    '''
    执行K折交叉验证
    参数：
    - k: 折数
    - X_train: 训练特征
    - y_train: 训练标签
    - num_epochs: 训练轮数
    - lr: 学习率
    - weight_decay: 权重衰减系数
    - batch_size: 批量大小
    返回：
    - train_l_sum: 训练集平均损失
    - valid_l_sum: 验证集平均损失
    '''
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        print(f"第{i+1}折验证...")
        X_tr, y_tr, X_val, y_val = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, X_tr, y_tr, X_val, y_val, num_epochs, lr, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f"折{i+1}，训练log rmse：{train_ls[-1]:.4f}，验证log rmse：{valid_ls[-1]:.4f}")
    return train_l_sum / k, valid_l_sum / k

# ---------------------- 分割线 ----------------------

print("执行K折交叉验证...")

# 设置超参数
k = 5
num_epochs = 100
lr = 0.01
weight_decay = 0.001
batch_size = 64

train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f"{k}折验证：平均训练log rmse：{train_l:.4f}，平均验证log rmse：{valid_l:.4f}")

# ---------------------- 分割线 ----------------------

print("训练最终模型并对测试集进行预测...")

def train_and_predict(train_features, test_features, train_labels, test_data,
                      num_epochs, lr, weight_decay, batch_size):
    '''
    训练模型并对测试集进行预测
    参数：
    - train_features: 训练特征
    - test_features: 测试特征
    - train_labels: 训练标签
    - test_data: 测试数据集（包含Id列）
    - num_epochs: 训练轮数
    - lr: 学习率
    - weight_decay: 权重衰减系数
    - batch_size: 批量大小
    '''
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    print(f"训练完成，最终训练log rmse：{train_ls[-1]:.4f}")
    # 对测试集进行预测
    net.eval()
    with torch.no_grad():
        preds = net(test_features).numpy()
    # 将预测结果保存到csv文件
    test_data['SalePrice'] = pd.Series(preds.reshape(-1))
    submission = pd.concat([test_ids, test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
    print("预测结果已保存到submission.csv")

train_and_predict(train_features, test_features, train_labels, test_data,
                  num_epochs, lr, weight_decay, batch_size)

# ---------------------- 分割线 ----------------------

# 总结
'''
总结：
- 在数据预处理阶段，确保所有特征都是数值类型非常重要。
- 可以使用 `all_features.dtypes` 检查每一列的类型。
- 如果存在 `object` 类型的列，需要对其进行处理，例如删除或转换为数值类型。
- 转换为 PyTorch 张量之前，确保所有数据都是数值类型，且没有缺失值或非数值的数据。

- get_net(input_features, hidden_sizes=[256,128,64], dropout_rates=[0.2,0.2,0.2]): 构建多层感知机模型
    - input_features: 输入特征数量
    - hidden_sizes: 隐藏层的单元数列表，默认值为[256,128,64]
    - dropout_rates: 对应隐藏层的dropout率列表，默认值为[0.2,0.2,0.2]

- log_rmse(net, features, labels): 计算模型在给定数据上的对数均方根误差
    - net: 神经网络模型
    - features: 输入特征
    - labels: 真实标签

- train(net, train_features, train_labels, test_features, test_labels,
        num_epochs, learning_rate, weight_decay, batch_size): 训练模型并返回损失
    - net: 神经网络模型
    - train_features: 训练特征
    - train_labels: 训练标签
    - test_features: 验证特征（可选）
    - test_labels: 验证标签（可选）
    - num_epochs: 训练轮数
    - learning_rate: 学习率
    - weight_decay: 权重衰减系数
    - batch_size: 批量大小

- get_k_fold_data(k, i, X, y): 返回第i折交叉验证的数据
    - k: 折数
    - i: 第i折
    - X: 所有特征
    - y: 所有标签

- k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size): 执行K折交叉验证
    - k: 折数
    - X_train: 训练特征
    - y_train: 训练标签
    - num_epochs: 训练轮数
    - lr: 学习率
    - weight_decay: 权重衰减系数
    - batch_size: 批量大小

- train_and_predict(train_features, test_features, train_labels, test_data,
                    num_epochs, lr, weight_decay, batch_size): 训练模型并对测试集进行预测
    - train_features: 训练特征
    - test_features: 测试特征
    - train_labels: 训练标签
    - test_data: 测试数据集（包含Id列）
    - num_epochs: 训练轮数
    - lr: 学习率
    - weight_decay: 权重衰减系数
    - batch_size: 批量大小
'''

# ---------------------- 分割线 ----------------------

# 练习：

# 1. 把预测提交给Kaggle，它有多好？

#    答：将生成的submission.csv提交到Kaggle，可以查看在测试集上的得分。根据模型的性能和参数设置，得分会有所不同。

# 2. 能通过直接最小化价格的对数来改进模型吗？如果试图预测价格的对数而不是价格，会发生什么？

#    答：可以尝试对标签取对数，使得目标更接近正态分布，从而可能提高模型的性能。需要修改训练时的损失函数和预测时的反变换。

# 3. 用平均值替换缺失值总是好主意吗？提示：能构造一个不随机丢失值的情况吗？

#    答：不一定。如果缺失值不是随机分布的，用平均值替换可能引入偏差。应该根据特征的实际情况考虑如何处理缺失值。

# 4. 通过\(K\)折交叉验证调整超参数，从而提高Kaggle的得分。

#    答：可以在K折交叉验证中尝试不同的超参数组合，例如学习率、权重衰减、隐藏层大小等，选择验证集上表现最好的模型。

# 5. 通过改进模型（例如，层、权重衰减和dropout）来提高分数。

#    答：可以增加模型的复杂度，例如增加隐藏层数量、单元数，或者使用更高级的模型。同时，使用正则化技术防止过拟合。

# 6. 如果我们没有像本节所做的那样标准化连续的数值特征，会发生什么？

#    答：如果不对数值特征进行标准化，可能会导致不同特征之间的数值差异过大，影响模型的训练效果，导致模型收敛变慢或者性能下降。

