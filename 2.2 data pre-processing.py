import os
import pandas as pd
import numpy as np
import torch

# 预处理：写入数据
try:
    #确保当前目录下存在data文件夹，否则创建
    # os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    os.makedirs('data', exist_ok=True)
    # data_file = os.path.join('..', 'data', 'house_tiny.csv')
    data_file = os.path.join('data', 'housing_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price,target\n')
        f.write('NA,"Pave,",127500,1\n')
        f.write('2,NA,106000,0\n')
        f.write('4,NA,178100,1\n')
        f.write('NA,NA,140000,0\n')
        f.write('3,"NA",148000,1\n')
        f.write('NA,"NA",140000,0\n')
        #NA是一个占位符，表示缺失值
    print("文件已成功创建！")
except Exception as e:
    print("创建文件时发生错误：", e)

#根据house_tiny.csv写一个字典data
data = {
    'NumRooms': [1.0, 2.0, 3.0, np.nan, 4.0, 5.0, np.nan],
    'Alley': ['Pave', 'NA', 'NA', 'NA', 'Pave', 'Pave', 'NA'],
    'Price': [127500, 106000, 178100, 140000, 148000, 140000, 127500],
    'target': [1, 0, 1, 0, 1, 0, 1]
}
housing_data1 = pd.DataFrame(data, index=range(2, 9))   #不包含9

print("1. 读取数据")
# 1. 读取数据
#csv文件是一种常见的文本文件格式，用于存储表格数据。
data_file = os.path.join('data', 'housing_tiny.csv')    # 数据文件路径为当前目录下的data文件夹中的housing.csv文件
housing_data = pd.read_csv(data_file)

# 假设你的DataFrame是housing_data
housing_data = housing_data.reset_index(drop=True)  # 删除原始索引
housing_data.index += 1  # 将索引加1，使其从1开始

print("housing_data原始数据预览:")
print(housing_data)
print(".head()[默认显示前5行]数据预览:")
print(housing_data.head())
print("housing_data1数据预览:")
print(housing_data1)
#加.head()打印DataFrame的前几行数据（默认是前5行）
# head()方法有一个可选参数n，用于指定要显示的行数。例如，housing_data.head(10)会显示前10行。

# 查看数据的基本信息
print("\n数据基本信息:")
print(housing_data.info())
#.info() 方法用于获取DataFrame的基本信息，包括数据类型、非空值数量等。

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6 entries, 0 to 5
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   NumRooms  3 non-null      float64
 1   Alley     1 non-null      object 
 2   Price     6 non-null      int64  
 3   target    6 non-null      int64  
dtypes: float64(1), int64(2), object(1)
memory usage: 324.0+ bytes
None

<class 'pandas.core.frame.DataFrame'>
这行告诉我们 housing_data 是一个Pandas DataFrame类的实例。

RangeIndex: 6 entries, 0 to 5
这描述了DataFrame的索引。在这里，我们有一个RangeIndex，它是从0开始到5结束的，总共有6个条目。这意味着DataFrame有6行。

Data columns (total 4 columns):
这告诉我们DataFrame中有4列数据。
接下来的部分列出了每一列的具体信息：
列的信息：
0 NumRooms 3 non-null float64
    0 是列的序号（从0开始计数）。
    NumRooms 是列的名称。
    3 non-null 表示这一列有3个非空值。
    float64 是这一列的数据类型，表示64位浮点数。
1 Alley 1 non-null object
    1 是列的序号。
    Alley 是列的名称。
    1 non-null 表示这一列只有1个非空值。
    object 通常用于存储字符串或混合数据类型。在Pandas中，字符串默认被存储为object类型。
2 Price 6 non-null int64
    2 是列的序号。
    Price 是列的名称。
    6 non-null 表示这一列有6个非空值。
    int64 是这一列的数据类型，表示64位整数。
3 target 6 non-null int64
    3 是列的序号。
    target 是列的名称。
    6 non-null 表示这一列有6个非空值。
    int64 是这一列的数据类型。

dtypes: float64(1), int64(2), object(1)
这提供了一个数据类型的汇总。在这里，我们有1列是float64类型，2列是int64类型，和1列是object类型。

memory usage: 324.0+ bytes
这告诉我们DataFrame当前使用了多少内存。在这个例子中，它使用了大约324字节或稍多一些的内存。

None
这是 info() 方法的返回值。在大多数情况下，这个方法用于其副作用（即打印信息到控制台），而不是为了获取一个返回值。因此，它返回None。

'''

# 查看数据的统计信息
print("\n数据统计信息:")
print(housing_data.describe())
#.describe() 方法用于获取DataFrame的统计信息，包括计数、均值、标准差、最小值、25%分位数、中位数、75%分位数和最大值等。
#当字段对应的数据为非数字时，不会进行统计分析
#count(计数)  mean(均值)  std(标准差)  min(最小值)  25%(25%分位数)  50%(中位数)  75%(75%分位数)  max(最大值)
'''
       NumRooms          Price    target
count       3.0       6.000000  6.000000
mean        3.0  139933.333333  0.500000
std         1.0   23781.645584  0.547723
min         2.0  106000.000000  0.000000
25%         2.5  130625.000000  0.000000
50%         3.0  140000.000000  0.500000
75%         3.5  146000.000000  1.000000
max         4.0  178100.000000  1.000000
'''

# 查看数据的形状
print("\n数据形状:")
print(housing_data.shape)
#.shape 是一个元组，用于获取DataFrame的行数和列数。即数据集数与字段数

# 查看数据的列名
print("\n数据列名:")
print(housing_data.columns)
#.columns 是一个列表，用于获取DataFrame的列名。

# 查看数据的索引
print("\n数据索引:")
print(housing_data.index)
#.index 是一个列表，用于获取DataFrame的行索引。

#更改索引
housing_data.index = range(2, len(housing_data) + 2)    #不包含len(housing_data) + 2
print("\n更改索引后数据预览:")
print(housing_data)
#更改step
housing_data.index = range(1, 2*len(housing_data) + 1, 2)    #不包含len(housing_data) + 1
print("\n更改索引后数据预览:")
print(housing_data)
#.index = range(1, len(housing_data) + 1) 这行代码将DataFrame的索引从默认的范围索引（从0开始）更改为从1开始的范围索引。

# 查看数据的类型
print("\n数据类型:")
print(housing_data.dtypes)
#.dtypes 是一个Series，用于获取DataFrame中每列的数据类型。
# 在Pandas库中，当你查看DataFrame或Series的数据类型时，输出的dtype: object表示对应列的数据类型是Python中的通用对象类型。
# 1.列中包含混合数据类型：如果某一列原本应该只包含一种数据类型（如整数或浮点数），但实际上由于某些原因（如数据输入错误、缺失值被填充为字符串等）包含了多种数据类型，Pandas会将该列的数据类型设置为object，以便能够容纳这些不同类型的值。
# 2.列中包含字符串：在Pandas中，字符串类型的数据通常会被存储为object类型，因为字符串本质上是Python对象。所以，如果你的DataFrame中有一列是文本数据（如你例子中的Alley列），那么它的数据类型就会是object。
# 3.空值（NaN）的处理：在某些情况下，如果一列中包含空值（NaN），并且这些空值与其他数值类型的数据混合在一起，Pandas可能会将整个列的数据类型设置为object。这是因为NaN在Pandas中是浮点数类型的一个特殊值，但如果列中原本的数据类型是整数，为了容纳NaN，Pandas不会将整数列自动转换为浮点数列（因为这样会改变其他非空值的数据类型），而是会将整个列的数据类型改为更通用的object。

# 查看数据的非空值数量
print("\n非空值数量:")
print(housing_data.count())
#.count() 方法用于获取DataFrame中每列非空值的数量。

print("\n2. 数据探索和清洗")
print("2.1 处理缺失值")
# 2. 数据探索和清洗
# 2.1 处理缺失值
print("\n缺失值分析:")
print(housing_data.isnull().sum())
#.isnull() 用于检查数据中是否存在缺失值，返回一个布尔型的DataFrame，其中True表示缺失值，False表示非缺失值。
#.sum() 用于计算每列中缺失值的数量。


# 使用中位数填充数值型特征的缺失值
#筛选出特定columns(列)
num_cols = housing_data.select_dtypes(include=['int64', 'float64']).columns
housing_data[num_cols] = housing_data[num_cols].fillna(housing_data[num_cols].median())
# 这段代码首先通过select_dtypes方法筛选出housing_data数据框（DataFrame）中所有数据类型为int64和float64的列，然后将这些列名存储在num_cols变量中。接着，使用fillna方法将这些数值型列中的缺失值（NaN）替换为各自列的中位数。
#处理后的housing_data
print("\n先使用中位数填充数值型特征的缺失值处理后的housing_data:")
print(housing_data)

# 使用众数填充类别型特征的缺失值
#筛选出特定columns(列)
cat_cols = housing_data.select_dtypes(include=['object']).columns
housing_data[cat_cols] = housing_data[cat_cols].fillna(housing_data[cat_cols].mode().iloc[0])
# 这段代码首先通过select_dtypes方法筛选出housing_data数据框（DataFrame）中所有数据类型为object的列，然后将这些列名存储在cat_cols变量中。接着，使用fillna方法将这些类别型列中的缺失值（NaN）替换为各自列的众数。
# 在大多数情况下，当我们使用 .mode() 来填充缺失值时，我们只关心最频繁出现的值，即第一行的值。这就是 .iloc[0] 的作用：它确保我们只选择每列的第一个众数。
# 因此，.iloc[0] 是必要的，除非你确定每列都只有一个众数，或者你愿意接受 .mode() 返回的 DataFrame 中的其他行作为填充值（这通常不是期望的行为）。
#处理后的housing_data
print("\n后使用众数填充类别型特征的缺失值处理后的housing_data:")
print(housing_data)

print("\n缺失值处理后:")
print(housing_data.isnull().sum())

#fillna函数：
#fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)
#value：用于填充缺失值的标量值或字典。如果是标量值，则将所有缺失值替换为该值；如果是字典，则根据字典的键值对来填充缺失值。
#method：用于填充缺失值的方法。可以是'ffill'（前向填充）、'bfill'（后向填充）或其他插值方法。
#axis：指定填充方向。0表示按列填充，1表示按行填充。
#inplace：是否原地修改数据框。如果为True，则在原始数据框上进行修改；如果为False，则返回一个新的修改后的数据框。
#limit：用于限制连续缺失值的最大填充数量。
#downcast：用于向下转换数据类型。如果为'signed'（有符号）或'unsigned'（无符号），则尝试将数据类型转换为更小的类型。

# .median() 方法用于计算DataFrame中每列的中位数。
# .mode() 方法用于计算DataFrame中每列的众数。
# .quantile(q=0.5) 方法用于计算DataFrame中每列的分位数。默认情况下，计算的是中位数。　q=0.4 代表 40%分位数
# .iloc[] 用于按位置索引选择数据。    如：df.iloc[0] 表示选择第一行数据，df.iloc[:, 0] 表示选择第一列数据

# 2.2 处理异常值
print("\n2.2 处理异常值")
# 使用四分位数法识别并删除异常值
for col in num_cols:    #num_
    q1 = housing_data[col].quantile(0.25)
    q3 = housing_data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    housing_data = housing_data[(housing_data[col] >= lower) & (housing_data[col] <= upper)]
# 这段代码用于根据四分位距（IQR）方法过滤掉异常值（outliers）。具体步骤如下：
# 计算指定列col的第一四分位数（q1，即25%分位数）和第三四分位数（q3，即75%分位数）。
# 计算四分位距iqr = q3 - q1。
# 根据iqr计算异常值的下界lower = q1 - 1.5 * iqr和上界upper = q3 + 1.5 * iqr。
# 使用布尔索引过滤掉housing_data中指定列col的值小于lower或大于upper的行，即只保留在[lower, upper]范围内的数据。

#过滤后的housing_data
print("\n异常值处理后:")
print(housing_data)

print("\n异常值处理后数据行数:", len(housing_data))

# 3. 特征工程
print("\n3. 特征工程")
# 3.1 类别特征编码
print("3.1 类别特征编码")

print("\n类别特征编码处理前的housing_data:")
print(housing_data.head())

# print(housing_data)本身只是用来打印housing_data这个DataFrame的内容，它不会显示DataFrame的内部结构或编码方式的变化。无论housing_data是否经过独热编码，print函数都会以表格的形式输出DataFrame中的数据。
# 然而，独热编码对housing_data的内容确实有影响，这种影响不是通过简单的print调用就能直接看出来的。当您使用pd.get_dummies(housing_data, drop_first=True)对housing_data进行独热编码后，DataFrame中的分类变量会被转换成一系列的二进制列。这意味着，如果原始DataFrame中包含分类数据，那么编码后的DataFrame将会有更多的列，并且这些新列将包含0和1来表示原始的分类信息。

# 使用独热编码（One-Hot Encoding）对类别特征进行编码
#pd.get_dummies()函数用于将分类变量（也称为类别型数据或定性数据）转换为独热编码的形式。
housing_data = pd.get_dummies(housing_data, drop_first=True)

# 处理后的housing_data
print("\n类别特征编码处理后的housing_data:")
print(housing_data.head())
# 这段代码使用pd.get_dummies函数对housing_data中的类别特征进行独热编码（One-Hot Encoding）。
# 独热编码是一种常用的特征编码方法，它将类别特征转换为二进制特征，每个类别对应一个新的二进制特征列。


# 这行代码使用了Pandas库的get_dummies函数，它的主要作用是将分类变量（也称为类别型数据或定性数据）转换为一种能够提供给机器学习算法使用的格式，即独热编码（One-Hot Encoding）。
# 在机器学习中，很多算法无法直接处理分类数据，因此需要将它们转换为数值型数据。独热编码就是一种常用的转换方法，它为每个类别创建一个新的二进制列，表示原始数据中的每个唯一类别。如果原始数据中的某个样本属于某个类别，则在新创建的对应类别的列中标记为1，否则标记为0。

# drop_first=True：这个参数的作用是去掉每个类别独热编码后的第一列，以避免多重共线性问题。多重共线性是指在回归模型中，自变量之间存在高度的相关性，这会导致模型的预测精度下降。通过去掉一个类别（通常是第一个类别）的编码列，可以确保模型的稳定性。

# 3.2 数值特征标准化   #sklearn是一个开源的机器学习库，提供了丰富的机器学习算法和工具，用于数据预处理、特征工程、模型训练和评估等任务。
print("\n3.2 数值特征标准化")
# 对数值特征进行标准化处理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housing_data[num_cols] = scaler.fit_transform(housing_data[num_cols])
# 标准化处理，即将数值特征缩放到均值为0，方差为1的分布中
# 这样做是为了改善后续机器学习模型的性能和稳定性。

# 处理后的housing_data
print("\n数值特征标准化处理后的housing_data:")
print(housing_data.head())

# 4. 划分训练集和测试集
print("\n4. 划分训练集和测试集")
#.drop(columns=['target'], axis=1) 表示删除名为 'target' 的列。
# axis=1 表示删除列。类似sum,mean中的dim
# train_test_split() 函数用于将数据集划分为训练集和测试集。
from sklearn.model_selection import train_test_split
X = housing_data.drop(labels='target', axis=1)  #建议始终使用axis参数   若不指定，drop会尝试删除名为 'target' 的行或列
#相当于X = housing_data.drop(columns='target')
y = housing_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#输出X_train,X_test,y_train,y_test
print("\nX_train, X_test, y_train, y_test:")
print(f"X_train\n{X_train}\nX_test\n{X_test}\ny_train\n{y_train}\ny_test\n{y_test}\n")

# 这行代码是使用scikit-learn库中的train_test_split函数来将数据集拆分为训练集和测试集。具体来说：
# X 通常代表特征数据，即输入变量。
# y 代表目标数据，即我们想要预测的输出变量。
# test_size=0.2 表示测试集将包含原始数据的20%。因此，训练集将自动包含剩下的80%。
# random_state=42 是一个随机种子，用于确保每次拆分时都能得到相同的结果。这在需要重复实验或比较不同模型时非常有用，因为它确保了数据拆分的一致性。
#
# 函数train_test_split会返回四个数组：
# X_train：用于训练的特征数据子集。
# X_test：用于测试的特征数据子集。
# y_train：与X_train相对应的目标数据（即标签）。
# y_test：与X_test相对应的目标数据（即标签）。
# 这行代码的目的是为了将原始数据集（由X和y组成）随机拆分为两个独立的部分：一个用于训练模型（X_train和y_train），另一个用于评估模型的性能（X_test和y_test）。这是机器学习中常见的一个步骤，有助于了解模型在未见过的数据上的表现如何。

# drop 函数可以接收以下主要参数：
# labels：要删除的行或列的名称或索引。可以是一个字符串（对于单个标签）或一个字符串/整数的列表（对于多个标签）。在你的例子中，'target' 就是要删除的列的名称。
# axis：指定要删除的是行还是列。axis=0 表示删除行，axis=1 表示删除列。在你的例子中，axis=1 表示要删除列。
# index / columns：这两个参数可以替代 labels 和 axis，用于更明确地指定要删除的是行索引还是列名称。通常，使用 labels 和 axis 就足够了，但在某些情况下，为了代码的清晰性，可能会选择使用 index 或 columns。
# level：如果数据框（DataFrame）是多层索引的，这个参数用于指定要删除的层级。
# inplace：是否在原地修改数据框。如果设置为 True，则不会返回新的数据框，而是直接在原始数据框上进行修改。默认情况下，这个参数是 False，意味着 drop 函数会返回一个新的数据框，原始数据框保持不变。
# errors：如果尝试删除不存在的标签，这个参数控制函数的行为。可以是 'ignore'（忽略错误，不删除任何内容也不报错）、'raise'（抛出错误）或其他有效值。

# 得到的 X_train 和 y_train 是一组对应的训练数据和标签，用于训练模型；而 X_test 和 y_test 则是一组对应的测试数据和标签，用于评估模型的性能。
# X_train 是训练数据，它包含了用于训练模型的特征（即除了 target 列之外的所有列）。
# y_train 是训练标签，它对应了 X_train 中每个数据点的目标值（即 target 列的值）。
# 换句话说，X_train 包含了模型需要学习的输入数据，而 y_train 包含了模型应该学会预测的对应输出或标签。

# 5. 转换为张量格式
print("\n5. 转换为张量格式")
print("\n训练集张量:")
X_train_tensor = torch.tensor(X_train.to_numpy(dtype=float), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(dtype=float), dtype=torch.float32)
print(X_train_tensor.shape, y_train_tensor.shape)

print("\n测试集张量:")
X_test_tensor = torch.tensor(X_test.to_numpy(dtype=float), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(dtype=float), dtype=torch.float32)
print(X_test_tensor.shape, y_test_tensor.shape)
print("\n张量:")
print(X_train_tensor)
print(y_train_tensor)
print(X_test_tensor)
print(y_test_tensor)

# 6. 保存处理后的数据
print("\n6. 保存处理后的数据")
# 保存处理后的训练集和测试集到data目录
#index=False 表示不保存行索引
X_train.to_csv('data/X_train.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)


# CSV（Comma-Separated Values）文件是一种常见的文本文件格式，用于存储表格数据，如电子表格或数据库。CSV文件由任意数量的记录组成，记录之间以某种换行符分隔（例如\n）；每条记录由字段组成，字段之间的分隔符是其他字符或字符串，最常见的是逗号或制表符。
# 通常，所有记录都有完全相同的字段序列，即每一行都有相同数量的字段，并且对应的字段表示相同的数据类型。CSV文件通常不包含任何格式化信息，如字体样式或颜色，只包含纯文本数据。
# 一个简单的CSV文件示例如下：
# Name,Age,Occupation
# John Doe,30,Engineer
# Jane Smith,28,Designer
# Bob Johnson,35,Manager
#
# 注意，如果字段中包含逗号、换行符或双引号等特殊字符，通常需要用双引号将整个字段括起来
# 如："你好，熊长青！"