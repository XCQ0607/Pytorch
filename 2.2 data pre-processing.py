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
        f.write('NA,Pave,127500,1\n')
        f.write('2,NA,106000,0\n')
        f.write('4,NA,178100,1\n')
        f.write('NA,NA,140000,0\n')
    print("文件已成功创建！")
except Exception as e:
    print("创建文件时发生错误：", e)

# 1. 读取数据
data_file = os.path.join('data', 'housing_tiny.csv')    # 数据文件路径为当前目录下的data文件夹中的housing.csv文件
housing_data = pd.read_csv(data_file)
print("原始数据预览:")
print(housing_data.head())

# 2. 数据探索和清洗
# 2.1 处理缺失值
print("\n缺失值分析:")
print(housing_data.isnull().sum())

# 使用中位数填充数值型特征的缺失值
num_cols = housing_data.select_dtypes(include=['int64', 'float64']).columns
housing_data[num_cols] = housing_data[num_cols].fillna(housing_data[num_cols].median())

# 使用众数填充类别型特征的缺失值
cat_cols = housing_data.select_dtypes(include=['object']).columns
housing_data[cat_cols] = housing_data[cat_cols].fillna(housing_data[cat_cols].mode().iloc[0])

print("\n缺失值处理后:")
print(housing_data.isnull().sum())

# 2.2 处理异常值
# 使用四分位数法识别并删除异常值
for col in num_cols:
    q1 = housing_data[col].quantile(0.25)
    q3 = housing_data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    housing_data = housing_data[(housing_data[col] >= lower) & (housing_data[col] <= upper)]

print("\n异常值处理后数据行数:", len(housing_data))

# 3. 特征工程
# 3.1 类别特征编码
housing_data = pd.get_dummies(housing_data, drop_first=True)

# 3.2 数值特征标准化   #sklearn是一个开源的机器学习库，提供了丰富的机器学习算法和工具，用于数据预处理、特征工程、模型训练和评估等任务。
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housing_data[num_cols] = scaler.fit_transform(housing_data[num_cols])

# 4. 划分训练集和测试集
from sklearn.model_selection import train_test_split
X = housing_data.drop('target', axis=1)
y = housing_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 转换为张量格式
print("\n训练集张量:")
X_train_tensor = torch.tensor(X_train.to_numpy(dtype=float), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(dtype=float), dtype=torch.float32)
print(X_train_tensor.shape, y_train_tensor.shape)

print("\n测试集张量:")
X_test_tensor = torch.tensor(X_test.to_numpy(dtype=float), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(dtype=float), dtype=torch.float32)
print(X_test_tensor.shape, y_test_tensor.shape)