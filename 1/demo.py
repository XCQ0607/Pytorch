import torch
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
if torch.cuda.is_available():
    print("当前CUDA设备:", torch.cuda.current_device())
    print("设备名称:", torch.cuda.get_device_name(0))

from torch.utils.data import Dataset
from PIL import Image
import os
import cv2  #用于读取图片
#安装库代码：pip install opencv-python

#input [img]   label ant
#除了可以使用PTL中的Image.open()方法，也可以使用opencv库来读取图片
#opencv库示例
# import cv2
# img = cv2.imread("E:\\PycharmProjects\\Pytorch\\dataset\\train\\ants\\0013035.jpg")   #imread作用：读取图片 参数：图片路径
# cv2.imshow("img",img) #imshow作用: 显示图片 参数： 1.窗口名称 2.图片数据
# cv2.waitKey(0)        # 等待键盘输入，参数为0表示无限等待
# cv2.destroyAllWindows()   # 销毁所有窗口


class MyData(Dataset):
    def __init__(self,root_dir,label_dir):    # 该魔术方法当创建一个事例对象时，会自动调用该函数
        self.root_dir = root_dir # self.root_dir 相当于类中的全局变量(传入的参数转化为全局变量)
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir) # 字符串拼接，根据是Windows或Lixus系统情况进行拼接
        self.img_path = os.listdir(self.path) # 获得路径下所有图片的地址

    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "/dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

print(f"蚂蚁数据集长度：{len(ants_dataset)}")
print(f"蜜蜂数据集长度：{len(bees_dataset)}")
train_dataset = ants_dataset + bees_dataset # train_dataset 就是两个数据集的集合了
print(f"总数据集长度：{len(train_dataset)}")

# print("第100张照片：")
# print(train_dataset[100])
# img,label = train_dataset[100]  # 获得第200张图片和标签,对应构造函数def __getitem__(self,idx)，魔法函数__getitem__是一个魔法函数，当使用索引访问对象时，会自动调用该函数
# print("label：",label)
# img.show()
#
# #延时5s
# import time
# time.sleep(5)
#
# print("第200张照片：")
# print(train_dataset[200])
# img,label = train_dataset[200]  # 获得第200张图片和标签,对应构造函数def __getitem__(self,idx)，魔法函数__getitem__是一个魔法函数，当使用索引访问对象时，会自动调用该函数
# print("label：",label)
# img.show()

#将每个数据的label数据输出到dataset文件夹中ant_label文件夹或者bee_label文件夹中，文件名为图片名,label数据也就是bee或ant
try:
    os.makedirs("/dataset/train/ants_label")
    os.makedirs("/dataset/train/bees_label")
except:
    print("文件夹已存在")

for i in range(len(train_dataset)):
    img,label = train_dataset[i]
    #获取文件名（有后缀名）
    name = img.filename
    print(name)
    #获取文件名（无后缀名）
    name1 = os.path.splitext(name)[0]    #splitext作用：分割文件名和后缀名 参数：文件名，[0]表示获取文件名，[1]表示获取后缀名
    print(name1)
    #无路径无后缀
    name2 = os.path.splitext(os.path.basename(name))[0]    #os.path.basename作用：获取文件名 参数：文件名
    print(name2)
    #获取文件后缀名
    name3 = os.path.splitext(name)[1]
    print(name3)
    #获取文件路径
    name4 = os.path.dirname(name)    #os.path.dirname作用：获取文件路径 参数：文件名
    print(name4)


    # img.show()
    # img.save("E:\\PycharmProjects\\Pytorch\\dataset\\train\\ant_lable\\"+str(i)+".jpg")
    with open("E:\\PycharmProjects\\Pytorch\\dataset\\train"+f"\\{label}"+"_label"+f"\\{name2}"+".txt","w") as f:
        f.write(label)

