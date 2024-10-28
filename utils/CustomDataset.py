from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
# from model_choose import PViT
class customDataset(Dataset):
    def __init__(self, df, label=0, pathColumn0=2, shuffle=False, transform=None):
        # --------------------------------------------
        # 初始化路径、转换以及其他参数
        # --------------------------------------------

        if shuffle:
            df = df.sample(frac=1)  # 如果shuffle为True，随机打乱数据
        self.df = df.reset_index(drop=True)  # 重置数据框索引，确保索引是顺序的
        self.label = self.df.iloc[:, label]  # 从数据框中获取标签列
        self.transform = transform  # 用户提供的数据转换函数
        self.pathColumn0 = pathColumn0  # 图像路径所在列的索引
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize([0.5] * 3, [0.5] * 3)  # 标准化图像数据
        ])  # 默认的数据转换

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. 从文件中读取数据（使用numpy.fromfile、PIL.Image.open等）
        # 2. 对数据进行预处理（使用torchvision.Transform等）
        # 3. 返回数据（例如图像和标签）
        # --------------------------------------------

        image = self.transform(Image.open(self.df.iloc[index, self.pathColumn0]).convert('RGB'))
        # 从指定路径打开图像文件并应用数据转换

        return self.df.iloc[index, self.pathColumn0], image, self.label[index]
        # 返回文件路径、图像数据和对应的标签

    def __len__(self):
        # --------------------------------------------
        # 指示数据集的总大小
        # --------------------------------------------
        return len(self.df)

class two_img_customDataset(Dataset):
    def __init__(self, df, label=0, pathColumn0=2, shuffle=False, transform=None):
        # --------------------------------------------
        # 初始化路径、转换以及其他参数
        # --------------------------------------------

        if shuffle:
            df = df.sample(frac=1)  # 如果shuffle为True，随机打乱数据
        self.df = df.reset_index(drop=True)  # 重置数据框索引，确保索引是顺序的
        self.label = self.df.iloc[:, label]  # 从数据框中获取标签列
        self.transform = transform  # 用户提供的数据转换函数
        self.pathColumn0 = pathColumn0  # 图像路径所在列的索引
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize([0.5] * 3, [0.5] * 3)  # 标准化图像数据
        ])  # 默认的数据转换

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. 从文件中读取数据（使用numpy.fromfile、PIL.Image.open等）
        # 2. 对数据进行预处理（使用torchvision.Transform等）
        # 3. 返回数据（例如图像和标签）
        # --------------------------------------------

        image0 = self.transform(Image.open(self.df.iloc[index, 2]).convert('RGB'))
        image1 = self.transform(Image.open(self.df.iloc[index, 3]).convert('RGB'))

        img_list  =[image0,image1]
        # 从指定路径打开图像文件并应用数据转换

        return self.df.iloc[index, 2] ,img_list, self.label[index]
        # 返回文件路径、图像数据和对应的标签

    def __len__(self):
        # --------------------------------------------
        # 指示数据集的总大小
        # --------------------------------------------
        return len(self.df)

# import pandas as pd 
# input_csv_path = 'dd\ScratchViT_Attention_dataset.csv'
# df = pd.read_csv(input_csv_path,encoding='unicode_escape')
# train_loader = DataLoader(two_img_customDataset(df[df['set'] != 2], shuffle = True), batch_size=16)

# model = PViT().to(self.device)
# for batch_id,(image_name,img,label) in enumerate(train_loader):
#     print(batch_id,image_name,img,label)
   
    # label = torch.autograd.Variable(label).to(self.device)
    # model(torch.autograd.Variable(img[0]).to(self.device), torch.autograd.Variable(img[1]).to(self.device))
    