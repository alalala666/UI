import json
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
 #pip install grad-cam
 #
def DenseNet_Heatmap(model,img_path,save_path):
    model.eval()
    target_layers = [model.features[-1]]  # 拿到最后一个层结构
   # print(target_layers)
    data_transform = transforms.Compose([transforms.ToTensor(),#transforms.Resize((224, 224)),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)
 
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers)
    # 感兴趣的label
    target_category = 1  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    # 儲存圖像
    plt.savefig(save_path) 
    plt.close('all')
    #plt.show()
 
# model = models.densenet121(pretrained=models.DenseNet121_Weights.DEFAULT)
# img_path = "cats_and_dogs/dogs/dog.19.jpg"
# DenseNet_Heatmap(model,img_path)
 