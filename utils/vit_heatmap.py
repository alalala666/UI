import json
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM#, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn

def ViT_Heatmap(model = models.vit_b_16(pretrained=1),img_path = str,save_path = str):
 #pip install grad-cam
    def reshape_transform(tensor, height=14, width=14):
        '''
        不同参数的Swin网络的height和width是不同的，具体需要查看所对应的配置文件yaml
        height = width = config.DATA.IMG_SIZE / config.MODEL.NUM_HEADS[-1]
        比如该例子中IMG_SIZE: 224  NUM_HEADS: [4, 8, 16, 32]
        height = width = 224 / 32 = 7
        '''

        
        
        # # Flatten last two dimensions of input tensor
        # tensor = tensor.view(1, -1, 768)

        # # Reshape flattened tensor into output shape (1, 49, 768)
        # tensor = tensor.view(1, 49, 768)
        tensor = tensor[:, :-1  , :]
        ##print(tensor.shape)
        ###print(result)
        # .reshape(tensor.size(0),
        #     height, width, tensor.size(2))
        tensor = torch.tensor(tensor)
        
        #tensor = torch.randn((1, 196, 768))
        torch.set_printoptions(precision=4, sci_mode=True)

        tensor = tensor.reshape(1, 196, 768)
        ##print(tensor,'asvd')
        tensor = tensor.view(1, height ,  width , 768)

        # Bring the channels to the first dimension,
        # like in CNNs.
        ###print(result)
        tensor = tensor.transpose(2, 3).transpose(1, 2)
        
        return tensor


   # model.eval()
    # model = models.vit_b_16(pretrained=1)
    target_layers = [model.encoder] # 拿到最后一个层结构

    #print(target_layers)
    data_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224, 224)),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    import cv2
    img = cv2.resize(img, (224, 224))
 
    input_tensor = data_transform(Image.open(img_path)).unsqueeze(0)
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    predicted = torch.max(model(data_transform(Image.open(img_path).convert('RGB')).unsqueeze(0)).data, 1)[1].item()
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  reshape_transform = reshape_transform)
    
    #print(predicted)
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(predicted)],eigen_smooth=1,
                        aug_smooth=1)  # 实例化
     # 計算 array 中值為 0 的數量
    zero_count = np.sum(grayscale_cam == 0)
    #print("Number of zeros:", zero_count)

    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    # from pytorch_grad_cam.utils.image import show_cam_on_image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉為 RGB
    img = img / 255.0  # 縮放範圍到 [0, 1]
    img = img.astype(np.float32)  # 確保為 float32 型態
    #print(grayscale_cam)

    # 計算 array 中值為 0 的數量
    zero_count = np.sum(grayscale_cam == 0)
    #print("Number of zeros:", zero_count)
    visualization = show_cam_on_image(img.astype(dtype=np.float32),  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.savefig(save_path)
    plt.close('all') 
 
# model = models.vit_b_16(pretrained=1)
# img_path = "cats_and_dogs/dogs/dog.19.jpg"
# ViT_Heatmap(model,img_path)
# model = CustomModel(models.vit_b_16(pretrained=1))(torch.randn(1,3,224,224))