import cv2
import torch,torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import os
from PIL import Image

def Swin_Heatmap(model=torchvision.models.swin_b(1) ,img_path = str,save_path = str):
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--use-cuda', action='store_true', default=False,
                            help='Use NVIDIA GPU acceleration')
        parser.add_argument(
            '--image-path',
            type=str,
            default='./images/both.png',
            help='Input image path')
        parser.add_argument('--aug_smooth', action='store_true',
                            help='Apply test time augmentation to smooth the CAM')
        parser.add_argument(
            '--eigen_smooth',
            action='store_true',
            help='Reduce noise by taking the first principle componenet'
            'of cam_weights*activations')

        args = parser.parse_args()
        args.use_cuda = args.use_cuda and torch.cuda.is_available()
        # if args.use_cuda:
        #     print('Using GPU for acceleration')
        # else:
        #     print('Using CPU for computation')

        return args

    def reshape_transform(tensor, height=7, width=7):
        '''
        不同参数的Swin网络的height和width是不同的，具体需要查看所对应的配置文件yaml
        height = width = config.DATA.IMG_SIZE / config.MODEL.NUM_HEADS[-1]
        比如该例子中IMG_SIZE: 224  NUM_HEADS: [4, 8, 16, 32]
        height = width = 224 / 32 = 7
        '''
        #print(tensor.shape)
        # Flatten last two dimensions of input tensor
        tensor = tensor.view(1, -1, 1024)

        # Reshape flattened tensor into output shape (1, 49, 768)
        tensor = tensor.view(1, 49, 1024)
        result = tensor.reshape(tensor.size(0), height ,  width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        #print(result)
        return result
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.
    """
    args = get_args()
   # print(model)
    model.eval()

    # if args.use_cuda:
    #     model = model
	
    # 作者这个地方应该写错了，误导了我很久，输出model结构能发现正确的target_layers应该为最后一个stage后的LayerNorm层
    # target_layers = [model.layers[-1].blocks[-1].norm2]
    target_layers = [model.norm]
	
    # transformer会比CNN额外多输入参数reshape_transform
    cam = GradCAM(model=model, target_layers=target_layers,
                   reshape_transform=reshape_transform)
	
    # 保证图片输入后为RGB格式，cv2.imread读取后为BGR
    rgb_img = cv2.imread(img_path)[:, :, ::-1]
    
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
    
	

    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5]*3, [0.5]*3)
            ])
    img = Image.open(img_path).convert('RGB')
    img = torch.unsqueeze(transform(img), dim=0)
    output = model(img)
    #predicted = torch.max(output.data)
    predicted = torch.max(output.data, 1)[1].item()
    #class_map = {0: "Chihuahua", 1: "tobby cat",2: "Chihuahua", 3: "tobby cat",4: "Chihuahua", 5: "tobby cat",6: "Chihuahua"}
    #class_id = 0
    #class_name = class_map[class_id]
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(predicted)],
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.imshow(cam_image)
    plt.savefig(save_path) 
    plt.close('all')

 
# # model = models.vit_b_16(pretrained=1)
# img_path = "cats_and_dogs/dogs/dog.19.jpg"

# swin_cam(img_path)