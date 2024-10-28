import torchvision
import torch.nn as nn

import torch
from functools import partial
from torchvision.models.vision_transformer import _vision_transformer,Encoder

class CBAM(torch.nn.Module):
    def __init__(self, channel, reduction=3):
        super(CBAM, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc1 = torch.nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # channel attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attention = self.sigmoid(avg_out + max_out)
        x = x * attention

        # spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * attention

        return x
        
class PViT(torch.nn.Module):
    def __init__(self,cbam = False):
        super(PViT, self).__init__()
       
        self.vit = _vision_transformer(
        patch_size=2**4,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=None,
        progress=True,
        )
        
        self.vit.encoder0 = Encoder(
            seq_length=98+1,
            num_layers=3,
            num_heads=12,
            hidden_dim=768*2,
            mlp_dim=3072,
            dropout=0.1,
            attention_dropout=0.1,
            norm_layer= partial(torch.nn.LayerNorm, eps=1e-6),
        )
        self.vit.encoder1 = Encoder(
            seq_length=196+1,
            num_layers=12,
            num_heads=12,
            hidden_dim=768*2,
            mlp_dim=3072, 
            dropout=0.1,
            attention_dropout=0.1,
            norm_layer= partial(torch.nn.LayerNorm, eps=1e-6),
        )
        self.vit.class_token = torch.nn.Parameter(torch.zeros(1, 1, 768*2))

        self.MLP = torch.nn.Linear(in_features=768*2, out_features=2)
        self.cbam_x = CBAM(3)
        self.cbam_y = CBAM(3)
        self.cbam = cbam
       
    def forward(self, x,y):
        if self.cbam:
            x = self.cbam_x(x)
            y = self.cbam_y(y)

        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)

        x = self.vit._process_input(x)
        y = self.vit._process_input(y)

        x = torch.cat([x[:, :],y[:, :]], dim=2)
        x = torch.cat([batch_class_token, x], dim=1) 
        x = self.vit.encoder1(x)
        x = self.MLP(x[:, 0])

        return x

def model_choose(model_name, num_classes, pretrain):
    '''
    宣告model 
    NetWork有:
        DenseNet -> 'densenet'
        Vision Transformer  -> 'vit'
        Swin Transformer -> 'swin'
        用str的方式輸入。
    '''
    def vit_model(num_classes, pretrain):
        # Download pretrained ViT weights and model
        vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT if pretrain else False
        pretrained_vit = torchvision.models.vit_b_16(weights=vit_weights)
        
        # Freeze all layers in pretrained ViT model 
        for param in pretrained_vit.parameters():
            param.requires_grad = True
        
        # Update the preatrained ViT head 
        embedding_dim = 768  # ViT_Base 16*16*3=768
        pretrained_vit.heads = nn.Linear(in_features=embedding_dim, out_features=num_classes)
        
        return pretrained_vit
    
    def swin_model(num_classes, pretrain):
        # 載入預訓練參數
        swin_weights = torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1 if pretrain else None
        model = torchvision.models.swin_v2_b(weights=swin_weights)
        
        for param in model.parameters():
            param.requires_grad = True
        
        embedding_dim = 1024  # 根據swin版本不同而改變 S-> 768 B->1024
        model.head = nn.Linear(in_features=embedding_dim, out_features=num_classes)
        
        return model

    def densenet201(num_classes, pretrain):
        densenet201 = torchvision.models.densenet201(pretrained=pretrain)
        for param in densenet201.parameters():
            param.requires_grad = True
        densenet201.classifier = nn.Linear(1920, num_classes)
        return densenet201

    model = None
    if model_name == "vit":
        model = vit_model(num_classes, pretrain)
    elif model_name == "swin":
        model = swin_model(num_classes, pretrain)
    elif model_name == "densenet":
        model = densenet201(num_classes, pretrain)
    elif model_name == 'multi_input_vit':
        model = PViT(False)
    elif model_name == 'multi_input_vit_cbam':
        model = PViT(True)



    return model
