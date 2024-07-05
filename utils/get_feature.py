import torchvision
from torchvision import transforms
import h5py
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.CustomDataset import customDataset

torch.set_num_threads(1)
              
def vit_get_feature(pred_path,save_folder,model_path,k):
    #print("vit_get_feature_fold_" + str(k))
    def vit_feature(img_path):
        '''
        extract feature from an image
        '''
        # image ->(3,224,224)
        transform = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        # use path open image
        img = (img_path)#.convert('RGB')
        #img = transform(img)
        #通道多一條
        #img = torch.unsqueeze(img, dim=0)
        
        CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if CUDA else "cpu")
        img = img.to(device)
        ###print(img.shape)
        # create model
        #model = torch.load(model_path)
        model.eval()
        
        
        with torch.no_grad():
            x = model._process_input(img)
            n = x.shape[0]
            batch_class_token = model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = model.encoder(x)
            x = x[:, 0]
            featuree = x.view(768).cpu().numpy()
            ##print(featuree.shape)
            return featuree
    '''
    extract feature from all image
    '''

    if model_path =='':
        model = torchvision.models.vit_b_16(weights=1)
    else:
        model = torch.load(model_path)
    model.cuda()
    
   
    img_list = []
    feature_list = []
    input_csv_path = str(pred_path).split('*')[1]
    df = pd.read_csv(input_csv_path)
    train_loader = DataLoader(customDataset(df, shuffle = False,pathColumn0=1), batch_size=1)
    mission = tqdm(total=len(train_loader.dataset))
    for batch_idx, (name,data, target) in enumerate(train_loader):

        img_list.append(name)
        img_feature = vit_feature(data)
        feature_list.append(img_feature)
        mission.update()

        ##print(save_folder+'\\'+pred_path.split('_')[-2].split('/temp/')[-1]+'_data.h5',batch_idx)
                
    #return 0
    #write in h5
    with h5py.File(save_folder+'\\'+pred_path.split('_')[-2].split('/temp/')[-1]+'_data.h5', 'w') as f:
        f.create_dataset("path", data=img_list)
        f.create_dataset("feature", data=feature_list)

def swin_get_feature(pred_path,save_folder,model_path,k):
        #print('swin_get_feature_fold_' + str(k))
        
        def swin_feature(img_path):
            transform = transforms.Compose([
                    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
            ]) 
            #img = Image.open(img_path)#.convert('RGB')
            img = img_path
            #img = torch.unsqueeze(img, dim=0)
            CUDA = torch.cuda.is_available()
            device = torch.device("cuda" if CUDA else "cpu")
            img = img.to(device)
            # create model
            
            model.eval()
            with torch.no_grad():
                #https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029/2
                input = model.features(img)
                #7,7,1024 -> 1,1,1024
                avgPooll = nn.AdaptiveAvgPool2d(1)
                input = torch.transpose(input, 1, 3)#把通道维放到最后
                output = avgPooll(input)
                #swin b 1024 features 1,1,1024-> 1024
                featuree = output.view(1024).cpu().numpy()
                return featuree
        if model_path == '':
            model = torchvision.models.swin_v2_b(1)
        else:
            model = torch.load(model_path)
        img_list = []
        feature_list = []
        input_csv_path = str(pred_path).split('*')[1]
        df = pd.read_csv(input_csv_path)
        train_loader = DataLoader(customDataset(df, shuffle = False,pathColumn0=1), batch_size=1)
        mission = tqdm(total=len(train_loader.dataset))
        for batch_idx, (name,data, target) in enumerate(train_loader):
            mission.update()
        # img_path = pred_path+"\\"+file_class+"\\"+filename
            img_list.append(name)
            img_feature = swin_feature(data)
            feature_list.append(img_feature)
            #c.append(np.append(file_class,img_feature))
            #count += 1
            ##print(save_folder+'\\'+pred_path.split('_')[-2].split('/temp/')[-1]+'_data.h5',batch_idx)
           
        
                    
        #return 0
        #write in h5
        with h5py.File(save_folder+'\\'+pred_path.split('_')[-2].split('/temp/')[-1]+'_data.h5', 'w') as f:
            f.create_dataset("path", data=img_list)
            f.create_dataset("feature", data=feature_list)

def densenet_get_feature(pred_path,save_folder,model_path,k):
    #print('densenet_get_feature_fold_' + str(k))
    # model = torch.load(model_path)
    
    def densenet_feature(img_path):
        transform = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
        ]) 
        img = (img_path)#.convert('RGB')
        # img = transform(img)
        # img = torch.unsqueeze(img, dim=0)
        CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if CUDA else "cpu")
        img = img.to(device)
        # create model
        
       
        with torch.no_grad():
            input = model.features(img)
            avgPooll = nn.AdaptiveAvgPool2d(1)
            output = avgPooll(input)
            output = torch.transpose(output, 1, 3)#把通道维放到最后
            featuree = output.view(1920).cpu().numpy()
            ##print(featuree.shape)
            return featuree
    img_list = []
    feature_list = []
    if model_path =='':
             model = torchvision.models.densenet201(pretrained=1)
    else:
            model = torch.load(model_path)
    model.cuda()
    model.eval()
    img_list = []
    feature_list = []
    input_csv_path = str(pred_path).split('*')[1]
    df = pd.read_csv(input_csv_path)
    train_loader = DataLoader(customDataset(df, shuffle = False,pathColumn0=1), batch_size=1)
    
    mission = tqdm(total=len(train_loader.dataset))
    for batch_idx, (name,data, target) in enumerate(train_loader):
        mission.update()
       # img_path = pred_path+"\\"+file_class+"\\"+filename
        img_list.append(name)
        img_feature = densenet_feature(data)
        feature_list.append(img_feature)
        #c.append(np.append(file_class,img_feature))
        #count += 1
        ##print(save_folder+'\\'+pred_path.split('_')[-2].split('/temp/')[-1]+'_data.h5',batch_idx)
        
    #return 0
    #write in h5
    with h5py.File(save_folder+'\\'+pred_path.split('_')[-2].split('/temp/')[-1]+'_data.h5', 'w') as f:
        f.create_dataset("path", data=img_list)
        f.create_dataset("feature", data=feature_list)
