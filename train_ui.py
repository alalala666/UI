import tkinter as tk,os
from tkinter import ttk
import xml.etree.ElementTree as ET
from tkinter import filedialog
import subprocess,os
import tkinter.messagebox as messagebox
try:
    from screeninfo import get_monitors
except:
    os.system("pip install screeninfo")
    from screeninfo import get_monitors

root = tk.Tk()
# def project_name_listener(*args):#root.after(10, delayed_processing)  # 设置延迟为1秒

# def project_name_listener(*args):
#     # print("Entry A changed to:", project_name.get())
#     # print('us' == project_name.get())

#     process = subprocess.check_output("conda info --envs", stdin=subprocess.PIPE, shell=True)
#     if  (('\\'+project_name.get()+'\\') in str(process)) and (project_name.get()!= ''):
#         switch_env()
#         train_btn.config(state="normal")
#         check_env_btn.config(state="normal")
#         retireve_btn.config(state="normal")
#         inference_btn.config(state="normal")
#         anwser_btn.config(state="normal")
#         ml_btn.config(state="normal")
#     else: 
#         train_btn.config(state="disabled")
#         check_env_btn.config(state="disabled")
#         retireve_btn.config(state="disabled")
#         inference_btn.config(state="disabled")
#         anwser_btn.config(state="disabled")
#         ml_btn.config(state="disabled")

def get_Current_virtual_environment():
    '''
    返回當前的虛擬機名稱
    '''
    process = subprocess.check_output("conda info --envs", stdin=subprocess.PIPE, shell=True)
    #print((process))
    Current_virtual_environment = str(process).split('*')[0]
    Current_virtual_environment = Current_virtual_environment.split('\\n')[-1]
    Current_virtual_environment = Current_virtual_environment.split(' ')[0]
    #print('當前虛擬環境 : ',Current_virtual_environment)
    return Current_virtual_environment
# print("當前虛擬環境 :　",get_Current_virtual_environment())
def check_in_env(env = str):
    '''
    確認虛擬環境是否存在
    '''
    process = str(subprocess.check_output("conda info --envs", stdin=subprocess.PIPE, shell=True))
    return env in str(process)

def change_setting():
    # 解析XML文件
    tree = ET.parse(setting.get())
    root = tree.getroot()

    # 找到<k>节点并修改其文本为5
    for k_node in root.iter('project_name'):
        k_node.text = project_name.get()

    # 保存修改后的XML文件
    tree.write(setting.get(), encoding='utf-8', xml_declaration=True)

def switch_env():
    '''
    切換ENV
    '''
    change_setting()
    Current_virtual_environment = get_Current_virtual_environment()
    process = subprocess.check_output("conda info --envs", stdin=subprocess.PIPE, shell=True)
    if ((project_name.get())!= Current_virtual_environment) and (('\\'+project_name.get()+'\\') in str(process)):
        text = ("當前虛擬環境 :　"+get_Current_virtual_environment())
        text1 = ("切換至虛擬環境 :　"+project_name.get())
        messagebox.showinfo("提示", '虛擬環境已存在，系統將自動切換虛擬環境，若有操作需要再執行一次。 \n\n'+text + '\n' +text1)
        root.destroy()
        os.system("activate "+project_name.get()+" && python train_ui.py")

def make_env():
  '''
  創建虛擬環境
  '''
  command = "conda create --name "+project_name.get()+" python=3.9"
  process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)
  process.communicate(input=b'y\n')
  change_setting()
  os.system('conda list')

  command = "activate "+project_name.get()+" && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
  os.system(command)

  command = "activate "+project_name.get()+" && pip install h5py"
  os.system(command)

  command = "activate "+project_name.get()+" && pip install scipy"
  os.system(command)
  
  command = "activate "+project_name.get()+" && pip install matplotlib"
  os.system(command)

  command = "activate "+project_name.get()+" && pip install pandas"
  os.system(command)

  command = "activate "+project_name.get()+" && pip install scikit-learn"
  os.system(command)
  os.system('conda list')
  command = "activate "+project_name.get()+" && pip install tensorboard"
  os.system(command)

  command = "activate "+project_name.get()+" && pip install torchmetrics"
  os.system(command)

  command = "activate "+project_name.get()+" && pip install tqdm"
  os.system(command)

  command = "activate "+project_name.get()+" && pip install screeninfo"
  os.system(command)

  root.destroy()
  os.system("activate "+project_name.get()+" && python train_ui.py")

def make_env2():
  '''
  創建虛擬環境 測試用
  '''
  command = "conda create --name "+project_name.get()+" python=3.9"
  process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)
  process.communicate(input=b'y\n')
  os.system('conda list')
  change_setting()

#   command = "activate "+project_name.get()+" && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
#   os.system(command)

#   command = "activate "+project_name.get()+" && pip install h5py"
#   os.system(command)

#   command = "activate "+project_name.get()+" && pip install scipy"
#   os.system(command)
  
#   command = "activate "+project_name.get()+" && pip install matplotlib"
#   os.system(command)

#   command = "activate "+project_name.get()+" && pip install pandas"
#   os.system(command)

#   command = "activate "+project_name.get()+" && pip install scikit-learn"
#   os.system(command)
#   os.system('conda list')
#   command = "activate "+project_name.get()+" && pip install tensorboard"
#   os.system(command)

#   command = "activate "+project_name.get()+" && pip install torchmetrics"
#   os.system(command)

#   command = "activate "+project_name.get()+" && pip install tqdm"
#   os.system(command)

#   command = "activate "+project_name.get()+" && pip install screeninfo"
#   os.system(command)

  root.destroy()
  os.system("activate "+project_name.get()+" && python train_ui.py")

def check_env_lib2():
    '''
    check lib 是否安裝完成 測試用
    '''
    env_check = 'lib OK!'
    if not check_env():
        import tkinter.messagebox as messagebox
        messagebox.showinfo("提示", "請先安裝虛擬環境")
        return 'please create env'
    switch_env()
    try:
        import tkinter as tk
        import os
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np
        from sklearn.model_selection import  StratifiedKFold
        from sklearn.metrics import accuracy_score, confusion_matrix
        import pandas
        import csv
        import csv
        import numpy as np,pandas
        from sklearn import svm
        from sklearn.model_selection import  StratifiedKFold
        from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import BaggingClassifier
        import numpy as np
        import csv
        import ast
        from collections import Counter
        import numpy as np
        import os  # 系统操作，如路径操作
        import csv  # 用于CSV文件操作
        import time  # 用于时间相关功能
        import h5py  # 用于HDF5文件操作
        import numpy as np  # NumPy库，用于数值计算
        import matplotlib.pyplot as plt  # 用于绘制图表
        from PIL import Image  # Python Imaging Library，用于图像处理
        from scipy.stats import chi2_contingency  # 用于卡方检验
        from sklearn.metrics import confusion_matrix  # 用于混淆矩阵计算
        import torch  # PyTorch深度学习框架
        import torchvision  # PyTorch官方的计算机视觉库
        from torchvision import transforms  # 用于数据预处理和数据集
        from torch.autograd import Variable  # 用于自动求导
        import torch.nn as nn  # PyTorch中的神经网络模块
        from torchmetrics.functional.classification import multiclass_auroc  # 用于多类别AUC计算
        from torch.utils.tensorboard import SummaryWriter  # 用于TensorBoard日志
        from torch.utils.tensorboard import SummaryWriter  # 用于TensorBoard日志
        from torchmetrics.functional.classification import multiclass_auroc  # 用于多类别AUC计算
        from torch.autograd import Variable  # 用于自动求导
        import torch.nn as nn  # PyTorch中的神经网络模块
        import torch  # PyTorch深度学习框架
        import warnings  # 用于警告管理
        from utils.get_feature import swin_get_feature, vit_get_feature, densenet_get_feature  # 从自定义模块导入特征提取函数
        from utils.model_choose import model_choose  # 从自定义模块导入模型选择函数
        from utils.vaild import evaluation_index  # 从自定义模块导入评估指标计算函数
        from utils.draw_roc import draw_roc  # 从自定义模块导入绘制ROC曲线函数
        from utils.CustomDataset import customDataset,two_img_customDataset  # 从自定义模块导入自定义数据集类
        from utils.cal_time import cal_time  # 从自定义模块导入时间计算函数
        from torch.utils.data import DataLoader
        import pandas as pd
        import ast
        from tqdm import tqdm
        import tkinter as tk
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import torchvision
        import torch.nn as nn
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        import torch
        from functools import partial
        from torchvision.models.vision_transformer import _vision_transformer,Encoder
        import torchvision
        from torchvision import transforms
        import h5py
        import torch.nn as nn
        import torch
        import pandas as pd
        from torch.utils.data import DataLoader
        from utils.CustomDataset import customDataset
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        import torch
        from PIL import Image
        env_check = 'lib OK!'
        print('lib OK!')
        # rooot = tk.Tk()
        # rooot.title("env check")
        # rooot.geometry("200x150")

        # tk.Label(rooot, font=("Times New Roman",18),text='\n'+env_check+'\n').pack()

        # greet_button = tk.Button(rooot, text="close", font=("Times New Roman",18),command=lambda: rooot.destroy())
        # greet_button.pack()

        # rooot.mainloop()
        return True
    except:
        from tkinter import messagebox
        train_btn.config(state="disabled")
        check_env_btn.config(state="disabled")
        retireve_btn.config(state="disabled")
        inference_btn.config(state="disabled")
        anwser_btn.config(state="disabled")
        ml_btn.config(state="disabled")
        messagebox.showinfo("提示", "環境安裝未完成，請按下Create Environment，系通將自動重新安裝。")
        return False
    # try:
    #     os.system("activate " + project_name.get() +" && python check_env.py")
    # except:
    #     messagebox.showinfo("提示","請重新安裝環境")
    #     return False
    
def check_env_lib():
    '''
    check lib 是否安裝完成
    '''
    #print('check_env_lib')
    env_check = 'lib OK!'
    if not check_env():
        messagebox.showinfo("提示", "請先安裝虛擬環境")
        return 'please create env'
    switch_env()
    from tqdm import tqdm
    print('檢查lib')
    ##mission  = tqdm(total=49)
    try:
        import tkinter as tk
        #mission.update()
        import os
        #mission.update()
        from sklearn.ensemble import BaggingClassifier
        #mission.update()
        from sklearn.tree import DecisionTreeClassifier
        #mission.update()
        import numpy as np
        #mission.update()
        from sklearn.model_selection import  StratifiedKFold
        #mission.update()
        from sklearn.metrics import accuracy_score, confusion_matrix
        #mission.update()
        import pandas
        #mission.update()
        import csv
        #mission.update()
        import csv
        #mission.update()
        import numpy as np,pandas
        #mission.update()
        from sklearn import svm
        #mission.update()
        from sklearn.model_selection import  StratifiedKFold
        #mission.update()
        from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
        #mission.update()
        from sklearn.linear_model import LogisticRegression
        #mission.update()
        from sklearn.neural_network import MLPClassifier
        #mission.update()
        from sklearn.neighbors import KNeighborsClassifier
        #mission.update()
        from sklearn.ensemble import BaggingClassifier
        #mission.update()
        import numpy as np
        
        #mission.update()
        import csv
        #mission.update()
        import ast
        #mission.update()
        from collections import Counter
        #mission.update()
        import numpy as np
        #mission.update()
        import os  # 系统操作，如路径操作
        #mission.update()
        import csv  # 用于CSV文件操作
        #mission.update()
        import time  # 用于时间相关功能
        #mission.update()
        import h5py  # 用于HDF5文件操作
        #mission.update()
        import numpy as np  # NumPy库，用于数值计算
        #mission.update()
        import matplotlib.pyplot as plt  # 用于绘制图表
        #mission.update()
        from PIL import Image  # Python Imaging Library，用于图像处理
        #mission.update()
        from scipy.stats import chi2_contingency  # 用于卡方检验
        #mission.update()
        from sklearn.metrics import confusion_matrix  # 用于混淆矩阵计算
        #mission.update()
        import torch  # PyTorch深度学习框架
        #mission.update()
        import torchvision  # PyTorch官方的计算机视觉库
        #mission.update()
        from torchvision import transforms  # 用于数据预处理和数据集
        #mission.update()
        from torch.autograd import Variable  # 用于自动求导
        #mission.update()
        import torch.nn as nn  # PyTorch中的神经网络模块
        #mission.update()
        from torchmetrics.functional.classification import multiclass_auroc  # 用于多类别AUC计算
        #mission.update()
        from torch.utils.tensorboard import SummaryWriter  # 用于TensorBoard日志
        #mission.update()
        from torch.utils.tensorboard import SummaryWriter  # 用于TensorBoard日志
        #mission.update()
        from torchmetrics.functional.classification import multiclass_auroc  # 用于多类别AUC计算
        #mission.update()
        from torch.autograd import Variable  # 用于自动求导
        #mission.update()
        import torch.nn as nn  # PyTorch中的神经网络模块
        #mission.update()
        import torch  # PyTorch深度学习框架
        #mission.update()
        import warnings  # 用于警告管理
        #mission.update()
        from utils.get_feature import swin_get_feature, vit_get_feature, densenet_get_feature  # 从自定义模块导入特征提取函数
        #mission.update()
        from utils.model_choose import model_choose  # 从自定义模块导入模型选择函数
        #mission.update()
        from utils.vaild import evaluation_index  # 从自定义模块导入评估指标计算函数
        #mission.update()
        from utils.draw_roc import draw_roc  # 从自定义模块导入绘制ROC曲线函数
        #mission.update()
        from utils.CustomDataset import customDataset,two_img_customDataset  # 从自定义模块导入自定义数据集类
        #mission.update()
        from utils.cal_time import cal_time  # 从自定义模块导入时间计算函数
        #mission.update()
        from torch.utils.data import DataLoader
        #mission.update()
        import pandas as pd
        #mission.update()
        import ast
        #mission.update()
        import tkinter as tk
        #mission.update()
        import numpy as np
        #mission.update()
        from sklearn.metrics import confusion_matrix
        #mission.update()
        import torchvision
        #mission.update()
        import torch.nn as nn
        #mission.update()
        from sklearn.metrics import roc_curve, auc
        #mission.update()
        import matplotlib.pyplot as plt
        #mission.update()
        import torch
        #mission.update()
        from functools import partial
        #mission.update()
        from torchvision.models.vision_transformer import _vision_transformer,Encoder
        #mission.update()
        import torchvision
        #mission.update()
        from torchvision import transforms
        #mission.update()
        import h5py
        #mission.update()
        import torch.nn as nn
        #mission.update()
        import torch
        #mission.update()
        import pandas as pd
        #mission.update()
        from torch.utils.data import DataLoader
        #mission.update()
        from utils.CustomDataset import customDataset
        #mission.update()
        from torch.utils.data import Dataset, DataLoader
        #mission.update()
        from torchvision import transforms
        #mission.update()
        import torch
        #mission.update()
        from PIL import Image
        #mission.update()


        # env_check = 'lib OK!'
        # print(env_check)
        # rooot = tk.Tk()
        # rooot.title("env check")
        # rooot.geometry("200x150")

        # tk.Label(rooot, font=("Times New Roman",18),text='\n'+env_check+'\n').pack()

        # greet_button = tk.Button(rooot, text="close", font=("Times New Roman",18),command=lambda: rooot.destroy())
        # greet_button.pack()

        # rooot.mainloop()
        return True
    except:
        from tkinter import messagebox
        train_btn.config(state="disabled")
        check_env_btn.config(state="disabled")
        retireve_btn.config(state="disabled")
        inference_btn.config(state="disabled")
        anwser_btn.config(state="disabled")
        ml_btn.config(state="disabled")
        messagebox.showinfo("提示", "環境安裝未完成，請按下Create Environment，系通將自動重新安裝。")
        
        return False
    # try:
    #     os.system("activate " + project_name.get() +" && python check_env.py")
    # except:
    #     messagebox.showinfo("提示","請重新安裝環境")
    #     return False
    
def check_env(): 
    '''
    用來決定RUN是否GRAY掉
    '''
    
    #print("check_env")
    
    change_setting()
    # process用來看虛擬環境有那些
    process = subprocess.check_output("conda info --envs", stdin=subprocess.PIPE, shell=True)

    if  (('\\'+project_name.get()+'\\') in str(process)):
        
        Current_virtual_environment = get_Current_virtual_environment()
        if Current_virtual_environment != project_name.get():
            switch_env()
        return True
       
        # train_btn.config(state="normal")
        # check_env_btn.config(state="normal")
        # retireve_btn.config(state="normal")
        # inference_btn.config(state="normal")
        # anwser_btn.config(state="normal")
        # ml_btn.config(state="normal")
        # return True
    #若不在就gray掉
    # else: 
    #     train_btn.config(state="disabled")
    #     check_env_btn.config(state="disabled")
    #     retireve_btn.config(state="disabled")
    #     inference_btn.config(state="disabled")
    #     anwser_btn.config(state="disabled")
    #     ml_btn.config(state="disabled")
    return False
    
def test():

    # 建立彈跳視窗
    top = tk.Toplevel(root)

    # 設定彈跳視窗的屬性
    top.geometry("300x200")
    top.title("彈跳視窗")

        # 顯示彈跳視窗ㄜ
    top.mainloop()

def ML():
    if not check_env():
        messagebox.showinfo("提示", "請先安裝虛擬環境")
        return 'please create env'
    if not  check_env_lib():return 'please create env'
    from train import CBMIR  
    cb = CBMIR()
    cb.retireve_fold = int(str(retrival_model_path.get()).split('/')[-1][0])
    cb.data_path = data_path.get()
    cb.query_path = query_path.get()
    cb.target_path = target_path.get()
    cb.project_name=save_folder.get()
    cb.model_listt = []
    if ('swin')==str(retrival_model_path.get()).split('/')[-2]:
        cb.model_listt.append('swin')
    if('vit'==str(retrival_model_path.get()).split('/')[-2]):
        cb.model_listt.append('vit')
    if('densenet'==str(retrival_model_path.get()).split('/')[-2]):
        cb.model_listt.append('densenet')
    # if path.get() == 'all':
    #     cb.path_listt = []
    #     # for i in os.listdir(data_path.get()):
    #     #     cb.path_listt.append(i)
    #     cb.path_listt= [(cb.data_path.split('/')[-1])]
    # else:
    #     # cb.path_listt = []
    #     # for i in path.get() :
    #     #     if i == ',':
    #     #         continue
    #     #     else:
    #     #         cb.path_listt.append(i)
    #     cb.path_listt= [(cb.data_path.split('/')[-1])]
    cb.path_listt= [(cb.data_path.split('/')[-1])]
    cb.train_typee = []
    cb.K = 5
    if(str(retrival_model_path.get()).split('/')[-4]=='finetune'):
        cb.train_typee.append('finetune')
    else:
        cb.train_typee.append('trainfromscratch')

    ML_models = []
    if (SVM.get())==1:
        ML_models.append('SVM')
    if(Subspace_KNN.get()==1):
        ML_models.append('Subspace_KNN')
    if(Random_Forest.get()==1):
        ML_models.append('Random_Forest')
    if(ANN.get()==1):
        ML_models.append('ANN')
    cb.ML(ML_path.get(),ML_models)
    print("--------------------------------------------------")
    print("ML fin")
    print("--------------------------------------------------")

def Retrieval_Result_Features():
    if not check_env():
        messagebox.showinfo("提示", "請先安裝虛擬環境")
        return 'please create env'
   
    if not  check_env_lib():return 'please create env'

        
    from train import CBMIR  
    cb = CBMIR()#E:\S_UI\setting_result\finetune\S\vit\0.pth
    cb.retireve_fold = int(str(retrival_model_path.get()).split('/')[-1][0])
    cb.data_path = data_path.get()
    cb.query_path = query_path.get()
    cb.target_path = target_path.get()
    cb.project_name=save_folder.get()
    cb.model_listt = []
    if ('swin')==str(retrival_model_path.get()).split('/')[-2]:
        cb.model_listt.append('swin')
    if('vit'==str(retrival_model_path.get()).split('/')[-2]):
        cb.model_listt.append('vit')
    if('densenet'==str(retrival_model_path.get()).split('/')[-2]):
        cb.model_listt.append('densenet')
    cb.path_listt=[(cb.data_path.split('/')[-1])]
    # if path.get() == 'all':
    #     cb.path_listt = []
    #     for i in os.listdir(data_path.get()):
    #         cb.path_listt.append(i)
    # else:
    #     cb.path_listt = []
    #     for i in path.get() :
    #         if i == ',':
    #             continue
    #         else:
    #             cb.path_listt.append(i)

    cb.train_typee = []
    
    if(str(retrival_model_path.get()).split('/')[-4]=='finetune'):
        cb.train_typee.append('finetune')
    else:
        cb.train_typee.append('trainfromscratch')
   
    cb.batch_size = int(batch_size.get())
    cb.K = 5#int(k.get())
    print(cb.data_path,cb.project_name,cb.model_listt,cb.train_typee,cb.K,cb.batch_size,cb.num_epochs,cb.path_listt)
    #cb.foldd(cb.data_path)
    # cb.path_listt = [i for i in os.listdir(data_path.get())]
    cb.path_listt= [(cb.data_path.split('/')[-1])]
    cb.ML_answer = anwser.get()
    cb.make_ML_feature()
    print("--------------------------------------------------")
    print("Retrieval_Result_Features fin")
    print("--------------------------------------------------")

def retireve():
    # if not check_env():
    #     messagebox.showinfo("提示", "請先安裝虛擬環境")
    #     return 'please create env'
    #if not  check_env_lib():return 'please create env'
    from train import CBMIR  
    cb = CBMIR()#E:\S_UI\setting_result\finetune\S\vit\0.pth
    cb.retireve_fold = int(str(retrival_model_path.get()).split('/')[-1][0])
    cb.data_path = data_path.get()
    cb.query_path = query_path.get()
    cb.target_path = target_path.get()
    cb.project_name=save_folder.get()
    cb.model_listt = []
    if ('swin')==str(retrival_model_path.get()).split('/')[-2]:
        cb.model_listt.append('swin')
    if('vit'==str(retrival_model_path.get()).split('/')[-2]):
        cb.model_listt.append('vit')
    if('densenet'==str(retrival_model_path.get()).split('/')[-2]):
        cb.model_listt.append('densenet')
    #cb.path_listt= [(cb.data_path.split('/')[-1])]
    # if path.get() == 'all':
    #     cb.path_listt = []
    #     for i in os.listdir(data_path.get()):
    #         cb.path_listt.append(i)
    # else:
    #     cb.path_listt = []
    #     for i in path.get() :
    #         if i == ',':
    #             continue
    #         else:
    #             cb.path_listt.append(i)

    cb.train_typee = []
    
    if(str(retrival_model_path.get()).split('/')[-4]=='finetune'):
        cb.train_typee.append('finetune')
    else:
        cb.train_typee.append('trainfromscratch')
   
    cb.batch_size = int(batch_size.get())
    cb.K = 5#int(k.get())
    #print(cb.data_path,cb.project_name,cb.model_listt,cb.train_typee,cb.K,cb.batch_size,cb.num_epochs,cb.path_listt)
    #cb.foldd(cb.data_path)
    # cb.path_listt = [i for i in os.listdir(data_path.get())]
    cb.path_listt= [retrival_model_path.get().split('/')[-3]]#[(cb.data_path.split('/')[-1])]
    #cb.auto_train()
    cb.retireve()
    cb.make_query_img()
    print("--------------------------------------------------")
    print("retireve fin")
    print("--------------------------------------------------")

def train():
    #print("check_env")
    if not check_env():
        messagebox.showinfo("提示", "請先安裝虛擬環境")
        return 'please create env'
    print("檢查函式庫是否安裝完成，請稍後。")
    if not check_env_lib():return 'please create env'
    print("train")
    
    
    from train import CBMIR  
    cb = CBMIR()
    cb.input_two_img =Two_Image.get()
    cb.data_path = data_path.get()
    cb.data_path1 = data_path2.get()

    if epoch_lock.get():
        cb.max_epoch = int(Epoch.get())
        cb.earlystop_arg0 = 999
        cb.earlystop_arg1 = 999
    else:
        cb.max_epoch = 999
        cb.earlystop_arg0 = int(Earlystop_Epoch.get())
        cb.earlystop_arg1 = float(Earlystop_validate.get())
        
    cb.query_path = query_path.get()
    cb.target_path = target_path.get()
    cb.project_name=save_folder.get()

    cb.model_listt = []
    if (swinVar1.get())==1:
        cb.model_listt.append('swin')
    if(vitVar1.get()==1):
        if Two_Image.get():
            if Attention.get():
                cb.model_listt.append('multi_input_vit_cbam')
            else:
                cb.model_listt.append('multi_input_vit')
        else:
            cb.model_listt.append('vit')
    if(densenetVar1.get()==1):
        cb.model_listt.append('densenet')
    cb.path_listt= [(cb.data_path.split('/')[-1])] 
    # if path.get() == 'all':
    #     cb.path_listt = []
    #     for i in os.listdir(data_path.get()):
    #         cb.path_listt.append(i)
    # else:
    #     cb.path_listt = []
    #     for i in path.get() :
    #         if i == ',':
    #             continue
    #         else:
    #             cb.path_listt.append(i)
     
    
    cb.train_typee = []
    
    if(Finetune.get()==1):
        cb.train_typee.append('finetune')
    else:
        cb.train_typee.append('trainfromscratch')
    cb.batch_size = int(batch_size.get())
    cb.K = 5#int(k.get())

    cb.data_path = data_path.get()
    cb.project_name = save_folder.get()
    cb.test_mode =False
    cb.lr = float(Learning_rate.get())
    cb.optimizer = Optimizer.get()
    #cb.max_epoch = Epoch.get()
    cb.print_parameters()
    cb.auto_train()
    print("--------------------------------------------------")
    print("train fin")
    print("--------------------------------------------------")

def inference():
    if not check_env():
        messagebox.showinfo("提示", "請先安裝虛擬環境")
        return 'please create env'
    if not  check_env_lib():return 'please create env'
    from train import CBMIR  
    cb = CBMIR()
    cb.project_name = save_folder.get()
    cb.inference(inference_model_path.get(),inference_path.get())
    print("--------------------------------------------------")
    print("inference fin")
    print("--------------------------------------------------")
    return

def select_anwser_folder():
    folder_path = filedialog.askopenfilename()  # 打开对话框让用户选择文件夹
    if folder_path == '':return
    print("select_anwser_folder:", folder_path)
    anwser.set(folder_path)
    anwser_show.set('/'+folder_path+'/')

def select_target_folder():
    folder_path = filedialog.askdirectory()  # 打开对话框让用户选择文件夹
    if folder_path == '':return
    print("select_target_folder:", folder_path)
    target_path.set(folder_path)
    target_path_show.set('/'+folder_path+'/')

def select_retrival_model_path():
    folder_path = filedialog.askopenfilename()  # 打开对话框让用户选择文件夹
    if folder_path == '':return
    print("select_model_path:", folder_path)
    retrival_model_path.set(folder_path)

def select_inference_model_path():
    folder_path = filedialog.askopenfilename()  # 打开对话框让用户选择文件夹
    if folder_path == '':return
    print("select_model_path:", folder_path)
    inference_model_path.set(folder_path)

def select_inference_folder():
    folder_path = filedialog.askdirectory()  # 打开对话框让用户选择文件夹
    if folder_path == '':return
    print("select_inference_folder:", folder_path)
    inference_path.set(folder_path)

def select_ML_folder():
    folder_path = filedialog.askopenfilename()  # 打开对话框让用户选择文件夹
    if folder_path == '':return
    print("select_inference_folder:", folder_path)
    ML_path.set(folder_path)

def select_query_folder():
    folder_path = filedialog.askdirectory()  # 打开对话框让用户选择文件夹
    if folder_path == '':return
    print("select_query_folder:", folder_path)
    query_path.set(folder_path)
    query_path_show.set('/'+folder_path+'/')

def select_folder():
    folder_path = filedialog.askdirectory()  # 打开对话框让用户选择文件夹
    if folder_path == '':return
    print("select_folder:", folder_path)
    data_path.set(folder_path)
    data_path_show.set('/'+folder_path+'/')

def select_folder2():
    folder_path = filedialog.askdirectory()  # 打开对话框让用户选择文件夹
    if folder_path == '':return
    print("select_folder:", folder_path)
    data_path2.set(folder_path)
    data_path2_show.set('/'+folder_path+'/')

def browse_folder(): 
    folder_selected = filedialog.askopenfilename()
    if folder_selected == '':return
    
    setting.set(folder_selected)
    # setting_path = setting.get()
    print('select setting:'+setting.get())
    project_name.set(([heading.text for heading in ET.parse(setting.get()).getroot().iter('project_name')][0]))
    data_path.set([heading.text for heading in ET.parse(setting.get()).getroot().iter('dataset_path0')][0])
    data_path2.set([heading.text for heading in ET.parse(setting.get()).getroot().iter('dataset_path1')][0])

    swinVar1.set('swin' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('network')][0])
    vitVar1.set('vit' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('network')][0])
    densenetVar1.set('densenet' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('network')][0])
    Attention.set([heading.text for heading in ET.parse(setting.get()).getroot().iter('Attention')][0])

    scratch.set('trainfromscratch' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('train_type')][0])
    Finetune.set('finetune' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('train_type')][0])
    batch_size.set( [heading.text for heading in ET.parse(setting.get()).getroot().iter('batch_size')][0])
    
        
    Earlystop_Epoch.set([heading.text for heading in ET.parse(setting.get()).getroot().iter('Earlystop_Epoch')][0])
    Earlystop_validate.set([heading.text for heading in ET.parse(setting.get()).getroot().iter('Earlystop_validate')][0])

    
    #k.set( [heading.text for heading in ET.parse(setting.get()).getroot().iter('k')][0])
    save_folder.set([heading.text for heading in ET.parse(setting.get()).getroot().iter('project_name')][0])
    query_path.set(  [heading.text for heading in ET.parse(setting.get()).getroot().iter('query_dataset')][0])
    target_path.set(  [heading.text for heading in ET.parse(setting.get()).getroot().iter('target_dataset')][0])
    #retrieve_Var1.set( "True" == [heading.text for heading in ET.parse(setting.get()).getroot().iter('retrieve')][0])
  #  query_show.set("True" ==[heading.text for heading in ET.parse(setting.get()).getroot().iter('make_query_imgs')][0])
    anwser.set(  [heading.text for heading in ET.parse(setting.get()).getroot().iter('answer')][0])
    #inference.set( "True" == [heading.text for heading in ET.parse(setting.get()).getroot().iter('inference')][0])
    SVM.set('SVM' in[heading.text for heading in ET.parse(setting.get()).getroot().iter('model')][0])
    Random_Forest.set('Random_Forest' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('model')][0])
    Subspace_KNN.set('Subspace_KNN' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('model')][0])
    ANN.set('ANN' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('model')][0])
    
    root.update()
    check_env()
    
def check_two_image():
#  test()
 if  Two_Image.get():
  train_dataset_input2.config(state="normal")
  DenseNet_btn.config(state="disabled")
  densenetVar1.set(value=0)
  SwinViT_btn.config(state="disabled")
  swinVar1.set(value=0)
  Attention_btn.config(state="normal")
 
  
 else:
  train_dataset_input2.config(state="disabled")
  DenseNet_btn.config(state="normal")
  SwinViT_btn.config(state="normal")
  Attention_btn.config(state="disabled")
  Attention.set(False)

def set_epoch():
    if  epoch_lock.get():
        Earlystop_Epoch_Entry.config(state="disabled")
        Earlystop_validate_Entry.config(state="disabled")
        Earlystop_Epoch.set("")
        Earlystop_validate.set("")
        Epoch_Entry.config(state="normal")
    else:
        Earlystop_Epoch_Entry.config(state="normal")
        
        Earlystop_validate_Entry.config(state="normal")
        
        Epoch_Entry.config(state="disabled")
        Epoch.set('')
    
# 创建主窗口

root.title("UI")


frame_train1 = ttk.LabelFrame(root,text="")
frame_train1.grid(row=0, column=0, padx=0,columnspan=4)

setting_path = 'setting.xml'
setting = tk.StringVar(value=setting_path)
project_name = tk.StringVar(value=str([heading.text for heading in ET.parse(setting.get()).getroot().iter('project_name')][0]))
# project_name.trace_add("write", project_name_listener)


ttk.Label(frame_train1, font=("Times New Roman",14),text="　　　　　　　　　Settings").grid(row=0, column=3, padx=10)
tk.Button(frame_train1, font=("Times New Roman",14),text="　　選取　　", command=browse_folder).grid(row=0, column=4,sticky='en')
tk.Label(frame_train1, font=("Times New Roman",14),text='/'+setting.get()+'/').grid(row=0, column=5, padx=10)

ttk.Label(frame_train1, font=("Times New Roman",14),text="　　　　　　　　　　　　　　　　　　　　　　　　　").grid(row=1, column=6, padx=10,sticky='wn')


# def on_entry_complete(event):
#     change_setting()
#     check_env()


# project_name.get() = project_name.get()
# 最上方的 Project Name 输入
ttk.Label(frame_train1, font=("Times New Roman",14),text="　Project Name :　").grid(row=0, column=0,padx=0)
save_folder = tk.StringVar(value=[heading.text for heading in ET.parse(setting.get()).getroot().iter('project_name')][0])
project_name_Entry = tk.Entry(frame_train1,textvariable=project_name)

project_name_Entry.grid(row=0, column=1, padx=0,sticky='en')

ttk.Label(frame_train1, font=("Times New Roman",14),text="").grid(row=1, column=1, padx=0,sticky='en')
create_env_btn = tk.Button(frame_train1, text="Create Environment", command=make_env, font=("Times New Roman",14))
create_env_btn.grid(row=2, column=0, padx=10)
check_env_btn = tk.Button(frame_train1, text="Check Environment", command=check_env_lib, font=("Times New Roman",14))
check_env_btn.grid(row=2, column=1, padx=10)

#############################################
# 第1部分：Classification
#############################################

command = "conda info --envs"
process = subprocess.check_output(command, stdin=subprocess.PIPE, shell=True)
output =str(process)
output = output.split('*')[0]
output = output.split('\\n')[-1]
#print(output)




frame_train = ttk.LabelFrame(root,text="")
frame_train.grid(row=1, column=0,  padx=0,sticky='en')

Two_Image = tk.IntVar(value=0)
ttk.Label(frame_train, font=("Times New Roman",14),text="[ Classification ] 5-fold　　　　　　　").grid(row=0, column=0,columnspan=4,sticky='wn')
tk.Checkbutton(frame_train, text = "Two Image", variable = Two_Image,command=check_two_image).grid(row=0, column=3)

#--資料集路徑
ttk.Label(frame_train, font=("Times New Roman",14),text="Dataset：").grid(row=1, column=0,sticky='wn')
data_path = tk.StringVar(value= '')
train_dataset_input = tk.Button(frame_train,font=("Times New Roman",14),text="　　選取　　",command=select_folder).grid(row=1, column=1,sticky='wn') 

if data_path.get() == '':
    data_path_show =  tk.StringVar(value='')
else:
    data_path_show =  tk.StringVar(value='/' + data_path.get()+ '/')
tk.Label(frame_train, textvariable=data_path_show, font=("Times New Roman",14)).grid(row=1, column=2,sticky='wn')

data_path2 = tk.StringVar(value= '')
if data_path2.get() == '':
    data_path2_show =  tk.StringVar(value='')
else:
    data_path2_show =  tk.StringVar(value='/' + data_path.get()+ '/')
# data_path2_show =  tk.StringVar(value='/' + data_path2.get()+ '/')
train_dataset_input2 = tk.Button(frame_train,font=("Times New Roman",14),text="　　選取　　",command=select_folder2)
train_dataset_input2.grid(row=1, column=3,sticky='wn') 
tk.Label(frame_train, textvariable=data_path2_show, font=("Times New Roman",14)).grid(row=1, column=4,sticky='wn')



Attention = tk.BooleanVar(value= [heading.text for heading in ET.parse(setting.get()).getroot().iter('Attention')][0])
# ttk.Label(frame_train, font=("Times New Roman",14),text="\nAttention").grid(row=4, column=3,columnspan=4,sticky='wn')
Attention_btn = tk.Checkbutton(frame_train, text = "Attention", variable = Attention)
Attention_btn.grid(row=2, column=3)


#--選擇模型訓練 vit or densenet or swin#--選擇模型訓練 vit or densenet or swin
swinVar1 = tk.IntVar(value='swin' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('network')][0])
vitVar1 = tk.IntVar(value='vit' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('network')][0])
densenetVar1 = tk.IntVar(value='densenet' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('network')][0])

tk.Label(frame_train, font=("Times New Roman",14),text='Network :  \n(可複選)').grid(row=3, column=0,sticky='wn')
SwinViT_btn = tk.Checkbutton(frame_train, text = "SwinViT", variable = swinVar1)
SwinViT_btn.grid(row=3, column=3)
ViT_btn = tk.Checkbutton(frame_train, text = "ViT", variable = vitVar1)
ViT_btn.grid(row=3, column=2)
DenseNet_btn = tk.Checkbutton(frame_train, text = "DenseNet", variable = densenetVar1)
DenseNet_btn.grid(row=3, column=1)
check_two_image()
#--訓練方式
path = tk.StringVar(value='all')
scratch = tk.IntVar(value='trainfromscratch' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('train_type')][0])
Finetune = tk.IntVar(value='finetune' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('train_type')][0])
tk.Label(frame_train, font=("Times New Roman",14),text='Transfer Learning:　　').grid(row=5, column=0,sticky='w')
tk.Checkbutton(frame_train, text = "Fine-Tune", variable = Finetune).grid(row=5, column=1,sticky='w')
tk.Label(frame_train, font=("Times New Roman",14),text='(不勾選則為Train from scratch)').grid(row=5, column=2)

#--Parameters
tk.Label(frame_train, font=("Times New Roman",14),text='參數設定：　').grid(row=6, column=0,sticky='wn')
tk.Label(frame_train, font=("Times New Roman",14),text='Batch Size').grid(row=7, column=1,sticky='wn')
batch_size = tk.StringVar(value= [heading.text for heading in ET.parse(setting.get()).getroot().iter('batch_size')][0])
tk.Entry(frame_train,textvariable=batch_size).grid(row=7, column=2)

# tk.Checkbutton(frame_train, text = "Epoch", font=("Times New Roman",14) ,variable = '').grid(row=9, column=1)
# # tk.Label(frame_train, font=("Times New Roman",14),text='Epoch').grid(row=9, column=1)
# Epoch = tk.StringVar(value= [heading.text for heading in ET.parse(setting.get()).getroot().iter('Epoch')][0])
# tk.Entry(frame_train,textvariable=Epoch).grid(row=9, column=2)

# tk.Checkbutton(frame_train, text = "Earlystop:　Epoch", font=("Times New Roman",14),variable = '').grid(row=9, column=1)
Earlystop_Label = tk.Label(frame_train, font=("Times New Roman",14),text="Earlystop:　Epoch").grid(row=9, column=1,sticky='wn')

Earlystop_Epoch = tk.StringVar(value= [heading.text for heading in ET.parse(setting.get()).getroot().iter('Earlystop_Epoch')][0])


Earlystop_Epoch_Entry = tk.Entry(frame_train,textvariable=Earlystop_Epoch)
Earlystop_Epoch_Entry.grid(row=9, column=2)

Earlystop_validate = tk.StringVar(value= [heading.text for heading in ET.parse(setting.get()).getroot().iter('Earlystop_validate')][0])
Earlystop_validate_Label = tk.Label(frame_train, font=("Times New Roman",14),text="Validate accuracy ").grid(row=9, column=3,sticky='wn')
Earlystop_validate_Entry = tk.Entry(frame_train,textvariable=Earlystop_validate)
Earlystop_validate_Entry.grid(row=9, column=4)

tk.Label(frame_train, font=("Times New Roman",14),text="　　　　　　").grid(row=9, column=5,sticky='wn')
epoch_lock = tk.BooleanVar(value=False)
Epoch_btn = tk.Checkbutton(frame_train, text = "Epoch", fg='black', font=("Times New Roman",14) ,variable = epoch_lock,command=set_epoch)
Epoch_btn.grid(row=9, column=6)
Epoch = tk.StringVar(value='')
Epoch_Entry = tk.Entry(frame_train,textvariable=Epoch)
Epoch_Entry.grid(row=9, column=7)
set_epoch()


tk.Label(frame_train, font=("Times New Roman",14),text='Learning rate').grid(row=10, column=1)
Learning_rate = tk.StringVar(value= [heading.text for heading in ET.parse(setting.get()).getroot().iter('Learning_rate')][0])
tk.Entry(frame_train,textvariable=Learning_rate).grid(row=10, column=2)

tk.Label(frame_train, font=("Times New Roman",14),text='Optimizer').grid(row=11, column=1)
Optimizer = tk.StringVar(value= [heading.text for heading in ET.parse(setting.get()).getroot().iter('Optimizer')][0])
#tk.Entry(frame_train,textvariable=Optimizer).grid(row=11, column=2)

#combo_var = tk.StringVar()
combo = ttk.Combobox(frame_train, textvariable=Optimizer)
combo['values'] = ('SGD', 'Adam')
combo.grid(row=11, column=2, padx=10, pady=10)

# tk.Label(frame_train, font=("Times New Roman",14),text='執行：　　　　　　　　　　　　　　　　　　　　',font=("Times New Roman",14)).grid(row=11, column=2)
train_btn = tk.Button(frame_train,font=("Times New Roman",14),text=" Run ",command=train)
train_btn.grid(row=11, column=6)
tk.Label(frame_train, font=("Times New Roman",14),text='----------------------------------------------------------------------------------------------------------------------------------------------').grid(row=12, column=0,columnspan=9)


inference_path = tk.StringVar()
inference_model_path = tk.StringVar()

tk.Label(frame_train, font=("Times New Roman",14),text='[ Inference ] ').grid(row=13, column=0,sticky='wn')
ttk.Label(frame_train, font=("Times New Roman",14),text="Model :　　").grid(row=14, column=0)
retrival_btn = tk.Button(frame_train, font=("Times New Roman",14),text="　　選取　　", command=select_inference_model_path).grid(row=14, column=1)
tk.Label(frame_train, textvariable=inference_model_path, font=("Times New Roman",14)).grid(row=14, column=2,sticky='wn')



ttk.Label(frame_train, font=("Times New Roman",14),text="Test set :　　").grid(row=15, column=0)
retrival_btn = tk.Button(frame_train, font=("Times New Roman",14),text="　　選取　　", command=select_inference_folder).grid(row=15, column=1)
tk.Label(frame_train, textvariable=inference_path, font=("Times New Roman",14)).grid(row=15, column=2,sticky='wn')


inference_btn = tk.Button(frame_train,font=("Times New Roman",14),text=" Run ",command=inference)
inference_btn.grid(row=17, column=6)



#############################################
# # 第2部分：检索
#############################################
frame_search = ttk.LabelFrame(root,text="")
frame_search.grid(row=1, column=1, pady=0,sticky='n')
ttk.Label(frame_search, font=("Times New Roman",14),text="[ Retrieval] Top-10").grid(row=0, column=0,sticky='wn')

retrival_model_path = tk.StringVar()
query_path = tk.StringVar(value=[heading.text for heading in ET.parse(setting.get()).getroot().iter('query_dataset')][0])
target_path = tk.StringVar(value=[heading.text for heading in ET.parse(setting.get()).getroot().iter('target_dataset')][0])

query_path_show = tk.StringVar(value='/' + query_path.get()+ '/')
target_path_show =  tk.StringVar(value='/' + target_path.get()+ '/')

ttk.Label(frame_search, font=("Times New Roman",14),text="Model :").grid(row=1, column=0,sticky='wn')
retrival_btn = tk.Button(frame_search, font=("Times New Roman",14),text="　　選取　　", command=select_retrival_model_path).grid(row=1, column=1)
tk.Label(frame_search, textvariable=retrival_model_path, font=("Times New Roman",14)).grid(row=1, column=2,sticky='wn')

ttk.Label(frame_search, font=("Times New Roman",14),text="Target Database :　　").grid(row=3, column=0)
retrival_btn = tk.Button(frame_search, font=("Times New Roman",14),text="　　選取　　", command=select_target_folder).grid(row=3, column=1)

tk.Label(frame_search, font=("Times New Roman",14),textvariable=target_path).grid(row=3, column=2,sticky='wn')



ttk.Label(frame_search, font=("Times New Roman",14),text="Query Dataset :　　").grid(row=5, column=0)
query_btn = tk.Button(frame_search, font=("Times New Roman",14),text="　　選取　　", command=select_query_folder).grid(row=5, column=1)
tk.Label(frame_search, font=("Times New Roman",14),textvariable=query_path).grid(row=5, column=2,sticky='wn')



ttk.Label(frame_search, font=("Times New Roman",14),text="　　　　　　").grid(row=6, column=2)
retireve_btn = tk.Button(frame_search,font=("Times New Roman",14),text=" Run ",command=retireve)
retireve_btn.grid(row=6, column=3,sticky='en')
tk.Label(frame_search, font=("Times New Roman",14),text='----------------------------------------------------------------').grid(row=7, column=0,columnspan=4)

tk.Label(frame_search, font=("Times New Roman",14),text='[ Retrieval Result Features]').grid(row=9, column=0,sticky='wn',columnspan=4)
anwser  = tk.StringVar(value=  [heading.text for heading in ET.parse(setting.get()).getroot().iter('answer')][0])
anwser_show =  tk.StringVar(value='/'+anwser.get()+'/')

tk.Label(frame_search, font=("Times New Roman",14),text='True Labels:').grid(row=10, column=0,sticky='wn')
btn = tk.Button(frame_search, font=("Times New Roman",14),text="輸入答案", command=select_anwser_folder).grid(row=10, column=1,sticky='wn')
anwser_btn = tk.Button(frame_search,font=("Times New Roman",14),text=" Run ",command=Retrieval_Result_Features)
anwser_btn.grid(row=11, column=3,sticky='en')
tk.Label(frame_search, font=("Times New Roman",14),textvariable=anwser).grid(row=10, column=2,columnspan=4)

tk.Label(frame_search, font=("Times New Roman",14),text='----------------------------------------------------------------').grid(row=12, column=0,columnspan=4)

tk.Label(frame_search, font=("Times New Roman",14),text='[ Machine Learning] 10-fold').grid(row=13, column=0,sticky='wn',columnspan=4)
ML_csv  = tk.StringVar(value=  [heading.text for heading in ET.parse(setting.get()).getroot().iter('answer')][0])



SVM = tk.IntVar(value='SVM' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('model')][0])
Random_Forest = tk.IntVar(value='Random_Forest' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('model')][0])
Subspace_KNN = tk.IntVar(value='Subspace_KNN' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('model')][0])
ANN = tk.IntVar(value='ANN' in [heading.text for heading in ET.parse(setting.get()).getroot().iter('model')][0])

tk.Label(frame_search, font=("Times New Roman",14),text='Classifier :  \n(可複選)').grid(row=14, column=0,sticky='wn')
tk.Checkbutton(frame_search, text = "SVM", variable = SVM).grid(row=15, column=1,sticky='wn')
tk.Checkbutton(frame_search, text = "Random_Forest", variable = Random_Forest).grid(row=15, column=2,sticky='wn')
tk.Checkbutton(frame_search, text = "Subspace_KNN", variable = Subspace_KNN).grid(row=17, column=1,sticky='wn')
tk.Checkbutton(frame_search, text = "ANN", variable = ANN).grid(row=17, column=2,sticky='wn')
#tk.Label(frame_search, font=("Times New Roman",14),text='ML:').grid(row=18, column=0,sticky='wn')

ML_path = tk.StringVar()
# btn = tk.Button(frame_search, font=("Times New Roman",14),text="　　選取　　", command=select_ML_folder).grid(row=18, column=1,sticky='wn')
# tk.Label(frame_search, textvariable=ML_path, font=("Times New Roman",14)).grid(row=18, column=2,sticky='en')

ml_btn = tk.Button(frame_search,font=("Times New Roman",14),text=" Run ",command=ML)
ml_btn.grid(row=20, column=3,sticky='en')


primary_monitor = get_monitors()[0]
# 设置窗口大小
window_width = int(primary_monitor.width * 0.95)
window_height = int(primary_monitor.height * 0.8)
x = (primary_monitor.width - window_width) // 2
y = (primary_monitor.height - window_height) // 2
# 设置窗口大小和位置
root.geometry('{}x{}+{}+{}'.format(window_width, window_height, x, y))



check_two_image()
# check_env()
print("當前虛擬環境 :　",get_Current_virtual_environment())
root.mainloop()
