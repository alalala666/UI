#tensorboard --logdir=./run
#跑不出來:
#1.用絕對路徑
#2.換port 8088
#tensorboard --logdir=C:\Users\CNN\Downloads\ShauYuYan\S\runs_0706_rp --port 8088
#conda remove --name new --all 
#--------------------------------------------------------
#                       import
#--------------------------------------------------------
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np,openpyxl
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import csv
import csv
import numpy as np, pandas
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
import os  # 系統操作，例如路徑操作
import csv  # 用於 CSV 檔案操作
import time  # 用於時間相關功能
import h5py  # 用於 HDF5 檔案操作
import numpy as np  # NumPy 函式庫，用於數值計算
import matplotlib
import matplotlib.pyplot as plt  # 用於繪製圖表
from PIL import Image  # Python Imaging Library，用於圖像處理
from scipy.stats import chi2_contingency  # 用於卡方檢定
# import scipy.special._cdflib
from sklearn.metrics import confusion_matrix  # 用於混淆矩陣計算
import torch  # PyTorch 深度學習框架
import torchvision  # PyTorch 官方的電腦視覺函式庫
from torchvision import transforms  # 用於資料前處理和資料集
from torch.autograd import Variable  # 用於自動求導
import torch.nn as nn  # PyTorch 中的神經網路模組
from torchmetrics.functional.classification import multiclass_auroc  # 用於多類別 AUC 計算
from torch.utils.tensorboard import SummaryWriter  # 用於 TensorBoard 日誌
from torch.utils.tensorboard import SummaryWriter  # 用於 TensorBoard 日誌
from torchmetrics.functional.classification import multiclass_auroc  # 用於多類別 AUC 計算
from torch.autograd import Variable  # 用於自動求導
import torch.nn as nn  # PyTorch 中的神經網路模組
import torch  # PyTorch 深度學習框架
import warnings  # 用於警告管理
from utils.get_feature import swin_get_feature, vit_get_feature, densenet_get_feature  # 從自訂模組匯入特徵萃取函式
from utils.model_choose import model_choose  # 從自訂模組匯入模型選擇函式
from utils.vaild import evaluation_index  # 從自訂模組匯入評估指標計算函式
from utils.draw_roc import draw_roc,draw_Multiclass_ROC  # 從自訂模組匯入繪製 ROC 曲線函式
from utils.CustomDataset import customDataset,two_img_customDataset  # 從自訂模組匯入自訂資料集類別
from utils.cal_time import cal_time  # 從自訂模組匯入時間計算函式
from utils.make_input_csv import MAKE_CLASSIFICATION_DATASET,MAKE_INFERENCE_DATASET,MAKE_RETRIEVAL_DATASET  # 從自訂模組匯入時間計算函式
from torch.utils.data import DataLoader
import pandas as pd
import ast
from tqdm import tqdm
import shutil
torch.set_num_threads(1)  # 设置PyTorch线程数
warnings.filterwarnings('ignore')
np.seterr(divide='ignore',invalid='ignore')
matplotlib.use('Agg')
print('import OK!')

class EarlyStopping:
    '''
    收斂就停止訓練
    '''
    def __init__(self, patience=10, delta=1, verbose=False):
        '''
        定義類別 EarlyStopping，並在初始化函數中定義類別的屬性。

        patience：表示要等待多少個 epoch 才能判斷損失是否有下降。如果訓練 patience 個 epoch 後，驗證集的損失沒有下降，則停止訓練。

        delta：設定容忍的損失值變化，若驗證集的損失沒有下降超過 delta，就把這個 epoch 視為沒有進步。


        verbose：表示是否在屏幕上輸出調試訊息。

        counter：計算驗證集損失沒有下降的 epoch 數量。

        best_score：保存最好的驗證集損失。

        early_stop：表示是否停止訓練。

        val_loss_min：保存當前最小的驗證集損失。將其初始化為正無窮。
        '''
        self.patience = patience
        self.delta = delta
        
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss, model):
        '''
        在類別中實現了 __call__() 函數，可以讓該類別的對象像函數一樣被調用。
        當類別對象被調用時，這個函數會被執行。
        '''
        #將驗證集損失取負，因為我們希望監控的是損失的下降。
        score = val_loss
        '''
        如果 best_score 屬性是空的，表示這是第一次調用這個函數，
        所以將 score 賦值給 best_score，並保存模型權重。
        '''
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < float(self.best_score) + float(self.delta):
            self.counter += 1
            # if self.verbose:
            #     print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        '''
        保存模型權重的函數。如果 verbose 為 True，則在屏幕上輸出一條消息。
        然後使用 PyTorch 提供的 torch.save() 函數保存模型權重。
        最後，將當前的驗證集損失賦值給 val_loss_min。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.val_loss_min = val_loss

class CBMIR():
    def __init__(self):
        self.data_path =str
        self.data_path = str#" input_data\\" 
        self.data_path1 = str


        self.target_path = str
        self.query_path = str

        self.project_name  = str

        self.model_listt=['swin','vit','densenet']
        self.path_listt=['ALL']
        self.train_typee = ['finetune']

        self.num_epochs = 0
        self.max_epoch = 999
        self.earlystop_arg0 = 10
        self.earlystop_arg1 = 0.01

        self.batch_size = 16
        self.top_list = [10]
        self.K = 5
        self.repeat_times = 0
        self.split_82 = True 
        self.test_mode = False #只跑3 epoch ，用於檢查程式是否能順暢運行
        self.lr = 0.01
        self.optimizer = 'sgd'

        self.ML_answer = str
        self.retireve_fold = -1

        self.input_two_img = False
        self.fast_loader = False
    
    @staticmethod
    def move_sheet_to_first(excel_name, sheet_name):
            """
            將指定的工作表移動到第一個位置並保存 Excel 文件。

            :param excel_name: Excel 文件的路徑
            :param sheet_name: 要移動的工作表名稱
            """
            # 加載 Excel 文件
            workbook = openpyxl.load_workbook(excel_name)

            # 判斷工作表是否存在
            if sheet_name in workbook.sheetnames:
                # 移動工作表到第一位
                move_sheet = workbook[sheet_name]
                workbook._sheets.insert(0, workbook._sheets.pop(workbook._sheets.index(move_sheet)))

                # 保存 Excel 文件
                workbook.save(excel_name)
            else:
                pass
                # print(f"工作表 '{sheet_name}' 不存在於文件中.")
    
    def print_parameters(self):
        print("test_mode:", self.test_mode)
        # print("repeat_times:", self.repeat_times)

        print("input_two_img:", self.input_two_img)
        print("data_path:", self.data_path)
        print("data_path1:", self.data_path1)
        
        print("model_listt:", self.model_listt)
        print("path_listt:", self.path_listt)
        print("train_typee:", self.train_typee)
        # print("num_epochs:", self.num_epochs)
        print("max_epoch:", self.max_epoch)
        print("batch_size:", self.batch_size)
        print("lr:", self.lr)
        print("optimizer:", self.optimizer)

        print("save_path:", self.project_name)
        print("top_list:", self.top_list)
        print("K:", self.K)
        print("target_path:", self.target_path)
        print("query_path:", self.query_path)
        
        print("ML_answer:", self.ML_answer)
        # print("retireve_fold:", self.retireve_fold)
        
    def train_model(self,model_name = str,num_epochs = int,batch_size = int ,pretrain = bool ,train_path = str,test_path = str,save_path= str,fold= int,input_csv_path=str):
        def train(train_loader, model,optimizer, criterion, epoch, num_epochs, batch_size):
            
            total_train = 0
            correct_train = 0
            train_loss = 0
            print("train",model_name,"fold:",fold)
            mission = tqdm(total=len(train_loader.dataset))
            for batch_idx, (name,data, target) in enumerate(train_loader):
                for i in range(len(target)):
                    mission.update()
                model.train()

                if self.input_two_img:
                    target = torch.autograd.Variable(target).cuda()
                     # clear gradient
                    optimizer.zero_grad()
                    output = model(torch.autograd.Variable(data[0]).cuda(), 
                                   torch.autograd.Variable(data[1]).cuda())
                else:
                    data = Variable(data)
                    target = Variable(target)
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()

                    # clear gradient
                    optimizer.zero_grad()

                    # Forward propagation
                    output = model(data) 




                loss = criterion(output, target) 

                # Calculate gradients
                loss.backward()
                
                # Update parameters
                optimizer.step()

                predicted = torch.max(output.data, 1)[1]
                total_train += len(target)
                correct_train += sum((predicted == target).float())
                train_loss += loss.item()

                # if batch_idx % 1 == 0:
                #     print("Train Epoch: {} [iter： {}/{}], acc： {:.6f}, loss： {:.6f}".format(
                    # epoch, batch_idx+1, len(train_loader),
                    # correct_train / float((batch_idx + 1) * batch_size),
                    # train_loss / float((batch_idx + 1) * batch_size)))
                if  self.test_mode and  batch_idx  == 10 :  break
              #  break
            train_acc_ = 100 * (correct_train / float(total_train))
            train_loss_ = train_loss / total_train
                    
            return train_acc_.item(), train_loss_

        def validate(valid_loader, model, criterion, epoch, num_epochs, batch_size): 
            model.eval()
            total_valid = 0
            correct_valid = 0
            valid_loss = 0
            total_auc = 0
            total_specificity=0
            total_sensitivity=0
            total_ppv=0
            total_npv=0
            y_true = []
            y_scores = []
            y_prob = []
           # count = 0
            auc_preds = None
            auc_target = None

            print("validate",model_name,"fold:",fold)
            mission = tqdm(total=len(valid_loader.dataset))
            for batch_idx, (name,data, target) in enumerate(valid_loader):
                # if  self.test_mode and  batch_idx  == 10 :  break
                for i in range(len(target)):
                    mission.update()
                model.eval()
                if self.input_two_img:
                    target = torch.autograd.Variable(target).cuda()
                    output = model(torch.autograd.Variable(data[0]).cuda(), 
                                   torch.autograd.Variable(data[1]).cuda())
                else:
                    data = Variable(data)
                    target = Variable(target)
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()


                    # Forward propagation
                    output = model(data) 
                loss = criterion(output, target) 

                if auc_preds == None:
                    auc_preds = output
                    auc_target = target
                else:
                    auc_preds = torch.cat((auc_preds,output))
                    auc_target = torch.cat((auc_target,target))


                predict = torch.softmax(output, dim=1) 
                predict = torch.max(output.data, 1)[1]
                prob = torch.softmax(output, dim=1)
                prob = prob.tolist()
                total_valid += len(target)
                correct_valid += sum((predict == target).float())


                y_true.extend(target.tolist())
                y_scores.extend(predict.tolist())
                y_prob.extend(prob)

                valid_loss += loss.item()

                # if batch_idx % 1 == 0:
                #     count+=1
                    
            _,sensitivity,specificity,ppv,npv = evaluation_index(y_scores,y_true)
            total_auc = multiclass_auroc(auc_preds, auc_target, num_classes=num_classes, average="macro", thresholds=None)
            

            ###########################################
            #百分比轉換
            ###########################################
            valid_acc_ = 100 * (correct_valid / float(total_valid))
            valid_loss_ = valid_loss / total_valid
            total_auc = 100 *float("{:.2f}".format(total_auc))
            total_specificity =100 *float( "{:.2f}".format(specificity))
            total_sensitivity = 100 *float("{:.2f}".format(sensitivity))
            total_ppv = 100 *float("{:.2f}".format(ppv))
            total_npv =100 *float( "{:.2f}".format(npv))

            ###########################################
            #用於確定這fold否跑完
            ###########################################
            if not os.path.exists(save_path + '\\' + str(fold)):
                os.makedirs(save_path + '\\' + str(fold))

            # ###########################################
            # #這只能用於二分類
            # ###########################################
            # if 2 == int(next(csv.reader( open(input_csv_path.split('.csv')[0]+'_detail.csv', newline='')))[1]):
            #     draw_roc(y_true,y_prob,save_path+'\\'+ str(fold)+'\ROC.png')
            
            
            mapp = {}
            classes = []
            conut = 0
            for row_count, i in enumerate(open(input_csv_path.split('.csv')[0]+'_detail.csv', newline='')):
                if row_count < 2:continue
                classes.append(i.split(',')[0])
                mapp.update({i.split(',')[0]:conut})
                conut += 1
            test_labels = y_true
            pred_labels = y_scores
           
            max_str = ''
            for i in classes:
                 if len(i) > len(max_str):max_str = i

            if 0: #確定啥時需要畫圖
                cm = confusion_matrix(test_labels, pred_labels)
                # 繪製混淆矩陣
                #classes = int(next(csv.reader( open(input_csv_path.split('.csv')[0]+'_detail.csv', newline='')))[1])
                size = len(classes) * len(max_str)
                fig, ax = plt.subplots(figsize=(size,size))

                im = ax.imshow(cm, cmap=plt.cm.Blues)
                ax.set_xticks(np.arange(len(classes)))
                ax.set_yticks(np.arange(len(classes)))
                ax.set_xticklabels(classes, fontsize=40)
                ax.set_yticklabels(classes, fontsize=40)
                # 手动调整子图的边距
                fig.subplots_adjust(left=0.35, right=0.85, bottom=0.3, top=0.85)

                
                ax.set_xlabel('\nPredicted Class\n', fontsize=50)
                # 设置垂直标签
                ax.set_ylabel('\nTrue Class\n', fontsize=50)#true_class
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                        rotation_mode="anchor")
                for test_label in range(len(classes)):
                    for pred_label in range(len(classes)):
                    #  print(pred_label,test_label)
                    # print((len(classes)))
                        ax.text(pred_label, test_label, "{:.0f}".format(
                            cm[test_label, pred_label]),
                            ha="center",
                            va="top",
                            color="black",
                            fontdict={'fontname': 'Times New Roman', 'fontsize': 70})

                #ax.text(len(classes) - 0.5, -0.5, str(mapp), ha='right', va='center', fontsize=30, color='black')

                plt.title("Confusion matrix\n\n", fontsize=50)
                # plt.colorbar(im)
                cbar = plt.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=35)
                plt.savefig(save_path+'\\'+ str(fold)+'\Confusion matrix.png')
                plt.clf()
            
                plt.close(plt.figure())

                plt.close('all')


            return valid_acc_.item(), valid_loss_,total_auc,total_specificity,total_sensitivity,total_ppv,total_npv

        def training_loop(model, train_loader, valid_loader, seconds, csv_name, num_epochs, batch_size,fold,save_path):
            # if not os.path.exists(csv_name):    
            #     with open(csv_name, 'a+', newline='') as csvfile:
            #                 writer = csv.writer (csvfile)
            #                 writer.writerow(["fold", "num_epochs", 
            #                     (str("train_acc_")),
            #                     (str("train_loss_")),
            #                     (str("valid_acc_")),
            #                     (str("valid_loss_")),"specificity","sensitivity","ppv","npv","auc",
            #                     "cost_time"])
            total_train_loss = []
            total_valid_loss = []
            total_train_accuracy = []
            total_valid_accuracy = []
            training_accuracy = []
            valid_accuracy = []
            auc = []
            specificity = []
            sensitivity = []
            ppv = []
            npv = []
            max_acc = -1 # init 存下acc最高的model 
            
  
            early_stopping = EarlyStopping(patience=self.earlystop_arg0, delta=100 * self.earlystop_arg1)

            criterion = nn.CrossEntropyLoss()
            #optimizer = torch.optim.AdamW(model.parameters())
            #optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

            optimizer = torch.optim.SGD(model.parameters(),lr=self.lr)
            if self.optimizer == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(),lr=self.lr)
            elif self.optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(),lr=self.lr)


            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)


            # training
            epoch = 0
            best_model_path = ''
            while (early_stopping.early_stop == False) and (epoch<int(self.max_epoch)):

                epoch += 1
                self.num_epochs = epoch
                num_epochs = self.num_epochs
                train_acc_, train_loss_ = train(train_loader, model,optimizer, criterion, epoch, num_epochs, batch_size)
                total_train_loss.append(train_loss_)
                total_train_accuracy.append(train_acc_)
                formatted = "{:.2f}".format(train_acc_)
                training_accuracy.append(float(formatted))

                # validation
                with torch.no_grad():
                    valid_acc_, valid_loss_,total_auc,total_specificity,total_sensitivity,total_ppv,total_npv= validate(valid_loader , model, criterion, epoch,num_epochs, batch_size)

                    total_valid_loss.append(valid_loss_)
                    total_valid_accuracy.append(valid_acc_)
                    formatted = "{:.2f}".format(valid_acc_)
                    valid_accuracy.append(float(formatted))
                    auc.append(total_auc)
                    specificity.append(total_specificity)
                    sensitivity.append(total_sensitivity)
                    ppv.append(total_ppv)
                    npv.append(total_npv)

                    
                     # init 存下acc最高的model 
                    if valid_acc_ > max_acc:
                        max_acc = valid_acc_
                        best_model_path = save_path + '\\'+str(fold)+".pth"
                        torch.save(model,best_model_path)
                        
                    

                cost_time = cal_time(int(time.time()-seconds))

                print('================================================================================================================================')
                print("Epoch: {}， Train acc： {:.6f}， Train loss： {:.6f}， Valid acc： {:.6f}， Valid loss： {:.6f}， Time： {}".format(
                    epoch, 
                    train_acc_, train_loss_,
                    valid_acc_, valid_loss_,cost_time))
                print('================================================================================================================================')
                #寫入訓練資料

                early_stopping((valid_acc_), model)
                scheduler.step()
               # print("Early stopping",epoch,early_stopping.early_stop)
               
                
                # with open(csv_name, 'a+', newline='') as csvfile:
                #         writer = csv.writer (csvfile)
                #         writer.writerow([fold, num_epochs, 
                #             (str(train_acc_)),
                #             (str(train_loss_))[:7],
                #             (str(valid_acc_)),
                #             (str(valid_loss_))[:7],specificity[-1],sensitivity[-1],ppv[-1],npv[-1],auc[-1],
                #             cost_time])
                data = {
                    "fold": fold,
                    "num_epochs": num_epochs,
                    "train_acc_": str(train_acc_),
                    "train_loss_": str(train_loss_)[:7],
                    "valid_acc_": str(valid_acc_),
                    "valid_loss_": str(valid_loss_)[:7],
                    "specificity": specificity[-1],
                    "sensitivity": sensitivity[-1],
                    "ppv": ppv[-1],
                    "npv": npv[-1],
                    "auc": auc[-1],
                    "cost_time": cost_time
                }
                xlsx_path = self.project_name + '\Classification.xlsx'
                sheet_name = model_name

                if not os.path.exists(xlsx_path):openpyxl.Workbook().save(xlsx_path)
                
                workbook = openpyxl.load_workbook(xlsx_path, data_only=True)
                if sheet_name not in workbook.sheetnames:
                    workbook.create_sheet(sheet_name)
                    
                    workbook[model_name].append([str(i) for i in data])
                    workbook.save(xlsx_path)
                    
                workbook[model_name].append([str(data[i]) for i in data])
                workbook.save(xlsx_path)
            print("====== END ==========")




            self.draw_confuse_matirx_AND_roc_img(best_model_path,valid_loader,save_path,fold,input_csv_path)
            return total_train_loss, total_valid_loss, training_accuracy, valid_accuracy,specificity,sensitivity,ppv,npv,auc
        
        #get num class
        num_classes = int(next(csv.reader( open(input_csv_path.split('.csv')[0]+'_detail.csv', newline='')))[1])
        model =model_choose(model_name, num_classes, pretrain).cuda()

        df = pd.read_csv(input_csv_path,encoding='unicode_escape')
        
        if self.input_two_img:
            train_loader = DataLoader(two_img_customDataset(df[df['set'] != fold], shuffle = True), batch_size=batch_size)
            valid_loader = DataLoader(two_img_customDataset(df[df['set'] == fold]), batch_size=batch_size)
            
        else:
            train_loader = DataLoader(customDataset(df[df['set'] != fold], shuffle = True), batch_size=batch_size)
            valid_loader = DataLoader(customDataset(df[df['set'] == fold]), batch_size=batch_size)
        if self.fast_loader:
            train_loader = DataLoader(customDataset(df[df['set'] != fold], shuffle = True), batch_size=batch_size,pin_memory=True,num_workers=3)
        seconds = time.time()

        #詳細訓練資料路徑
        csv_name =save_path + '\\'+'output.csv'

        #assert False
        _, _, total_train_accuracy, total_valid_accuracy,specificity,sensitivity,ppv,npv,auc = training_loop(model, train_loader, valid_loader,seconds,csv_name,num_epochs,batch_size,fold,save_path)

        
        return total_train_accuracy,total_valid_accuracy,specificity,sensitivity,ppv,npv,auc

    def auto_train(self):
        '''
        開始訓練
        '''
        def two_img_make_input_csv(path0 = '',path1 = ''):
            save_path = self.project_name
            # path0 = path0 +'/' + os.listdir(path0)[0]
            # path1 = path1 +'/' + os.listdir(path1)[0]
            # if self.input_two_img:self.path_listt=[path0.split('/')[-1] +'_'+ path1.split('/')[-1]]
            dataset = path0.split('/')[-1] +'_'+ path1.split('/')[-1]
            csv_path = save_path +'/' +dataset + '_dataset.csv'
            detail_csv_path = save_path +'/' +dataset + '_dataset_detail.csv'
            if os.path.exists(save_path +'/' + dataset + '_dataset_detail.csv'):
                return 'data exist'
            csv.writer(open(csv_path, 'w', newline='')).writerow(["category",'set', "img_path0", "img_path1"])  
            
           
            #detail_csv_path = save_path +'/' + dataset + '_dataset_detail.csv'
            count = 0
            label_map = {}
            
            for category_id,category in enumerate(os.listdir(path0)):
                label_map.setdefault(category,category_id)
                detail_csv_path = save_path +'/' + dataset + '_dataset_detail.csv'
                csv.writer(open(detail_csv_path, 'w', newline='')).writerow(['class_num:',category_id+1,'dataset_path:',path0,path1])
                for id, img in enumerate(os.listdir(path0 + '/' + category)):#Using enumerate  to easily divide the set
                    count += 1
            csv.writer(open(detail_csv_path, 'a+', newline='')).writerow(['5-fold : ']) # make title and init csv
            
            listA = [[],[]]
            
            for category_id,category in enumerate(os.listdir(path0)):
                for id, img in enumerate(os.listdir(path0 + '/' + category)):#Using enumerate  to easily divide the set
                    img_path = path0 +'/'+ category + '/' + img
                    listA[0].append(img_path)
            for category_id,category in enumerate(os.listdir(path1)):
                check_calculation_map = {}
                for id, img in enumerate(os.listdir(path1 + '/' + category)):#Using enumerate  to easily divide the set
                    img_path = path1 +'/'+ category + '/' + img
                    listA[1].append(img_path)
                    check_calculation_map.setdefault(id % 5, 0)
                    check_calculation_map[id % 5] += 1
                csv.writer(open(detail_csv_path, 'a+', newline='')).writerow([category, check_calculation_map])
        
            # print(listA[0][1])
            # print(listA[1][1])
            for id , _ in enumerate(listA[1]):
                    #print(listA[0][id],listA[1][id])
                    name = str(listA[0][id]).split('/')[-1]
                    label = str(listA[0][id]).split('/')[-2]
                    csv.writer(open(csv_path, 'a+', newline='')).writerow([label_map[label], id % 5,listA[0][id],listA[1][id]])
                    check_calculation_map.setdefault(id % 5, 0)
                    check_calculation_map[id % 5] += 1

            return "end two_img_make_input_csv"
         
        

        #----------------------------------------
        #----訓練前準備
        # 建立資料夾並列印檢索結果標頭
        #----------------------------------------

        if self.test_mode == True:self.max_epoch = 2
        if self.model_listt == ['all']:self.model_listt =['swin','vit','densenet']
        if self.train_typee == ['all']:self.train_typee =['finetune','trainfromscratch']
        # if self.input_two_img:
        #     self.data_path = self.data_path +'/' + os.listdir(self.data_path)[0]
        #     self.data_path1 = self.data_path1 +'/' + os.listdir(self.data_path1)[0]
        if self.input_two_img:
            self.path_listt=[self.data_path.split('/')[-1] +'_'+ self.data_path1.split('/')[-1]]
        else:
            self.path_listt=[self.data_path.split('/')[-1]]
        for train_type in self.train_typee:
            for model_list in self.model_listt:
                for path_list in self.path_listt:
                    for k in range(self.K):
                        save_pathh = os.path.join(self.project_name, train_type+'_classification', path_list, model_list)
                        os.makedirs(save_pathh, exist_ok=True)

        
        print('-----------------------make_input_csv-----------------------')  
        if self.input_two_img:
            two_img_make_input_csv(self.data_path,self.data_path1)
        else:
            datamaker = MAKE_CLASSIFICATION_DATASET(self.data_path,self.project_name)
            datamaker.make_input_excel()
            datamaker.save_sheet_as_csv('classification_dataset', self.project_name+'/temp/classification_dataset.csv')
            datamaker.save_sheet_as_csv('classification_dataset_Summary', 'CAT_DOG/temp/classification_dataset_detail.csv')
        print('make_input_csv OK !')     
        print('------------------------------------------------------------') 
        #----------------------------------------
        #-----開始訓練
        # 訓練並印出訓練細節
        #----------------------------------------

        # 這段程式碼初始化了三個迴圈，
        # 分別迭代不同的超參數值，例如訓練類型、模型清單和路徑清單。
        # train_typee、model_listt和path_listt是包含這些超參數不同值的清單。
        for train_type in range(len(self.train_typee)):
            for model_list in range(len(self.model_listt)):
                for path_list in range(len(self.path_listt)):
                    # 這裡的程式碼初始化了五個變數，
                    # 用於存儲平均訓練準確度、平均測試準確度、平均特異度、平均靈敏度和平均曲線下面積（AUC）。
                    # 這些變數將在訓練模型後用於計算這些指標的平均值。
                    avg_train_acc = 0
                    avg_test_acc = 0
                    avg_specificity = 0
                    avg_sensitivity = 0
                    avg_ppv = 0
                    avg_npv = 0
                    avg_auc = 0 

                    sd_train_acc = []
                    sd_test_acc = []
                    sd_specificity = []
                    sd_sensitivity = []
                    sd_ppv = []
                    sd_npv = []
                    sd_auc = [] 
                    #計時開始
                    seconds = time.time()
                    # 這段程式碼對於迭代的每一組超參數，使用五個指標來計算模型的平均表現。
                    # 在訓練和測試期間，先將 pretrain 設置為 True 或 False，
                    # 具體取決於 train_typee 變數是否等於 'finetune'。
                    # 此外，還需設置其他變數，如訓練和測試數據集的路徑，以及保存模型的路徑。
                    # 然後，該函數 train_model() 會使用這些變數來訓練模型，並返回五個指標的值。
                    for k in range(self.K):
                        input_csv_path = self.project_name +'/temp/classification_dataset.csv'
                        pretrain = True if self.train_typee[train_type] =='finetune' else False
        
                        save_pathh = self.project_name + '\\'+ self.train_typee[train_type]+'_classification\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list]
                        train_path = 'input_data\\' +self.path_listt[path_list] +'\\fold' + str(k) + '\\train'
                        test_path = 'input_data\\' +self.path_listt[path_list] +'\\fold' + str(k) + '\\test' 
                        
                        model_path = save_pathh+"\\"+str(k+1)+".pth"

                        #用於檢查是否訓練過
                        # 0704 先拿掉
                        #if os.path.exists(model_path):continue
                            
                        total_train_accuracy, total_valid_accuracy,specificity,sensitivity,ppv,npv,auc = self.train_model(self.model_listt[model_list],self.num_epochs,self.batch_size,pretrain,train_path,test_path,save_pathh,k,input_csv_path)
                        avg_train_acc += float(total_train_accuracy[-1])/(self.K)
                        sd_train_acc.append(float(total_train_accuracy[-1]))

                        avg_test_acc += float(total_valid_accuracy[-1])/(self.K)
                        sd_test_acc.append(float(total_valid_accuracy[-1]))

                        avg_specificity += float(specificity[-1])/(self.K)
                        sd_specificity.append(float(specificity[-1]))

                        avg_sensitivity += float(sensitivity[-1])/(self.K)
                        sd_sensitivity.append(float(sensitivity[-1]))

                        avg_ppv += float(ppv[-1])/(self.K)
                        sd_ppv.append(float(ppv[-1]))

                        avg_npv += float(npv[-1])/(self.K)
                        sd_npv.append(float(npv[-1]))

                        avg_auc+= float(auc[-1])/(self.K)
                        sd_auc.append(float(auc[-1]))

                    cost_time = cal_time(int(time.time()-seconds))
                    #資料可視化
                    writer = SummaryWriter(self.project_name + '/runs/'+str(self.train_typee[train_type])
                                                  +'_'+str(self.model_listt[model_list])
                                                  +'_'+str(self.path_listt[path_list])
                                                  +'_'+str(self.K)
                                                  +'_'+str(self.repeat_times)
                                                )
                        
                    for n_iter in range(self.num_epochs):
                            writer.add_scalar('auc/auc_fold'+str(k),float(auc[n_iter]), n_iter)
                            writer.add_scalar('npv/npv_fold'+str(k),float(npv[n_iter]), n_iter)
                            writer.add_scalar('ppv/ppv_fold'+str(k),float(ppv[n_iter]), n_iter)
                            writer.add_scalar('sensitivity/sensitivity_fold'+str(k),float(sensitivity[n_iter]), n_iter)
                            writer.add_scalar('specificity/specificity_fold'+str(k), float(specificity[n_iter]), n_iter)
                            writer.add_scalar('train_Accuracy/train_fold'+str(k), float(total_train_accuracy[n_iter]), n_iter)
                            writer.add_scalar('test_Accuracy/test_fold'+str(k),float(total_valid_accuracy[n_iter]), n_iter)
                    writer.close()

#################################################### 0704 ################################

                # 創建數據框架
                mean_data = {
                    "Train Type": [self.train_typee[train_type]],
                    #"Path List": [path_listt],
                    "Model List": [self.model_listt[model_list]],
                    "Avg Train Acc": [round(avg_train_acc, 2)],
                    "Avg Test Acc": [round(avg_test_acc, 2)],
                    "Avg Specificity": [round(avg_specificity, 2)],
                    "Avg Sensitivity": [round(avg_sensitivity, 2)],
                    "Avg PPV": [round(avg_ppv, 2)],
                    "Avg NPV": [round(avg_npv, 2)],
                    "Avg AUC": [round(avg_auc, 2)],
                    "Cost Time": [cost_time]
                }

                std_data = {
                    "Train Type": [self.train_typee[train_type]],
                    #"Path List": [path_listt],
                    "Model List": [self.model_listt[model_list]],
                    "Std Train Acc": [np.std(sd_train_acc)],
                    "Std Test Acc": [np.std(sd_test_acc)],
                    "Std Specificity": [np.std(sd_specificity)],
                    "Std Sensitivity": [np.std(sd_sensitivity)],
                    "Std PPV": [np.std(sd_ppv)],
                    "Std NPV": [np.std(sd_npv)],
                    "Std AUC": [np.std(sd_auc)]
                }


                xlsx_path = self.project_name + '\Classification.xlsx'
                wb = openpyxl.load_workbook(xlsx_path, data_only=True)
                # 判斷檔案是否存在
                if not os.path.exists(xlsx_path):
                    openpyxl.Workbook().save(xlsx_path)
                if 'Mean' not in wb.sheetnames:
                    openpyxl.Workbook().save(xlsx_path)
                    
                    wb.create_sheet('Mean')
                    wb.create_sheet('Std')     # 新增工作表 3
                    wb['Mean'].append([i for i in mean_data])
                    wb['Std'].append([i for i in std_data])
                    
                    wb.remove(wb['Sheet'])
                    wb.save(xlsx_path)

                wb = openpyxl.load_workbook(xlsx_path, data_only=True)
                if self.model_listt[model_list] not in [cell.value for cell in wb['Mean']['C']]:
                    wb['Mean'].append([str(mean_data[i][0]) for i in mean_data])
                    wb['Std'].append([str(std_data[i][0]) for i in std_data])
                wb.save(xlsx_path)

        self.move_sheet_to_first(xlsx_path, 'Mean')
        self.move_sheet_to_first(xlsx_path, 'Std')


        shutil.rmtree(self.project_name+'/temp')
#################################################### 0704 ################################

    def retireve(self):
        print("-------------------------------------------------------")
        print("開始檢索")
        print("-------------------------------------------------------")
              
        print("make_target_input_csv")
       
        datamaker = MAKE_RETRIEVAL_DATASET(self.query_path,self.project_name,'query')
        datamaker.make_input_excel()
        datamaker.save_sheet_as_csv('query_dataset', self.project_name + '/temp/query_dataset.csv')

        datamaker = MAKE_RETRIEVAL_DATASET(self.target_path,self.project_name,'target')
        datamaker.make_input_excel()
        datamaker.save_sheet_as_csv('target_dataset', self.project_name + '/temp/target_dataset.csv')
        print("start:")
        
        if self.model_listt == ['all']:
            self.model_listt =['swin','vit','densenet']
        if self.train_typee == ['all']:
            self.train_typee =['finetune','trainfromscratch']
       # assert os.path.exists('save/finetune') ,'需要先訓練模型，請勾選train'
        def ap_compute(query_path = str,target_path = str,save_path = str,top = int):
    
            def mAP_output(path = str):
                num_classes = {}
                classes = {}
                mAP = 0
                with open(path, newline='') as csvfile:
                    next(csv.reader(csvfile))
                    for row in csv.reader(csvfile):
                        try:
                            roww = str(row[1])
                        except:
                            continue
                        if str(row[1]) not in classes:
                            classes.update({str(row[1]):float(row[0])})
                            num_classes.update({str(row[1]):1})
                        else:
                            classes.update({str(row[1]):(classes[str(row[1])]*num_classes[str(row[1])]+float(row[0]))/(1+num_classes[str(row[1])])})
                            num_classes.update({str(row[1]):num_classes[str(row[1])]+1})
                    for i in classes:
                        mAP += classes[i] / len(classes)
                    return(mAP,classes)
            
            #----------------------------------------------------------
            #              input feature and path
            #----------------------------------------------------------
            output_excel_path = self.project_name + '/Retrival.xlsx'
            sheetname = save_path#.replace('\\','_')
            # 將字串分割為列表
            parts = sheetname.split('\\')

            # 選取需要的部分 (第 2、5、6 個元素)
            selected_parts = [parts[1].split('_')[0], parts[3], parts[4]]

            # 將選取的部分重新組合為字串
            sheetname = '_'.join(selected_parts)
            # 創建一個新的工作簿和工作表
            if not os.path.exists(output_excel_path):
                openpyxl.Workbook().save(output_excel_path)
                # wb = openpyxl.load_workbook(output_excel_path, data_only=True)
                # wb.create_sheet('Summary')
                # wb.save(output_excel_path)
            wb = openpyxl.load_workbook(output_excel_path, data_only=True)
            if 'Summary' not in wb.sheetnames:wb.create_sheet('Summary')
            wb.create_sheet(sheetname)
            

            
            
            if 'Sheet'  in wb.sheetnames:wb.remove(wb['Sheet'])   



            ws = wb[sheetname]
            # 添加表頭
            ws.append(['precision','category','path'])
            
            feature_list = []
            feature_path = []
            query_list = []
            query_path_list = []
            with h5py.File(target_path, "r") as k:
                for i in range(len(k.get('feature'))):
                    feature_list.append(k.get('feature')[i])
                    feature_path.append(k.get('path')[i])
            with h5py.File(query_path, "r") as k:
                for i in range(len(k.get('feature'))):
                    query_list.append(k.get('feature')[i])
                    query_path_list.append(k.get('path')[i])
            #----------------------------------------------------------
            #               mAP compute
            #----------------------------------------------------------
            #將結果寫入csv
            #先宣告要用的欄位
            # ap , 分類 , 路徑
            with open(save_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['ap','category','path'])
           # print(len(feature_list))
            #mAP init
            mAP = 0
            #開始跑每張圖
            mission  = tqdm(total=len(query_list))
            for j in range(0,len(query_list)):
                #continue
                
                #print(j)
            
                #查詢影像
                query = query_list[j]
                query_path = query_path_list[j]
                #用來存放每張圖的cosin similarity
                score_map = {}
                #比對資料庫中的每一筆DATA
                for i in range(len(feature_list)):
                    #計算cosin similarity
                    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                    #需要轉tensor
                    cosine_similarity = cos(torch.from_numpy(query),torch.from_numpy(feature_list[i]))
                    #寫入 dic
                    score_map.update({cosine_similarity:feature_path[i]})

                    #寫入 dic
                    if (len(score_map) < top + 1):
                        score_map.update({cosine_similarity:feature_path[i]})
                        continue
                    if (cosine_similarity<(min(score_map))):
                        continue
                    score_map.update({cosine_similarity:feature_path[i]})
                    del score_map[min(score_map)]
                #將前 n 相似的輸出
                top_list = {}
                for i in range(top):
                    #每次都挑最大的放入dic 概念類似選擇排序
                    top_list.update({max(score_map):score_map[max(score_map)]})
                    #將最大的刪除
                    del score_map[max(score_map)]
                
                topk = []
                topk_path = []
                relevant = 0 
                #印出top-n串列
                for i in top_list:
                    if (str(query_path).split('\\\\')[-2]) == (str(top_list[i]).split('\\\\')[-2]):
                        relevant = relevant + 1
                    topk.append(str(top_list[i]).split('\\\\')[-2])
                    topk_path.append(str(top_list[i]))
                    #print(top_list[i])
                    
                #print(save_pathh,relevant,top)
                ap = relevant/top
                mAP += ap/len(query_list)
                #write in csv 
                with open(save_path, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([ap,(str(query_path).split('\\\\')[-2]),str(query_path),topk,topk_path])
                    # 添加數據行
                    ws.append([ap,(str(query_path).split('\\\\')[-2]),str(query_path)])
                    csvfile.close()
                mission.update()
            
            mAP,classes = mAP_output(save_path)
            
            # with open(save_path, 'a+', newline='') as csvfile:
            #         writer = csv.writer(csvfile)
            #         for row in classes:
            #             writer.writerow([row,classes[row]])
            #         writer.writerow(['mAP',mAP])
            wb['Summary'].append( [parts[1].split('_')[0], parts[3], parts[4],mAP])
            self.move_sheet_to_first(output_excel_path, 'Summary')
            
            #print('mAP : ',mAP)
            
            
           
            
            wb.save(output_excel_path)
            



            return mAP

        #開始檢索
        self.top_list.sort()
        for train_type in range(len(self.train_typee)):
            for model_list in range(len(self.model_listt)):
                for path_list in range(len(self.path_listt)):  
                    for top in range(len(self.top_list)):
                        avg_mAP = 0.0
                                               
                        seconds = time.time()
                        timee = ''
                        for k in range(self.K):
                            # print(self.retireve_fold)
                            # print(self.retireve_fold == -1)
                            if self.retireve_fold == -1:
                                print('all')
                            elif self.retireve_fold != k:
                                continue
                            

                            if self.train_typee[train_type] == "trainfromscratch":
                                save_pathh = self.project_name + '\\trainfromscratch_retrieve'+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\fold' + str(k) 
                                #continue
                            elif self.train_typee[train_type] == "finetune":
                                save_pathh = self.project_name + '\\finetune_retrieve'+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\fold' + str(k)
                            if not os.path.exists(save_pathh):
                                os.makedirs(save_pathh)
                            model_path = self.project_name + '\\'+ self.train_typee[train_type]+'_classification\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\'+str(k)+'.pth'
                            
                            # for dataset in os.listdir(self.query_path):
                            for i in range(1):
                                
                                target_dataset = self.project_name +'/temp/target_dataset.csv'
                                break
                            target_path =self.project_name +'/' +self.path_listt[path_list] + '*' +  target_dataset

                            # for dataset in os.listdir(self.query_path):
                            for i in range(1):
                                query_dataset = self.project_name +'/temp/query_dataset.csv'
                                break
                            query_path = self.project_name +'/' +self.path_listt[path_list] + '*' + query_dataset
                            
                            seconds = time.time() 
                            if not os.path.exists(save_pathh):
                                os.makedirs(save_pathh)
                            mAP = 0.0
                            if self.model_listt[model_list] == 'vit':
                                    if not os.path.exists(save_pathh +'\\target_data.h5'): 
                                        print('target_get_feature vit')
                                        vit_get_feature(target_path,save_pathh,model_path,k)
                                        print('query_get_feature vit')
                                        vit_get_feature(query_path,save_pathh,model_path,k)
                            if self.model_listt[model_list] == 'swin':
                                    if not os.path.exists(save_pathh +'\\target_data.h5'): 
                                        print('target_get_feature swin')
                                        swin_get_feature(query_path,save_pathh,model_path,k)
                                        print('query_get_feature swin')
                                        swin_get_feature(target_path,save_pathh,model_path,k)
                            if self.model_listt[model_list] == 'densenet':
                                    if not os.path.exists(save_pathh +'\\target_data.h5'): 
                                        print('target_get_feature densenet')
                                        densenet_get_feature(target_path,save_pathh,model_path,k)
                                        print('query_get_feature densenet')
                                        densenet_get_feature(query_path,save_pathh,model_path,k)
                            #continue
                            
                            ap_path = save_pathh + '\\top_' + str(self.top_list[top])
                            if not os.path.exists(ap_path):
                                os.makedirs(ap_path)
                            #continue
                            print('compute mAP')
                            mAP = ap_compute(save_pathh +'\\query_data.h5',save_pathh +'\\target_data.h5',ap_path+'\output.csv',(int(self.top_list[top])))
                            avg_mAP += mAP / int(self.K)
                        
                            cost_time = cal_time(int(time.time()-seconds))
                            with open(save_pathh+'\output.csv', 'a+', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([cost_time])
                            timee = cost_time
                        
        
        shutil.rmtree(self.project_name+'/temp')
    def make_query_img(self):
        print("-------------------------------------------------------")
        print("開始印檢索圖")
        print("-------------------------------------------------------")

        #----------------------------------------------------------
        def query_show(model_name = 'vit',model_path = str,img_path = str,feature_path_h5 = str,retrieve_save_path = str):
                
            def densenet_feature(img_path):
                        transform = transforms.Compose([
                                transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
                        ]) 

                        img = Image.open(img_path)
                        img = transform(img)
                        img = torch.unsqueeze(img, dim=0)
                        CUDA = torch.cuda.is_available()
                        device = torch.device("cuda" if CUDA else "cpu")
                        img = img.to(device)

                        # create model
                        
                    
                        #a = model.features.children
                        
                        model=torch.load(model_path)
                        with torch.no_grad():
                            input = model.features(img)
                            avgPooll = nn.AdaptiveAvgPool2d(1)

                            output = avgPooll(input)
                            output = torch.transpose(output, 1, 3)#把通道维放到最后
                            featuree = output.view(1920).cpu().numpy()
                            print(featuree.shape)
                            return featuree


            def swin_feature(img_path):


                            transform = transforms.Compose([
                                    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
                            ]) 

                            img = Image.open(img_path)
                            img = transform(img)
                            img = torch.unsqueeze(img, dim=0)
                            CUDA = torch.cuda.is_available()
                            device = torch.device("cuda" if CUDA else "cpu")
                            img = img.to(device)

                            # create model
                            #model = torch.load('swin_b-68c6b09e.pth')
                            model = torchvision.models.swin_b(weights = None)
                            model=torch.load(model_path)
                            #model = torchvision.models.swin_b(weights = 1)
                            #model.cuda()
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


            def vit_feature(img_path):
                        '''
                        extract feature from an image
                        '''
                        # image ->(3,224,224)
                        transform = transforms.Compose([
                                transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
                        ])

                        # use path open image
                    
                        img = Image.open(img_path)
                       # print(img_path, img_path[:-3])
                        img = transform(img)

                        #通道多一條
                        img = torch.unsqueeze(img, dim=0)
                        
                        CUDA = torch.cuda.is_available()
                        device = torch.device("cuda" if CUDA else "cpu")
                        img = img.to(device)
                        #print(img.shape)

                        # create model
                        #model = torch.load(model_path)
                        #model = torchvision.models.vit_b_16(weights = None) 
                        # model = torch.load('save_fin/finetune/MSI/vit/2.pth')
                        #model.cuda()
                        model.eval()
                        
                        
                        with torch.no_grad():
                            x = model._process_input(img)
                            n = x.shape[0]
                            batch_class_token = model.class_token.expand(n, -1, -1)
                            x = torch.cat([batch_class_token, x], dim=1)
                            x = model.encoder(x)
                            x = x[:, 0]
                            featuree = x.view(768).cpu().numpy()
                            #print(featuree.shape)
                            return featuree


            ap=0
            feature_list = []
            feature_path = []
            count = 0
            feature_list = []
            feature_path = []
            with h5py.File(feature_path_h5, "r") as k:
                for i in range(len(k.get('feature'))):
                    feature_list.append(k.get('feature')[i])
                    feature_path.append(k.get('path')[i])
            
            model =  torch.load(model_path)
            #----------------------------------------------------------
            #               mAP compute
            #----------------------------------------------------------

            #將結果寫入csv
            #先宣告要用的欄位
            # ap , 分類 , 路徑
        

            #查詢影像
            if model_name == 'vit':
                query  = (vit_feature(img_path))
            elif model_name == 'densenet':
                query  = (densenet_feature(img_path))
            elif model_name == 'swin':
                query  = (swin_feature(img_path))


            query_path = img_path
            path = img_path

            #用來存放每張圖的cosin similarity
            score_map = {}

            #比對資料庫中的每一筆DATA
            for i in range(len(feature_list)):

                #計算cosin similarity
                cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
                #需要轉tensor
                cosine_similarity = cos(torch.from_numpy(query),torch.from_numpy(feature_list[i]))
                #寫入 dic
                score_map.update({cosine_similarity:feature_path[i]})
            #將前 n 相似的輸出
            top_list = {}
            image = []
            image.append(img_path)
            #print(image)
            #top_10
            top = 10
            for i in range(top):
                #每次都挑最大的放入dic 概念類似選擇排序
                top_list.update({max(score_map):score_map[max(score_map)]})
                #將最大的刪除
                del score_map[max(score_map)]
            #print(top_list)
            relevant = 0 
            #印出top-n串列
            # for i in top_list:
            #     if (str(query_path).split('/')[-2]) == (str(top_list[i]).split('/')[-2]):
            #         relevant = relevant + 1
            

            #     #print(top_list[i])
            #     image.append(str((top_list[i]))[3:-2]+'*'+str(float(i))[:4])
            for i in top_list:
                    if (str(query_path).split('/')[-2]) == (str(top_list[i][0]).split('\\\\')[-2]):
                        relevant = relevant + 1
                    image.append(str(top_list[i][0])[2:-1].replace('\\\\', '\\')+'*'+str(float(i))[:4])
            ap = relevant/top

            #print(len(feature_list))
            #path = path.split('/')[1]+path.split('/')[2]+'_'+path.split('/')[3]
            fig, axs = plt.subplots(3, 4, figsize=(15, 10))
            fig.subplots_adjust(hspace=0.5, wspace=0.3)
            axs = axs.flatten()  # 将子图数组展开为一维数组
            for i, image_path in enumerate(image):
                # 读取图像
                with open(image_path.split('*')[0], 'rb') as f:
                    img = Image.open(f)
                    # 显示图像和标题
                    axs[i].imshow(img)
                    axs[i].get_xaxis().set_visible(False)  
                    axs[i].get_yaxis().set_visible(False)
                    path=path
                    if i == 0:
                        axs[i].set_title(f"{'query'} ")
                    else:
                        a = (image_path.split('*')[0]).split('\\')[-2]
                        axs[i].set_title(f"{i}\n {os.path.basename(image_path.split('*')[1])}\n{a}")
                    f.close()

            # 移除多余的子图
            for i in range(len(image), len(axs)):
                axs[i].remove()
            
            plt.savefig(retrieve_save_path, format='png', dpi=300)
            #plt.show()
            plt.clf()
            plt.close('all')

        count = 0
        img_num = 0
        save_path = self.project_name
        for train_type in range(len(self.train_typee)):
            for model_list in range(len(self.model_listt)):
                for path_list in range(len(self.path_listt)):
                    for k in range(self.K):
                        if self.retireve_fold == -1:
                                print('all')
                        elif self.retireve_fold != k:
                                continue
                        if self.train_typee[train_type] == "trainfromscratch":
                            feature_path = self.project_name + '\\trainfromscratch_retrival'+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\fold' + str(k) 
                        else:
                            feature_path = self.project_name + '\\finetune_retrieve'+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\fold' + str(k)
                        img_save_path = feature_path + '/retrieval_result_img'
                        if not os.path.exists(img_save_path):
                            os.makedirs(img_save_path)
                        feature_path += '\\target_data.h5'
                        model_path = self.project_name + '\\'+ self.train_typee[train_type]+'_classification\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\'+str(k)+'.pth'
                        model_name = str(self.model_listt[model_list])
                        
                        #print(feature_path,model_path,model_name,img_save_path)
                        query_show_img_count = 0 
                        for source in os.listdir(self.query_path):
                            for img in os.listdir(self.query_path + '/' + source):
                                query_show_img_count+=1
                        mission = tqdm(total=(query_show_img_count))  
                        for source in os.listdir(self.query_path):
                            for img in os.listdir(self.query_path + '/' + source):
                                #for img in os.listdir(self.query_path + '/' + source+ '/' + folder):
                                    img_path = self.query_path + '/' + source +'/'+img
                                   # print(img_path)
                                    img_save_path1 =img_save_path +  '/'
                                    img_save_path1 += str(img)
                                    query_show(model_name=model_name,
                                            model_path=model_path,
                                            img_path=  img_path,
                                            feature_path_h5=feature_path,
                                            retrieve_save_path=img_save_path1)
                                    mission.update()
                                    

                        # #print(test_path)
                        # for folder in os.listdir(test_path):
                        #     for img in os.listdir(test_path + '/' + folder):
                        #         img_num += 1

    def make_ML_feature(self):
        '''
        製作ML所需的特徵
        '''
        
        def make_Probability_feature(retrieval_detail_result=str,model_path=str,ML_feature_save_path=str,query_dataset_csv_path = str,answer_list = dict):
            model = torch.load(model_path).cuda()
            csv_path = query_dataset_csv_path

            count = 0 
            porb_feature_list = []
            for id,row in enumerate(csv.reader(open(query_dataset_csv_path, newline=''))):
                count += 1
            for id,row in enumerate(csv.reader(open(query_dataset_csv_path, newline=''))):
                try:
                    porb_list = []
                    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.5]*3, [0.5]*3)])
                    img_name = str(row[2])
                    img = torch.unsqueeze(transform(Image.open(img_name)), dim=0).cuda()
                    output = torch.softmax(model(img),dim=1).flatten().tolist()
                    porb_list.append(img_name)
                    porb_list.extend(output)
                    print('make_Probability_feature --> compute Probability : ',id)
                    porb_feature_list.append(porb_list)
                    #if self.test_mode:break
                except:
                    continue
           
            #使用檢索細節表
            csv_path = retrieval_detail_result

            #出現最多的類別(儲存位置)
            Retrieval_feature_path = ML_feature_save_path +  '/Probability.csv'

            ML_list = []
            sets =[]
           # csv.writer(open(the_Most_Similar_Category, 'w', newline='',encoding='utf-8')).writerow(['img_path','最像的類別'])
      

            for count ,row in enumerate(csv.reader(open(csv_path, newline=''))):
                try:
                    roww = ast.literal_eval(row[3])
                    sets.extend(roww)
                    print(roww)
                except:
                    continue

            # 创建独热编码

            one_hot_encodings = np.eye(len(set(sets)))

            # 将独热编码与对应的元素组合为字典
            one_hot_dict = dict(zip(set(sets), one_hot_encodings))
            
            title = ["img_path","label"]
            # for i in range(len(set(sets))):
            #     title.append('最像的類別(檢索第一名)')
            # for i in range(len(set(sets))):
            #     title.append('出現最多的類別(出現最多次)')
            for i in range(len(set(sets))):
                title.append('機率')
            csv.writer(open(Retrieval_feature_path, 'w', newline='',encoding='utf-8')).writerow(title)



            # for i in one_hot_dict:
            #     print(i,one_hot_dict[i])
            count =  -1 #count init
            for row in (csv.reader(open(csv_path, newline=''))):
                try:

                    
                    ML_sub_list = []
                    #print(porb_feature_list[count])
                    img_name = str(row[2])[3:-2]
                    roww = ast.literal_eval(row[3])
                    element_count = Counter(roww)
                    most_common_element = element_count.most_common(1)
                    

                    ML_sub_list.append(img_name)#影像名稱
                    ML_sub_list.append(answer_list[img_name])
                    #ML_sub_list.extend(one_hot_dict[roww[0]].tolist())#最像的類別
                    #ML_sub_list.extend(one_hot_dict[most_common_element[0][0]].tolist())#出現最多的類別

                    count += 1
                    if porb_feature_list[count][0] == img_name:
                        porb_feature_list[count].pop(0)
                        ML_sub_list.extend(porb_feature_list[count])
                    
                    csv.writer(open(Retrieval_feature_path, 'a+', newline='')).writerow(ML_sub_list)

                    print('make_Probability_feature : ',count)
                    

                except:
                    continue

            return None

        def make_Retrieval_feature(retrieval_detail_result=str,model_path=str,ML_feature_save_path=str,query_dataset_csv_path = str,answer_list = dict):
            model = torch.load(model_path).cuda()
            csv_path = query_dataset_csv_path



            #使用檢索細節表
            csv_path = retrieval_detail_result

            #出現最多的類別(儲存位置)
            Retrieval_feature_path = ML_feature_save_path +  '/Retrieval.csv'

            ML_list = []
            sets =[]
           # csv.writer(open(the_Most_Similar_Category, 'w', newline='',encoding='utf-8')).writerow(['img_path','最像的類別'])
      

            for count ,row in enumerate(csv.reader(open(csv_path, newline=''))):
                try:
                    roww = ast.literal_eval(row[3])
                    sets.extend(roww)
                    #print(roww)
                except:
                    continue

            # 创建独热编码

            one_hot_encodings = np.eye(len(set(sets)))

            # 将独热编码与对应的元素组合为字典
            one_hot_dict = dict(zip(set(sets), one_hot_encodings))
            
            title = ["img_path","label"]
            for i in range(len(set(sets))):
                title.append('最像的類別(檢索第一名)')
            for i in range(len(set(sets))):
                title.append('出現最多的類別(出現最多次)')
        
            csv.writer(open(Retrieval_feature_path, 'w', newline='',encoding='utf-8')).writerow(title)



            # for i in one_hot_dict:
            #     print(i,one_hot_dict[i])
            count =  -1 #count init
            for row in (csv.reader(open(csv_path, newline=''))):
                try:

                    count += 1
                    ML_sub_list = []
                    #print(porb_feature_list[count])
                    img_name = str(row[2])[3:-2]
                    roww = ast.literal_eval(row[3])
                    element_count = Counter(roww)
                    most_common_element = element_count.most_common(1)
                    

                    ML_sub_list.append(img_name)#影像名稱
                    ML_sub_list.append(answer_list[img_name])
                    ML_sub_list.extend(one_hot_dict[roww[0]].tolist())#最像的類別
                    ML_sub_list.extend(one_hot_dict[most_common_element[0][0]].tolist())#出現最多的類別

                    
                    csv.writer(open(Retrieval_feature_path, 'a+', newline='')).writerow(ML_sub_list)

                    print('make_Retrieval_feature : ',count)
                    

                except:
                    continue

            return None

        def make_Probability_Retrieval_feature(retrieval_detail_result=str,model_path=str,ML_feature_save_path=str,query_dataset_csv_path = str,answer_list = dict):
            model = torch.load(model_path).cuda()
            csv_path = query_dataset_csv_path

            count = 0 
            porb_feature_list = []
            for id,row in enumerate(csv.reader(open(query_dataset_csv_path, newline=''))):
                count += 1
            for id,row in enumerate(csv.reader(open(query_dataset_csv_path, newline=''))):
                try:
                    porb_list = []
                    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.5]*3, [0.5]*3)])
                    img_name = str(row[2])
                    img = torch.unsqueeze(transform(Image.open(img_name)), dim=0).cuda()
                    output = torch.softmax(model(img),dim=1).flatten().tolist()
                    porb_list.append(img_name)
                    porb_list.extend(output)
                    print('make_Probability_Retrieval_feature --> Probability : ',id)
                    porb_feature_list.append(porb_list)
                    #if self.test_mode:break
                except:
                    continue
          #  print(porb_feature_list[0])

            #使用檢索細節表
            csv_path = retrieval_detail_result

            #出現最多的類別(儲存位置)
            Retrieval_feature_path = ML_feature_save_path +  '/Probability_Retrieval.csv'

            ML_list = []
            sets =[]
           # csv.writer(open(the_Most_Similar_Category, 'w', newline='',encoding='utf-8')).writerow(['img_path','最像的類別'])
      

            for count ,row in enumerate(csv.reader(open(csv_path, newline=''))):
                try:
                    roww = ast.literal_eval(row[3])
                    sets.extend(roww)
                    #print(roww)
                except:
                    continue

            # 创建独热编码

            one_hot_encodings = np.eye(len(set(sets)))

            # 将独热编码与对应的元素组合为字典
            one_hot_dict = dict(zip(set(sets), one_hot_encodings))
            
            title = ["img_path","label"]
            for i in range(len(set(sets))):
                title.append('最像的類別(檢索第一名)')
            for i in range(len(set(sets))):
                title.append('出現最多的類別(出現最多次)')
            for i in range(len(set(sets))):
                title.append('機率')
            csv.writer(open(Retrieval_feature_path, 'w', newline='',encoding='utf-8')).writerow(title)



            # for i in one_hot_dict:
            #     print(i,one_hot_dict[i])
            count =  -1 #count init
            for row in (csv.reader(open(csv_path, newline=''))):
                try:

                    
                    ML_sub_list = []
                    #print(porb_feature_list[count])
                    img_name = str(row[2])[3:-2]
                    
                    roww = ast.literal_eval(row[3])
                    element_count = Counter(roww)
                    most_common_element = element_count.most_common(1)
                    

                    ML_sub_list.append(img_name)#影像名稱
                    ML_sub_list.append(answer_list[img_name])
                    ML_sub_list.extend(one_hot_dict[roww[0]].tolist())#最像的類別
                    ML_sub_list.extend(one_hot_dict[most_common_element[0][0]].tolist())#出現最多的類別

                    count += 1
                    if porb_feature_list[count][0] == img_name:
                        porb_feature_list[count].pop(0)
                        ML_sub_list.extend(porb_feature_list[count])
                    
                    csv.writer(open(Retrieval_feature_path, 'a+', newline='')).writerow(ML_sub_list)

                    print('make_Probability_Retrieval_feature : ',count)
                    

                except:
                   continue

            return None


       # answer_path = filedialog.askopenfilename()
        answer_path = self.ML_answer
        answer_list = dict()
        for id,row in enumerate(csv.reader(open(answer_path, newline=''))):
            if id == 0:continue
            answer_list[row[0]] = row[1] 
        for train_type in (self.train_typee):
            for model_list in (self.model_listt):
                for path_list in (self.path_listt):
                    for k in range(self.K):
                        if self.retireve_fold == -1:
                                print('all')
                        elif self.retireve_fold != k:
                                continue
                        if not os.path.exists(self.project_name + '/ML_feature/' + train_type +'/'+path_list+'/'+model_list+'/fold'+str(k)):
                            os.makedirs(self.project_name + '/ML_feature/' + train_type +'/'+path_list+'/'+model_list+'/fold'+str(k))
 
                        retrieval_detail_result = self.project_name +'/'+ train_type +'_retrieve/'+path_list+'/'+model_list+'/fold'+str(k)+'/top_10/output.csv'
                        model_path = self.project_name +'/'+ train_type +'/'+path_list+'/'+model_list+'/'+str(k)+'.pth'
                        ML_feature_save_path = self.project_name + '/ML_feature/' + train_type +'/'+path_list+'/'+model_list+'/fold'+str(k)
                        query_dataset_csv_path = self.project_name +'/' +path_list+ '_query_dataset.csv'

                        make_Retrieval_feature(retrieval_detail_result,model_path,ML_feature_save_path,query_dataset_csv_path,answer_list)
                        make_Probability_feature(retrieval_detail_result,model_path,ML_feature_save_path,query_dataset_csv_path,answer_list)
                        make_Probability_Retrieval_feature(retrieval_detail_result,model_path,ML_feature_save_path,query_dataset_csv_path,answer_list)

        #make_ML_feature

    def ML(self,ML_path,ML_models):       
        def multi_ML(data_path = str,model_name = str,model= svm.SVC(kernel='rbf',probability=True), k = int,save_path = str,random_state = 42):
            '''
            input:
                從指定的 data_path 路徑讀取 CSV 檔案，該檔案包含特徵和答案。

                使用分類器建立模型。

                使用 k-fold 交叉驗證對模型進行評估，獲取每個折疊的評估指標。

                ex:
                    multi_ML('path.csv',svm.SVC(kernel='rbf',probability=True),10)

            output:
                計算模型的平均準確率（mean_accuracy）、平均敏感度（mean_sensitivity）、
                平均特異度（mean_specificity）、平均陽性預測值（mean_ppv）、
                平均陰性預測值（mean_npv）和平均 AUC（mean_auc）。

                將模型儲存為 model.plk 檔案。

                將計算得到的評估指標以及對應的值儲存為一個包含以下欄位的CSV 檔案：
                    mean_accuracy、mean_sensitivity、mean_specificity、mean_ppv、mean_npv、mean_auc。
            
            '''

            df = pandas.read_csv(data_path,skiprows=1)
            X,y = df.iloc[:, 2:].values , df.iloc[:, 1].values
            
            class_num = {}
            for i in range(len(y)):
                key =  {y[i]:1} if y[i] not in class_num else {y[i]:class_num[y[i]]+1}
                class_num.update(key)


            # 進行5折交叉驗證
            kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

            # 儲存各次驗證的預測結果和真實標籤
            predicted_result = []
            true_labels = []
            prob = []
            for train_index, test_index in kfold.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # 訓練模型
                model.fit(X_train, y_train)

                if ('SVC' in str(model)):
                    prob.append(model._predict_proba_lr(X_test))
                else:
                    prob.append(model.predict_proba(X_test))
                    
                # 預測測試集
                y_pred = model.predict(X_test)

                # 保存預測結果和真實標籤
                predicted_result.append(y_pred)
                true_labels.append(y_test)


            accuracy = []
            sensitivity = []
            specificity = []
            ppv = []
            npv = []
            auc = []
            for fold in range(k):
                #print(f'---------------------------fold-{fold}---------------------------')

                cnf_matrix = confusion_matrix(true_labels[fold], predicted_result[fold])
                #print(cnf_matrix)

                if len(set(true_labels[fold].tolist())) > 2:
                        TP = np.diag(cnf_matrix)
                        TN = np.diag(cnf_matrix).sum()-TP
                        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
                        FN = (cnf_matrix).sum() - TP -TN -FP
                else:
                        TP = cnf_matrix[1, 1]
                        TN = cnf_matrix[0, 0]
                        FP = cnf_matrix[0, 1]
                        FN = cnf_matrix[1, 0]

                #print('TP , TN , FP , FN : ',TP , TN , FP , FN)
                import pandas as pd
                
                # if fold == 4:
                #      #print()
                sensitivity.append(np.mean((TP / (TP + FN))))
                specificity.append(np.mean((TN / (FP + TN))))
                #nan 補 0 
                ppv.append(np.mean(pd.Series((TP / (TP + FP))).fillna(0).tolist()))
                npv.append(np.mean(np.mean(pd.Series((TN / (TN + FN))).fillna(0).tolist())))
                accuracy.append(np.mean(((TP + TN) / (TP + TN + FP + FN))))

                if len(set(true_labels[fold].tolist())) > 2:
                    auc.append(roc_auc_score(y_true = true_labels[fold], y_score = prob[fold],multi_class='ovo'))
                else:
                    auc.append(roc_auc_score(y_true = true_labels[fold], y_score = prob[fold][:,1],multi_class='ovo'))
                #print('accuracy : ',accuracy_score(true_labels[fold], predicted_result[fold]))
                #print('accuracy : ',(accuracy)[fold])
                #print('sensitivity : ',(sensitivity)[fold])
                #print('specificity : ',(specificity)[fold])
                #print('ppv : ',(ppv)[fold])
                #print('npv : ',(npv)[fold])
                #print("auc:", auc[fold])
                #print()

            #print('class : ',class_num)
            mean_accuracy = sum(accuracy) / len(accuracy)    
            mean_sensitivity = sum(sensitivity) / len(sensitivity)    
            mean_specificity = sum(specificity) / len(specificity)    
            mean_ppv = sum(ppv) / len(ppv)    
            mean_npv = sum(npv) / len(npv)    
            mean_auc = sum(auc) / len(auc) 

            #print('---------------------------kfold_Mean---------------------------')
            print('accuracy : ',accuracy_score([num for sublist in true_labels for num in sublist],[num for sublist in predicted_result for num in sublist]))
            #print('accuracy : ',(mean_accuracy))
            #print('sensitivity : ',(mean_sensitivity))
            #print('specificity : ',(mean_specificity))
            #print('ppv : ',(mean_ppv))
            #print('npv : ',(mean_npv))
            #print("auc:", mean_auc)

            std_accuracy = np.std(accuracy)
            std_sensitivity = np.std(sensitivity)
            std_specificity =  np.std(specificity)
            std_ppv =  np.std(ppv)
            std_npv = np.std(npv)
            std_auc = np.std(auc)

            #print('---------------------------kfold_Std---------------------------')
            #print('accuracy : ',(std_accuracy))
            #print('sensitivity : ',(std_sensitivity))
            #print('specificity : ',(std_specificity))
            #print('ppv : ',(std_ppv))
            #print('npv : ',(std_npv))
            #print("auc:", std_auc)
            #print('---------------------------End---------------------------')
            
            with open (save_path, 'a+', newline='') as csvfile:
                csv.writer(csvfile).writerow([model_name])
                csv.writer(csvfile).writerow(['class : ',class_num])
                csv.writer(csvfile).writerow(['',"accuracy","sensitivity","specificity","ppv","npv","auc"])
                csv.writer(csvfile).writerow(['mean : ',round(mean_accuracy,2),round(mean_sensitivity,2),round(mean_specificity,2),round(mean_ppv,2),round(mean_npv,2),round(mean_auc,2)])
                csv.writer(csvfile).writerow(['std : ',round(std_accuracy,2),round(std_sensitivity,2),round(std_specificity,2),round(std_ppv,2),round(std_npv,2),round(std_auc,2)])
                csv.writer(csvfile).writerow(['\n\n'])

            return mean_accuracy
        '''
        # 定義模型
        model = []
        model.append(svm.LinearSVC(random_state=42,max_iter=5000))
        model.append(KNeighborsClassifier())
        model.append(BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42))

        for i in model:
            acc = multi_ML(data_path = 'setting_result/ML/finetune/S/vit/fold0/Probability_Retrieval.csv',model = i,k = 5,save_path='setting_result/ML/finetune/S/vit/fold0/Probability_Retrieval22.csv',random_state = 3681)
        '''
        for train_type in (self.train_typee):
            for model_list in (self.model_listt):
                for path_list in (self.path_listt):
                    for k in range(self.K):
                        if self.retireve_fold == -1:
                                print('all')
                        elif self.retireve_fold != k:
                                continue
                        if not os.path.exists(self.project_name + '/ML_result/' + train_type +'/'+path_list+'/'+model_list+'/fold'+str(k)):
                            os.makedirs(self.project_name + '/ML_result/' + train_type +'/'+path_list+'/'+model_list+'/fold'+str(k))
                        data_path = self.project_name + '/ML_feature/' + train_type +'/'+path_list+'/'+model_list+'/fold'+str(k)
                        save_path = self.project_name + '/ML_result/' + train_type +'/'+path_list+'/'+model_list+'/fold'+str(k)
                        for csv_path in os.listdir(data_path):
                            # 定義模型
                            ML_model = []
                            
                            if "SVM" in ML_models:
                              ML_model.append(["SVM",svm.LinearSVC(random_state=42,max_iter=5000)])
                            if"Subspace_KNN" in ML_models:
                                ML_model.append(["Subspace_KNN" ,BaggingClassifier(estimator=KNeighborsClassifier(),random_state=42)])
                            if("Random_Forest" in ML_models):
                                ML_model.append(["Random_Forest",BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)])
                            if("ANN" in ML_models):
                                ML_model.append(["ANN",MLPClassifier(max_iter=1200,random_state=42)])
                            
                            for model in ML_model:
                                print(model[0])
                                multi_ML(data_path = data_path + '/'+ csv_path,model_name = model[0],model = model[1],k = 5,save_path=save_path + '/'+ csv_path)
    
    def inference(self,model_path,dataset_path):
       
        datamaker = MAKE_INFERENCE_DATASET(dataset_path,self.project_name)
        datamaker.make_input_excel()
        datamaker.save_sheet_as_csv('inference_dataset', self.project_name + '/temp/inference_dataset.csv')      
              

        #model = torch.load(model_path)
        dataset = ''
        dataset = dataset_path.split('/')[-1]
        # for i in os.listdir(dataset_path):
        #     dataset=i
        #     break
        input_csv_path = self.project_name + '/temp/inference_dataset.csv'
        df = pd.read_csv(input_csv_path)
        valid_loader = DataLoader(customDataset(df, shuffle = True,pathColumn0=1), batch_size=1)
        model = torch.load(model_path).cuda()
        model.eval()
        total_valid = 0
        correct_valid = 0
        valid_loss = 0
        y_true = []
        y_scores = []
        y_prob = []
        count = 0

        info = []
        print('start') 
        mission = tqdm(total=len(valid_loader))  
        for batch_idx, (name,data, target) in enumerate(valid_loader):
            mission.update()
            sub_info = []
            model.eval()
            data, target = Variable(data), (target)[0] 
            
            if torch.cuda.is_available():
                data, target = data.cuda(), target#.cuda()

            output = model(data)

            predict = torch.softmax(output, dim=1) 
            predict = torch.max(output.data, 1)[1]
            prob = torch.softmax(output, dim=1).view(output.shape[1])
            prob = prob.tolist()
            total_valid += 1#len(target)
            correct_valid += sum((predict == int(target)).float())

            y_true.extend((list([int(target)])))
            y_scores.extend(predict.tolist())
            y_prob.extend(prob)

            sub_info.append(str(name)[2:-2])
            sub_info.append(int(target))
            sub_info.append(int(predict))
            sub_info.extend(prob)
            info.append(sub_info)
        
            #if batch_idx % 1 == 0:
            count+=1
            
        valid_acc_ = 100 * (correct_valid / float(total_valid))
        # output_csv_path = self.project_name +'/external_dataset/' +dataset
        # if not os.path.exists(output_csv_path):os.makedirs(output_csv_path)

        # output_csv_path = output_csv_path + '/output.csv'
        # csv.writer(open(output_csv_path, 'w', newline='')).writerow(['name','target','predict','Probability'])
        # for i in info: csv.writer(open(output_csv_path, 'a+', newline='')).writerow(i)
        # csv.writer(open(output_csv_path, 'a+', newline='')).writerow(['acc',valid_acc_.item()])    

        output_excel_path = self.project_name + '/Inference.xlsx'
        sheetname = dataset_path.split('/')[-1]
        # 創建一個新的工作簿和工作表
        if not os.path.exists(output_excel_path):
            openpyxl.Workbook().save(output_excel_path)
        wb = openpyxl.load_workbook(output_excel_path, data_only=True)
        wb.create_sheet(sheetname)
        if 'Sheet'  in wb.sheetnames:wb.remove(wb['Sheet'])   

        ws = wb[sheetname]
        # 添加表頭
        ws.append(['name', 'target', 'predict', 'Probability'])
        # 添加數據行
        for row in info:ws.append(row)
        # 添加準確率行
        ws.append(['acc', '', '', valid_acc_.item()])
        # 保存工作簿到 Excel 文件
        
        wb.save(output_excel_path)
        shutil.rmtree(self.project_name+'/temp')

    def draw_confuse_matirx_AND_roc_img(self,model,loader,save_img_path,fold,input_csv_path):
        '''
        繪製混淆矩陣圖與ROC圖(ROC圖只能用於二分類)
        '''
        mapp = {}
        classes = []
        conut = 0

        for row_count, i in enumerate(open(input_csv_path.split('.csv')[0]+'_detail.csv', newline='')):
            if row_count < 2:continue
            if i not in classes and len(str(i).split(',')[0])>1:
                classes.append(str(i).split(',')[0])
            if  len(str(str(i).split(',')[0]))>1:
                mapp.update({str(i).split(',')[0].split(',')[0]:conut} )
        classes = list(set(classes))
        max_str = ''
        for i in classes:
            if len(i) > len(max_str):max_str = i 
        
        test_labels = []
        pred_labels = []
        y_prob = []
        model = torch.load(model)
        #print("validate",model,"fold:",fold)
        mission = tqdm(total=len(loader))
        for batch_idx, (name,data, target) in enumerate(loader):
            # if  self.test_mode and  batch_idx  == 10 :  break
            mission.update()
            model.eval()
            if self.input_two_img:
                target = torch.autograd.Variable(target).cuda()
                output = model(torch.autograd.Variable(data[0]).cuda(), 
                               torch.autograd.Variable(data[1]).cuda())
            else:
                data = Variable(data)
                target = Variable(target)
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                output = model(data) 
          

            
            predict = torch.max(output.data, 1)[1]
            
            prob = (torch.softmax(output, dim=1))
                
            test_labels.extend(target.tolist())
            pred_labels.extend(predict.tolist())
            y_prob.extend(prob.tolist())
        if len(classes) == 2 :
            draw_roc(test_labels,y_prob,save_img_path+'\\'+ str(fold)+'\ROC.png')
        else:
            draw_Multiclass_ROC(test_labels,y_prob,save_img_path+'\\'+ str(fold)+'\ROC.png',input_csv_path.split('.csv')[0]+'_detail.csv')
        cm = confusion_matrix(test_labels, pred_labels)
        # 繪製混淆矩陣
        #classes = int(next(csv.reader( open(input_csv_path.split('.csv')[0]+'_detail.csv', newline='')))[1])
        size = len(classes) * len(max_str)
        size = max(size,11)
        fig, ax = plt.subplots(figsize=(size,size))     
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, fontsize=40)
        ax.set_yticklabels(classes, fontsize=40)
        # 手动调整子图的边距
        fig.subplots_adjust(left=0.35, right=0.85, bottom=0.3, top=0.85)        

        ax.set_xlabel('\nPredicted Class\n', fontsize=50)
        # 设置垂直标签
        ax.set_ylabel('\nTrue Class\n', fontsize=50)#true_class
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        for test_label in range(len(classes)):
            for pred_label in range(len(classes)):
            #  print(pred_label,test_label)
            # print((len(classes)))
                ax.text(pred_label, test_label, "{:.0f}".format(
                    cm[test_label, pred_label]),
                    ha="center",
                    va="top",
                    color="black",
                    fontdict={'fontname': 'Times New Roman', 'fontsize': 70})       
        #ax.text(len(classes) - 0.5, -0.5, str(mapp), ha='right', va='center', fontsize=30, color='black')      
        plt.title("Confusion matrix\n\n", fontsize=50)
        # plt.colorbar(im)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=35)
        plt.savefig(save_img_path+'\\'+ str(fold)+'\Confusion matrix.png')
        plt.clf()
        plt.close(plt.figure())     
        plt.close('all')  
        pass #function fin    

if __name__ == '__main__':  
    shutil.rmtree('CAT_DOG')  
    # shutil.rmtree('CAT_DOG/temp')  
    #shutil.move('CAT_DOG\CAT_DOG_dataset.xlsx')        
    cb =CBMIR()
    cb.fast_loader = not True
    cb.input_two_img =  not True
    cb.test_mode =   True
    cb.data_path = 'cats_and_dogs'
    cb.path_listt = ['cats_and_dogs']
    cb.batch_size = 2 ** 4
    cb.model_listt = ['vit']
    cb.train_typee = ['finetune']
    cb.project_name = 'CAT_DOG'
    cb.max_epoch = 2
    cb.auto_train()
    cb.inference('CAT_DOG/finetune_classification/cats_and_dogs/vit/1.pth','C:/Users/yccha/Downloads/shauyu/git/UI/cats_and_dogs')

    cb.query_path = cb.data_path
    cb.target_path = cb.data_path
    cb.retireve_fold = 1
    cb.retireve()
    cb.make_query_img()
