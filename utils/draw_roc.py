from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt,csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.svm import SVC
import torch 

def draw_Multiclass_ROC(y_test = list,y_score= list,save_path = str,detail_csv_path = str):
    label_mapp = dict()
    with open(detail_csv_path, newline='') as csvfile:
        for id,row in enumerate(csv.reader(csvfile)):
            if id < 2 : continue
            label_mapp[id - 2] = row[0]
    #print(label_mapp,label_mapp[1])
    max_str = ''
    for i in label_mapp:
        if len(label_mapp[i]) > len(max_str):max_str = label_mapp[i] 


    num_samples= (len(set(y_test)))
    y_test = np.eye(num_samples)[y_test]
    y_test = np.array(y_test)
    y_score = np.array(y_score)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_samples):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    size = int(len(max_str) * 0.65)
    size = int(len(max_str) * 0.65)
    size = max(7,size)
    plt.figure(figsize=(size, size))
    lw = 2
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    colors = cycle(['#%06x' % np.random.randint(0, 0xFFFFFF) for _ in range(num_samples)])

    for i, color in zip(range(num_samples), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(label_mapp[i], roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(save_path)
    plt.close('all')



def draw_roc(y_true,y_scores,save_path):
    # 假设你的模型输出了概率值或分数，以及相应的真实标签
    #y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 2]
    y_scores  = [item[1] for item in y_scores]

    # 计算ROC曲线的各种参数
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    #plt.show()
    plt.savefig(save_path)
    plt.close('all')

# draw_Multiclass_ROC( torch.randint(0, 3, (100,)).tolist(), 
#                      torch.softmax(torch.randn(100,3),dim=0).tolist(),
#                      'ROC.png',
#                     'setting_result/NeckB_level1_4Position_dataset_detail.csv')
