import numpy as np
from sklearn.metrics import confusion_matrix

def evaluation_index(true_labels = list,predicted_result = list, prob = list):
    '''
    input:
        true_labels      = [0,0,0,1,1,2]
        predicted_result = [1,2,0,1,1,1]
        evaluation_index(true_labels,predicted_result)
    output:
        計算模型的平均準確率（mean_accuracy）、
        平均敏感度（mean_sensitivity）、
        平均特異度（mean_specificity）、
        平均陽性預測值（mean_ppv）、
        平均陰性預測值（mean_npv）
    '''
    accuracy = []
    sensitivity = []
    specificity = []
    ppv = []
    npv = []

    
    cnf_matrix = confusion_matrix(true_labels, predicted_result)
    #print(cnf_matrix)
    '''
    TP (True Positive) 表示真陽性，表示模型正確地將正樣本分類為正樣本的數量。
    TN (True Negative) 表示真陰性，表示模型正確地將負樣本分類為負樣本的數量。
    FP (False Positive) 表示偽陽性，表示模型將負樣本錯誤地分類為正樣本的數量。
    FN (False Negative) 表示偽陰性，表示模型將正樣本錯誤地分類為負樣本的數量。
    '''
    if len(set(true_labels)) > 2 or len(set(predicted_result)) > 2:
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
    sensitivity.append(np.mean((TP / (TP + FN))))
    specificity.append(np.mean((TN / (FP + TN))))
    ppv.append(np.mean((TP / (TP + FP))))
    npv.append(np.mean((TN / (TN + FN))))
    accuracy.append(np.mean(((TP + TN) / (TP + TN + FP + FN))))
    return accuracy[0],sensitivity[0],specificity[0],ppv[0],npv[0]#,auc[0]

