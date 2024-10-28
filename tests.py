import csv,h5py,torch,numpy as np

def ap_compute(query_path = str,target_path = str,save_path = str,top = int): 
    def mAP_output(path = str):
            num_classes = {}
            classes = {}
            mAP = 0
            with open(path+'/output.csv', newline='') as csvfile:
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

    top = top

    #----------------------------------------------------------
    #              input feature and path
    #----------------------------------------------------------
    import time
    seconds = time.time()
    feature_list = []
    feature_Mo_part = []
    query_list = []
    query_Mo_part = []
    #query_feature_list_DenseNet201_all.h5
    #with h5py.File('20.h5', "r") as k:
    

    with h5py.File(target_path, 'r') as file:
            #feature_label = np.array(file['Mo_part']).tolist()#.append(k.get('Mo_part')[i])
            feature_new_path = np.array(file['new_path']).tolist()
            feature_Mo_part = np.array(file['Mo_part']).tolist()#.append(k.get('new_path')[i])
            feature_list = np.array(file['feature'])#.tolist()#.append(k.get('feature')[i])
            print('feature ok !')

    with h5py.File(query_path, "r") as file: 
            query_new_path= np.array(file['new_path']).tolist()
            query_Mo_part = np.array(file['Mo_part']).tolist()
            query_list= np.array(file['feature'])
    

    #----------------------------------------------------------
    #               mAP compute
    #----------------------------------------------------------

    #將結果寫入csv
    #先宣告要用的欄位
    # ap , 分類 , 路徑
    with open(save_path+'/output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ap','answer','retrival_result','query_path','result_path'])
    print(len(feature_list))
    #mAP init
    mAP = 0
    device = torch.device("cpu")
    #開始跑每張圖
    query_list = torch.from_numpy(query_list).to(device)
    feature_list = torch.from_numpy(feature_list).to(device) 
    for j in range(len(query_list)-1):
       #continue
        print(j,'/',len(query_list),'query:',query_Mo_part[j],'path:',query_new_path[j])
    
        #查詢影像
        query = query_list[j]
        answer = query_Mo_part[j]

        #用來存放每張圖的cosin similarity
        score_map = {}
        score_map_path = {}

        #比對資料庫中的每一筆DATA
        # for i in range(len(feature_list)-1):
        #     #計算cosin similarity
        #     cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        #     #需要轉tensor
        #     cosine_similarity = cos((query),(feature_list[i]))
        #     #寫入 dic
        #     score_map.update({cosine_similarity:feature_Mo_part[i]})7
        from tqdm import tqdm
        mission = tqdm(total=len(feature_list)) 
        for i in range(len(feature_list)):
            mission.update()
            #計算cosin similarity
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            #需要轉tensor
            #query = torch.tensor.numpy()
            cosine_similarity = cos((query),(feature_list[i]))
            #寫入 dic
            score_map.update({cosine_similarity:feature_Mo_part[i]})
            score_map_path.update({cosine_similarity:i})


            #寫入 dic
            if (len(score_map) < top + 1):
                score_map.update({cosine_similarity:feature_Mo_part[i]})
                score_map_path.update({cosine_similarity:i})
                continue
            if (cosine_similarity<(min(score_map))):
                continue
            score_map.update({cosine_similarity:feature_Mo_part[i]})
            score_map_path.update({cosine_similarity:i})
            del score_map[min(score_map)]
            del score_map_path[min(score_map_path)]
           

        #將前 n 相似的輸出
        top_list = []
        top_list_path = []
        for i in range(top):
            #每次都挑最大的放入dic 概念類似選擇排序
            top_list.append(score_map[max(score_map)])
            top_list_path.append(score_map_path[max(score_map_path)])
            #將最大的刪除
            del score_map[max(score_map)]
            del score_map_path[max(score_map_path)]
        
        relevant = 0 
        #印出top-n串列
        for i in top_list:
            if answer == i:
                relevant = relevant + 1
            #print(query_path,top_list[i],(str(query_path)[1:-1]) == (str(top_list[i])[1:-1]))

        #print(relevant,top)
        ap = relevant/top
        #print(save_path.split("\\")[-3:-2])
        print('answer: ',answer)
        print('result:',top_list,answer)
        print('ap:',ap)
        mAP += ap/len(query_list)

        retrival_result_path = []
        for i in top_list_path:
             retrival_result_path.append(feature_new_path[i])
        #write in csv 
        with open(save_path+'/output.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ap,(str(answer)),str(top_list),query_new_path[j],retrival_result_path])
            csvfile.close()
        #break
    
    
        #writer.writerow([cost_time])
    
    mAP,classes = mAP_output(save_path)
    print('mAP : ',mAP)
    with open(save_path+'/output.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in classes:
                writer.writerow([row,classes[row]])
            writer.writerow(['mAP',mAP])
    return mAP


ap_compute(query_path = 'query_DenseNet201_fold3.h5',
           target_path ='query_DenseNet201_fold3.h5',
           save_path = 'retrieve_model/',top = 10)