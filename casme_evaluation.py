"""
On CAS(ME)2 data set, sequence prediction and evaluation to generate TP,FP,FN,F1-score and so on
"""

from keras.models import load_model
import cv2
import numpy as np
import warnings
import os
import gc
from tqdm import tqdm
import joblib
import pandas as pd
from scipy import signal
from utils import to_pkl,TV_L1_os2
from interval import Interval
from peak_detection import find_peaks
warnings.filterwarnings("ignore")
tqdm.monitor_interval = 0

def save_data_to_pkl():
    """
    Read each video sequence in the data set,preprocess and produce optical flow image. Finally, u,v,f are saved as pkl
    """
    #save path of the pkl files of CAS(ME)2 
    result_path='data_result/casme_crop_result'
    face=pd.read_csv('casme_face_loc.csv')
    
    for index in face.index:
        fold=face.loc[index]['fold']
        sub_fold=face.loc[index]['sub_fold']
        #Get the face position of the first frame of each video sequence
        t,b,r,l=face.loc[index]['t'],face.loc[index]['b'],face.loc[index]['r'],face.loc[index]['l']
        path=os.path.join('data/Cas(me)^2',fold,sub_fold)
        pics=os.listdir(path)
        
        print(path+' done!')
        #save path of u,v,f
        u_path=os.path.join(result_path,fold+'_'+sub_fold+'_u.pkl')
        v_path=os.path.join(result_path,fold+'_'+sub_fold+'_v.pkl')
        simg_path=os.path.join(result_path,fold+'_'+sub_fold+'_img.pkl')
        
        if os.path.exists(u_path) and os.path.exists(v_path) and os.path.exists(simg_path):
            continue
        
        u=[]
        v=[]
        img=[]
        for i in tqdm(range(1,len(pics)+1)):
            if i>=1000:
                img_path =os.path.join(path,'img'+str(i)+'.jpg')
            else:
                img_path =os.path.join(path,'img'+str(i).zfill(3)+'.jpg')
            sub_img_path=os.path.join(path,'img001.jpg')
            
            onset=cv2.imread(sub_img_path)
            onset = onset[t:b,l:r]
            onset=cv2.resize(onset,(160,160))
            prev=cv2.imread(img_path)
            prev = prev[t:b,l:r]
            prev=cv2.resize(prev,(160,160))
            tmp_u,tmp_v=TV_L1_os2(prev,onset)
            
            u.append(tmp_u)
            v.append(tmp_v)
            img.append(prev)
            del tmp_u,tmp_v,prev
            gc.collect()
        #save u,v,f to pkl files    
        u=np.array(u,dtype="float")/255.0
        v=np.array(v,dtype="float")/255.0
        img=np.array(img,dtype="float")/255.0
        to_pkl(u,u_path)
        to_pkl(v,v_path)
        to_pkl(img,simg_path)
        del u,v,img
        gc.collect()
    
def predict_video_to_pkl(fold_num,pred_path):
    """
    Each video sequence of the data set is predicted and the probability sequence is saved as a pkl file
    @param fold_num:the 5-fold cross-validation
    @param pred_path:the save fold
    """
    #load model
    model = load_model("/output/casme_result/fold"+str(fold_num)+"/trained_model_f"+str(fold_num)+".h5")
    
    true=pd.read_csv('data/casme_groundTruth.csv')
    true['interval']=true['Offset']-true['Onset'] 
    
    face=pd.read_csv('casme_face_loc.csv')
    
    data_path='data/casme_crop_result'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
        
    #predict and save to pkl file
    for index in face.index:
        fold=face.loc[index]['fold']
        sub_fold=face.loc[index]['sub_fold']
        path=os.path.join('casme',fold,sub_fold)
        print('fold'+str(fold_num)+' '+path+' done!')
        
        pred_sub_path=os.path.join(pred_path,'casme_'+fold+'_'+sub_fold+'_5fold_'+str(fold_num)+'.pkl')
        if os.path.exists(pred_sub_path):
            continue
            
        img_path=os.path.join(data_path,fold+'_'+sub_fold+'_img.pkl')
        u_path=os.path.join(data_path,fold+'_'+sub_fold+'_u.pkl')
        v_path=os.path.join(data_path,fold+'_'+sub_fold+'_v.pkl')
        
        new_img=joblib.load(img_path)
        new_u=joblib.load(u_path)
        new_v=joblib.load(v_path)
        pred=model.predict([new_u,new_v,new_img])
        
        to_pkl(pred,pred_sub_path)
        del new_u,new_v,pred
        gc.collect()
    del model
    gc.collect()

def get_macro_interval(pred,fold,sub_fold,h):
    """
    obtain the macro intervals of a video sequence which is found by our method
    @param pred: the pred pkl file path of this video
    @param fold: the fold of CAS(ME)2
    @param sub_fold: the sub_fold of CAS(ME)2
    @param h: the height parameter
    @return :the macro intervals dataframe found
    """
    #get the probability sequence p
    pred=list(pred[:,0])
    tmp=pred.copy()
    pred=[]
    for i in range(5,len(tmp)):
        pred.append(np.mean(tmp[i-5:i])) 
    #get the intervals
    dis,low_w=80,10
    
    result=pd.DataFrame()
    peaks,proper = signal.find_peaks(pred,height=h,width=low_w,distance=dis)
    result['apex']=peaks+5
    result['Onset']=result['apex']-20
    result['Offset']=result['apex']+20
    result=result.loc[result['Offset']<=len(pred)]
    result=result.loc[result['Onset']>0]
    result['type']='macro-expression'
    result['fold']=fold
    result['sub_fold']=sub_fold
    result=result[['Onset','Offset','type','fold','sub_fold']]
    return result

def get_micro_interval(pred,fold,sub_fold,h): 
    """
    obtain the micro intervals of a video sequence which is found by our method
    @param pred: the pred pkl file path of this video
    @param fold: the fold of CAS(ME)2
    @param sub_fold: the sub_fold of CAS(ME)2
    @param h: the height parameter
    @return :the micro intervals dataframe found
    """
    pred=list(pred[:,0]) 
           
    result=pd.DataFrame()
    peaks,proper = signal.find_peaks(pred,height=(0,h),width=(1,20),distance=50)
    result['apex']=peaks
    result['Onset']=result['apex']-5
    result['Offset']=result['apex']+7
    result=result.loc[result['Offset']<=len(pred)]
    result=result.loc[result['Onset']>0]
    result['type']='micro-expression'
    result['fold']=fold
    result['sub_fold']=sub_fold
    result=result[['Onset','Offset','type','fold','sub_fold']]
    return result

def get_TP_FP_and_result(df,Type):
    """
    Compare the spotting intervals with the ground-truth to calculate TP, FP, F1-score, etc
    @param df: the dataframe of the spotting intervals 
    @param Type: the type of expression,'macro-expression' or 'micro-expression'
    @return :TP,FP,FN
    """
    
    df = df.drop_duplicates(keep='first')
    
    true=pd.read_csv('data/casme_groundTruth.csv')
    true['interval']=true['Offset']-true['Onset']
    
    result=pd.DataFrame(columns=['fold','sub_fold','TP','FP','FN','Type'])
    face=pd.read_csv('casme_face_loc.csv')
    #calculate TP, FP, FN
    for index in face.index:
        fold=face.loc[index]['fold']
        sub_fold=face.loc[index]['sub_fold']
        
        tmp_pred_macro=df.loc[(df['fold']==fold)&(df['sub_fold']==sub_fold)&(df['type']==Type)]
        tmp_true_macro=true.loc[(true['fold']==fold)&(true['sub_fold']==sub_fold)&(true['type']==Type)]
        m=len(tmp_true_macro)
        n=len(tmp_pred_macro)
        a=0
        tmp=[]
        for index, row in tmp_pred_macro.iterrows():
            tmp.append((row['Onset'],row['Offset']))
        for index, row in tmp_true_macro.iterrows():
            inter1 = Interval.between(row['Onset'],row['Offset'])
            flag=0
            for j in tmp:
                inter2=Interval.between(j[0],j[1])
                if inter1.overlaps(inter2)==True:
                    if inter1==inter2:
                      a+=1
                      flag=1
                    else:
                        it=inter1&inter2
                        X=it.upper_bound-it.lower_bound
                        L1=j[1]-j[0]
                        L2=row['Offset']-row['Onset']
                        #if k>0.5,it is a TP
                        if X/(L1+L2-X)>=0.5 and flag==0:
                            a+=1
                            flag=1
                        #else it is a FP
                        elif X/(L1+L2-X)>=0.5 and flag==1:
                            n-=1
        row={'fold':fold,'sub_fold':sub_fold,'TP':a,'FP':n-a,'FN':m-a,'Type':Type}
        result=result.append(row,ignore_index=True)
    #save the result to csv
    if Type=='macro-expression':
        result.to_csv('casme_macro.csv',index=False)
    else:
        result.to_csv('casme_micro.csv',index=False)
    TP,FP,FN=result.loc[result['Type']==Type][['TP','FP','FN']].sum()
    #print result
    print('###########################################################')
    print(result.loc[result['Type']==Type][['TP','FP','FN']].sum())
    print(Type+' Precision='+str(TP/(TP+FP)))
    print(Type+' Recall='+str(TP/(TP+FN)))
    print(Type+' F1_score:'+str(2*TP/(FP+FN+2*TP)))
    print('###########################################################')
    return TP,FP,FN

def get_result():
    """
    get the overall result of all metrics
    """
    #some parameters
    pred_path='pred/casme_pred'
    Type='macro-expression'
    Type1='micro-expression'
    face=pd.read_csv('casme_face_loc.csv')
    h=0.32
    
    #get the sum dataframe of the macro spotting intervals
    df=pd.DataFrame()
    for index in face.index:
        fold=face.loc[index]['fold']
        sub_fold=face.loc[index]['sub_fold']
        pred=[]
        for i in range(5):
            if i!=6:
                tmp=joblib.load(os.path.join(pred_path,'casme_'+fold+'_'+sub_fold+'_5fold_'+str(i)+'.pkl'))
                tmp=np.around(tmp, decimals=5)
                pred.append(tmp)
        pred=np.array(pred)
        pred=np.mean(pred,0)
        tmp=get_macro_interval(pred,fold,sub_fold,h)
        df=df.append(tmp,ignore_index=True)
        
    #get the sum dataframe of the micro spotting intervals
    df1=pd.DataFrame()
    for index in face.index:
        fold=face.loc[index]['fold']
        sub_fold=face.loc[index]['sub_fold']
        pred=[]
        for i in range(5):
            if i!=6:
                tmp=joblib.load(os.path.join(pred_path,'casme_'+fold+'_'+sub_fold+'_5fold_'+str(i)+'.pkl'))
                pred.append(tmp)
        pred=np.array(pred)
        pred=np.mean(pred,0)
        tmp=get_micro_interval(pred,fold,sub_fold,h)
        df1=df1.append(tmp,ignore_index=True)
    # calculate the sum TP, FP, F1-score, etc     
    TP1,FP1,FN1=get_TP_FP_and_result(df,Type)
    TP2,FP2,FN2=get_TP_FP_and_result(df1,Type1)
    TP=TP1+TP2
    FP=FP1+FP2
    FN=FN1+FN2
    print(TP,FP,FN)
    print('overall Precision='+str(TP/(TP+FP)))
    print('overall Recall='+str(TP/(TP+FN)))
    print('overall F1-score='+str(2.0*TP/(FP+FN+2*TP)))
  
if __name__ == "__main__":
    #preprocess data in video sequence and save to pkl file
    save_data_to_pkl()
    #predict and save P to pkl file
    for i in range(5):
        predict_video_to_pkl(i,'pred/casme_pred')
    #get the result of all metrics
    get_result()