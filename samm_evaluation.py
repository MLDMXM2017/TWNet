"""
On SAMM data set, sequence prediction and evaluation to generate TP,FP,FN,F1-score and so on
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
from utils import to_pkl,TV_L1_os2,detectFace_loc
from interval import Interval
from peak_detection import find_peaks
warnings.filterwarnings("ignore")
tqdm.monitor_interval = 0

def save_data_to_pkl():
    """
    Read each video sequence in the data set,preprocess and produce optical flow image. Finally, u,v,f are saved as pkl
    """
    #save path of the pkl files of SAMM
    path='data/SAMM'
    result_path='data_result/samm_crop_result'
    folders=os.listdir(path)
    
    for fold in folders:
        print(fold+' done!')
        #the save path of u,v,img
        u_path=os.path.join(result_path,'SAMM_'+fold+'_u.pkl')
        v_path=os.path.join(result_path,'SAMM_'+fold+'_v.pkl')
        simg_path=os.path.join(result_path,'SAMM_'+fold+'_img.pkl')
        
        if os.path.exists(u_path) and os.path.exists(v_path) and os.path.exists(simg_path):
            continue
        #read data
        sub_path=os.path.join(path,fold)
        pics=os.listdir(sub_path)
        u,v,img=[],[],[]
        num=len(pics[0].split('.')[0].split('_')[2])
        t,r,b,l=detectFace_loc(os.path.join(sub_path,fold+'_'+str(1).zfill(num)+'.jpg'))
        for i in tqdm(range(1,len(pics)+1)):
            img_path =os.path.join(sub_path,fold+'_'+str(i).zfill(num)+'.jpg')
            sub_img_path=os.path.join(sub_path,fold+'_'+str(1).zfill(num)+'.jpg')
            
            onset=cv2.imread(sub_img_path)
            onset = onset[t:b,l:r]
            onset=cv2.resize(onset,(160,160))
            prev=cv2.imread(img_path)
            prev = prev[t:b,l:r]
            prev=cv2.resize(prev,(160,160))
            tmp_u,tmp_v,tmp_os=TV_L1_os2(prev,onset)
            
            u.append(tmp_u)
            v.append(tmp_v)
            img.append(prev)
            del tmp_u,tmp_v,prev
            gc.collect()
        u=np.array(u,dtype="float")/255.0
        v=np.array(v,dtype="float")/255.0
        img=np.array(img,dtype="float")/255.0
        #save data to pkl path
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
    model = load_model("/output/samm_result/fold"+str(fold_num)+"/trained_model_f"+str(fold_num)+".h5") 
    
    true=pd.read_csv('data/SAMM_groudTruth.csv')
    true['interval']=true['Offset']-true['Onset']
    true=true.loc[(true['Onset']!=0)&(true['Onset']!=1)] 
    
    path='data/SAMM'
    data_path='data_result/samm_crop_result'
    folders=os.listdir(path)
    for fold in folders:
        print('fold'+str(fold_num)+' '+fold+' done!')
        pred_sub_path=os.path.join(pred_path,'SAMM_'+fold+'_5fold_'+str(fold_num)+'.pkl')
        if os.path.exists(pred_sub_path):
            continue
            
        img_path=os.path.join(data_path,'SAMM_'+fold+'_img.pkl')
        u_path=os.path.join(data_path,'SAMM_'+fold+'_u.pkl')
        v_path=os.path.join(data_path,'SAMM_'+fold+'_v.pkl')
        
        new_img=joblib.load(img_path)
        new_u=joblib.load(u_path)
        new_v=joblib.load(v_path)
        pred=model.predict([new_u,new_v,new_img])
        
        to_pkl(pred,pred_sub_path)
        del new_u,new_v,new_img,pred
        gc.collect()
    del model
    gc.collect()

def get_macro_interval(pred,subject,h):
    """
    obtain the macro intervals of a video sequence which is found by our method
    @param pred: the pred pkl file path of this video
    @param subject: the subject of SAMM
    @param h: the height parameter
    @return :the macro intervals dataframe found
    """
    dis,low_w=280,30
    pred=list(pred[:,0])
    
    l=len(pred)
    if l>2500:
        inter=140
    else:
        inter=125
    result=pd.DataFrame()
    #get the spotting intervals
    peaks,proper = signal.find_peaks(pred,height=h,width=low_w,distance=dis)
    result['apex']=peaks
    result['Onset']=result['apex']-inter
    result['Offset']=result['apex']+inter
    result.loc[result['Onset']<=0,'Onset']=1
    result.loc[result['Offset']>=len(pred),'Offset']=len(pred)
    result['Type']='Macro'
    result['Subject']=subject
    result=result[['Onset','Offset','Type','Subject']]
    return result

def get_micro_interval(pred,subject,h): 
    """
    obtain the micro intervals of a video sequence which is found by our method
    @param pred: the pred pkl file path of this video
    @param subject: the subject of SAMM
    @param h: the height parameter
    @return :the micro intervals dataframe found
    """
    pred=list(pred[:,0]) 
    l=len(pred)
    if l>2500:
        inter=60
    else:
        inter=50
        
    result=pd.DataFrame()
    peaks,proper = signal.find_peaks(pred,height=(0,h),width=(1,100),distance=200)
    result['apex']=peaks
    result['Onset']=result['apex']-inter
    result['Offset']=result['apex']+inter
    result=result.loc[result['Offset']<=len(pred)]
    result=result.loc[result['Onset']>0]
    result['Type']='Micro - 1/2'
    result['Subject']=subject
    result=result[['Onset','Offset','Type','Subject']]
    return result

def get_TP_FP_and_result(df,Type):
    """
    Compare the spotting intervals with the ground-truth to calculate TP, FP, F1-score, etc
    @param df: the dataframe of the spotting intervals 
    @param Type: the type of expression,'Macro' or 'Micro - 1/2'
    @return :TP,FP,FN
    """
    #handle csv
    df = df.drop_duplicates(keep='first')
    true=pd.read_csv('data/SAMM_groudTruth.csv')
    true['interval']=true['Offset']-true['Onset']
    true=true.loc[(true['Onset']!=0)&(true['Onset']!=1)]
    # #calculate TP, FP, FN
    result=pd.DataFrame(columns=['Subject','TP','FP','FN','Type'])
    for i in true['Subject'].unique():
        tmp_pred_macro=df.loc[(df['Subject']==i)&(df['Type']==Type)]
        tmp_true_macro=true.loc[(true['Subject']==i)&(true['Type']==Type)]
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
        row={'Subject':i,'TP':a,'FP':n-a,'FN':m-a,'Type':Type}
        result=result.append(row,ignore_index=True)
    result.to_csv('best_result.csv',index=False)
    if Type=='Macro':
        result.to_csv('samm_macro.csv',index=False)
    else:
        result.to_csv('samm_micro.csv',index=False)
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
    pred_path='pred/samm_pred'
    Type='Macro'
    Type1='Micro - 1/2'
    h=0.21
    path='data/SAMM'
    folders=os.listdir(path)
    #get the sum dataframe of the macro spotting intervals
    df=pd.DataFrame()
    for fold in folders:
        pred=[]
        for i in range(5):
            if i!=6:
                tmp=joblib.load(os.path.join(pred_path,'SAMM_'+fold+'_5fold_'+str(i)+'.pkl'))
                tmp=np.around(tmp, decimals=5)
                pred.append(tmp)
        pred=np.array(pred)
        pred=np.mean(pred,0)
        tmp=get_macro_interval(pred,fold,h)
        df=df.append(tmp,ignore_index=True)
    #get the sum dataframe of the micro spotting intervals
    df1=pd.DataFrame()
    for fold in folders:
        pred=[]
        for i in range(5):
            if i!=6:
                tmp=joblib.load(os.path.join(pred_path,'SAMM_'+fold+'_5fold_'+str(i)+'.pkl'))
                tmp=np.around(tmp, decimals=5)
                pred.append(tmp)
        pred=np.array(pred)
        pred=np.mean(pred,0)
        tmp=get_micro_interval(pred,fold,h)
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
        predict_video_to_pkl(i,'pred/samm_pred')
    #get the result of all metrics
    get_result()