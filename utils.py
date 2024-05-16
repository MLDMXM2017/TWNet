import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from PIL import Image, ImageDraw
import random
from keras.utils import to_categorical
import face_recognition


def to_pkl(content,pkl_file):
    """
    save data in pkl format
    @param content:data
    @param pkl_file: save path
    """
    with open(pkl_file, 'wb') as f:
        pickle.dump(content, f)
        f.close()

def TV_L1_os2(prev,curr,bound=15):
    """
    TV_L1 method 
    @param prev: Array, the first frame image array
    @param curr: Array, the current frame image array
    @param bound: Int, the bound
    @return: u,v,os; optical flow images
    """
    prev=cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr=cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    
    TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32
    
    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    
    u=flow[:, :, 0]
    v=flow[:, :, 1]
    
    (x,y)=u.shape
    u_x=u-u[:,[0]+[i for i in range(0,y-1)]]
    v_y=v-v[[0]+[i for i in range(0,x-1)],:]
    u_y=u-u[[0]+[i for i in range(0,x-1)],:]
    v_x=v-v[:,[0]+[i for i in range(0,y-1)]]
    os_=np.sqrt(u_x**2+v_y**2+0.5*(u_y+v_x)**2)
    
    u=u[:, : ,np.newaxis]
    v=v[:, : ,np.newaxis]
    os_=os_[:, : ,np.newaxis]
    return u,v,os_

def detectFace_loc(path):
    """
    Given a picture, it returns the top, right, bottom, left coordinates of the face in the picture
    @param path: image path
    @return: top, right, bottom, left coordinates of the face in the picture
    """
    image = cv2.imread(path)
    face_locations = face_recognition.face_locations(image)
    top, right, bottom, left = face_locations[0]
    return top, right, bottom, left

def generate_samm_trainset():
    """
    genarate the train data and the train label of SAMM,apex frame denotes 1,and other frames randomly sampled denotes 0.
    @return: the train set,Dataframe 
    """
    #handle csv
    df=pd.read_csv('data/casme_groundTruth.csv')
    df=df[['fold','sub_fold','Onset','Apex','Offset','type']]
    df=df.loc[df['type']=='macro-expression']
    df['last']=df['Offset']-df['Onset']
    df=df.loc[df['last']>=10]

    #generate the train set of SAMM dataset
    data=pd.DataFrame()
    for index, row in df.iterrows():
        if row['fold']=='none' or row['Apex']==0:
            continue
        if row['Apex']>=1000:
            apex_path=os.path.join('img'+str(row['Apex'])+'.jpg')
        else:
            apex_path=os.path.join('img'+str(row['Apex']).zfill(3)+'.jpg')
        first_path=os.path.join('img'+str(1).zfill(3)+'.jpg')
        if row['Onset']-20>=1000:
            path1=os.path.join('img'+str(row['Onset']-20)+'.jpg')
        else:
            path1=os.path.join('img'+str(row['Onset']-20).zfill(3)+'.jpg')
        path2=os.path.join('img'+str(25).zfill(3)+'.jpg')
        path3=os.path.join('img'+str(50).zfill(3)+'.jpg')
        new= pd.DataFrame({"fold":row['fold'],"sub_fold":row['sub_fold'],"onset":first_path,"frame":apex_path,"label":1},index=["0"])
        data=data.append(new,ignore_index=True)
        new= pd.DataFrame({"fold":row['fold'],"sub_fold":row['sub_fold'],"onset":first_path,"frame":path1,"label":0},index=["0"])
        data=data.append(new,ignore_index=True)
        new= pd.DataFrame({"fold":row['fold'],"sub_fold":row['sub_fold'],"onset":first_path,"frame":path2,"label":0},index=["0"])
        data=data.append(new,ignore_index=True)
        new= pd.DataFrame({"fold":row['fold'],"sub_fold":row['sub_fold'],"onset":first_path,"frame":path3,"label":0},index=["0"])
        data=data.append(new,ignore_index=True)
    
    tmp=pd.read_csv('data/casme_face_loc.csv')
    for index, row in tmp.iterrows():
        if len(df.loc[(df['fold']==row['fold'])&(df['sub_fold']==row['sub_fold'])])==0:
            first_path=os.path.join('img'+str(1).zfill(3)+'.jpg')
            path1=os.path.join('img'+str(25).zfill(3)+'.jpg')
            path2=os.path.join('img'+str(50).zfill(3)+'.jpg')
            path3=os.path.join('img'+str(75).zfill(3)+'.jpg')
            path4=os.path.join('img'+str(100).zfill(3)+'.jpg')
            new= pd.DataFrame({"fold":row['fold'],"sub_fold":row['sub_fold'],"onset":first_path,"frame":path1,"label":0},index=["0"])
            data=data.append(new,ignore_index=True)
            new= pd.DataFrame({"fold":row['fold'],"sub_fold":row['sub_fold'],"onset":first_path,"frame":path2,"label":0},index=["0"])
            data=data.append(new,ignore_index=True)
            new= pd.DataFrame({"fold":row['fold'],"sub_fold":row['sub_fold'],"onset":first_path,"frame":path3,"label":0},index=["0"])
            data=data.append(new,ignore_index=True)
            new= pd.DataFrame({"fold":row['fold'],"sub_fold":row['sub_fold'],"onset":first_path,"frame":path4,"label":0},index=["0"])
            data=data.append(new,ignore_index=True)
    data=data.drop_duplicates()
    return data

def load_samm_train_data():
    """
    load train set of SAMM dataset
    @return: u,v:List,dynamic image; img:List, static image; 
             label:List,label list; fold_idx: 5-fold cross-validation trn_idx and val_idx
    """
    trn_u_path='data/samm_train_u.pkl'
    trn_v_path='data/samm_train_v.pkl'
    trn_img_path='data/samm_train_img.pkl'
    trn_label_path='data/samm_train_label.pkl'
    trn_fold_path='data/samm_train_fold_idx.pkl'
    add=0
    if os.path.exists(trn_u_path) and os.path.exists(trn_v_path):
        #if pkl exists, directly read
        return pd.read_pickle(trn_u_path),pd.read_pickle(trn_v_path),pd.read_pickle(trn_img_path),pd.read_pickle(trn_label_path),pd.read_pickle(trn_fold_path)
    else:
        #else generate pkl,and return data
        prefix_path='data/Cas(me)^2'  #Put the data of casme2 in this path
        macro=generate_samm_trainset()
        u,v,img,label=[],[],[],[]
        #Read data in the dataFrame line by line
        for i in tqdm(macro.index):
            try:
                onset_path=os.path.join(prefix_path,macro.loc[i]['fold'],macro.loc[i]['sub_fold'],macro.loc[i]['onset'])
                frame_path=os.path.join(prefix_path,macro.loc[i]['fold'],macro.loc[i]['sub_fold'],macro.loc[i]['frame'])
                #generate the (u,v,f)
                t,r,b,l=detectFace_loc(onset_path)
                onset=cv2.imread(onset_path)
                onset = onset[t-add:b+add,l-add:r+add]
                onset=cv2.resize(onset,(160,160))
                prev=cv2.imread(frame_path)
                prev = prev[t-add:b+add,l-add:r+add]
                prev=cv2.resize(prev,(160,160))
                tmp_u,tmp_v,tmp_os=TV_L1_os2(prev,onset)
                    
                u.append(tmp_u)
                v.append(tmp_v)
                img.append(prev)
                label.append(macro.loc[i]['label'])
            except Exception as e:
                print(e)
                print(macro.loc[i]['frame'])
        u=np.array(u)
        v=np.array(v)
        img=np.array(img)
        label=np.array(label)
        #  Random disruption
        index = [i for i in range(len(u))]  
        random.shuffle(index) 

        u = u[index]
        v=v[index]
        img=img[index]
        label= label[index]
        #print(u.shape)
        #print(img.shape)
        split=[]
        skf = StratifiedKFold(n_splits=5,random_state=2020,shuffle=True)
        for trn_idx,val_idx in skf.split(u,label):
            split.append((trn_idx,val_idx))
        
        #to pkl
        to_pkl(u,trn_u_path)
        to_pkl(v,trn_v_path)
        to_pkl(img,trn_img_path)
        to_pkl(label,trn_label_path)
        to_pkl(split,trn_fold_path)
        return u,v,img,label,split
 
    
def generate_casme_trainset():
    """
    genarate the train data and the train label of CAS(ME)2,apex frame denotes 1,and other frames randomly sampled denotes 0.
    @return: the train set,Dataframe 
    """
    data=pd.DataFrame()
    df=pd.read_csv('data/SAMM_groundTruth.csv')
    #df=df.loc[df['Type']=='Macro']
    for index, row in df.iterrows():
        if row['Apex']>10000 or row['Onset']>10000:
            print(row)
        apex_path=row['Subject']+'_'+str(row['Apex']).zfill(4)+'.jpg'
        first_path=row['Subject']+'_'+str(1).zfill(4)+'.jpg'
        path1=row['Subject']+'_'+str(row['Onset']-20).zfill(4)+'.jpg'
        path2=row['Subject']+'_'+str(25).zfill(4)+'.jpg'
        path3=row['Subject']+'_'+str(50).zfill(4)+'.jpg'
        if row['Type']=='Macro':
            new= pd.DataFrame({"Subject":row["Subject"],"onset":first_path,"frame":apex_path,"label":1},index=["0"])
            data=data.append(new,ignore_index=True)
        new= pd.DataFrame({"Subject":row["Subject"],"onset":first_path,"frame":path1,"label":0},index=["0"])
        data=data.append(new,ignore_index=True)
        new= pd.DataFrame({"Subject":row["Subject"],"onset":first_path,"frame":path2,"label":0},index=["0"])
        data=data.append(new,ignore_index=True)
        new= pd.DataFrame({"Subject":row["Subject"],"onset":first_path,"frame":path3,"label":0},index=["0"])
        data=data.append(new,ignore_index=True)
    
    #print(len(data))
    #print(data.head())
    data=data.drop_duplicates()
    #print(len(data))
    #print(data['label'].value_counts())
    return data
    
    
def load_casme_train_data():
    """
    load train set of CASME dataset
    @return: u,v:List,dynamic image; img:List, static image; 
             label:List,label list; fold_idx:5-fold cross-validation trn_idx and val_idx
    """
    trn_u_path='data/casme_train_u.pkl'
    trn_v_path='data/casme_train_v.pkl'
    trn_img_path='data/casme_train_img.pkl'
    trn_label_path='data/casme_train_label.pkl'
    trn_fold_path='data/casme_train_fold_idx.pkl'
    add=0
    if os.path.exists(trn_u_path) and os.path.exists(trn_v_path):
        #if pkl exists, directly read
        return pd.read_pickle(trn_u_path),pd.read_pickle(trn_v_path),pd.read_pickle(trn_img_path),pd.read_pickle(trn_label_path),pd.read_pickle(trn_fold_path)
    else:
        #else generate pkl,and return data
        prefix_path='data/SAMM'  #Put the data of samm in this path
        macro=generate_casme_trainset() #use the Cross dataset assessment,use the SAMM datset to be the train set of CAS(ME)2
        u,v,img,label=[],[],[],[]
        #Read data in the dataFrame line by line
        for i in tqdm(macro.index):
            try:
                onset_path=os.path.join(prefix_path,macro.loc[i]['Subject'],macro.loc[i]['onset'])
                frame_path=os.path.join(prefix_path,macro.loc[i]['Subject'],macro.loc[i]['frame'])
                #generate the (u,v,f)
                t,r,b,l=detectFace_loc(onset_path)
                onset=cv2.imread(onset_path)
                onset = onset[t-add:b+add,l-add:r+add]
                onset=cv2.resize(onset,(160,160))
                prev=cv2.imread(frame_path)
                prev = prev[t-add:b+add,l-add:r+add]
                prev=cv2.resize(prev,(160,160))
                tmp_u,tmp_v,tmp_os=TV_L1_os2(prev,onset)
                    
                u.append(tmp_u)
                v.append(tmp_v)
                img.append(prev)
                label.append(macro.loc[i]['label'])
            except Exception as e:
                print(e)
                print(macro.loc[i]['frame'])
        u=np.array(u)
        v=np.array(v)
        img=np.array(img)
        label=np.array(label)
        #  Random disruption
        index = [i for i in range(len(u))]  
        random.shuffle(index) 

        u = u[index]
        v=v[index]
        img=img[index]
        label= label[index]
        #print(u.shape)
        #print(img.shape)
        split=[]
        skf = StratifiedKFold(n_splits=5,random_state=2020,shuffle=True)
        for trn_idx,val_idx in skf.split(u,label):
            split.append((trn_idx,val_idx))
        
        #to pkl
        to_pkl(u,trn_u_path)
        to_pkl(v,trn_v_path)
        to_pkl(img,trn_img_path)
        to_pkl(label,trn_label_path)
        to_pkl(split,trn_fold_path)
        return u,v,img,label,split

