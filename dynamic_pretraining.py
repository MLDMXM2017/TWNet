"""
the dynamic module pre-training model, using processed CK+ dataset
"""

import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras import backend as K
from keras import Input, Model
from keras.layers import Activation, Add,Dense, Conv2D, MaxPooling2D,Flatten,BatchNormalization,Dropout,concatenate,AveragePooling2D,MaxPool2D,UpSampling2D,Multiply,Lambda
from keras.initializers import RandomNormal
from keras import optimizers
from utils import to_pkl,TV_L1_os2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
K.set_image_data_format('channels_last')
import os
import argparse
from tensorflow.keras import callbacks

def pre_dynamicModule(input_shape):
    """
    the pre-training dynamic module model
    @param input_shape: the shape of input optical flow image u and v,shape=(160,160,1)
    @return :the model
    """
    input_u= Input(shape=input_shape)
    input_v=Input(shape=input_shape)
    initzers = RandomNormal(mean=0.0, stddev=0.01, seed=None)
    #the u path
    conv_u = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_initializer=initzers,
                    bias_initializer='zeros', padding='same')(input_u)
    batch_u = BatchNormalization()(conv_u)
    max_pooling_u = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(batch_u)
    drop_u = Dropout(0.5)(max_pooling_u)
    #the v path
    conv_v = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_initializer=initzers,
                    bias_initializer='zeros', padding='same')(input_v)
    batch_v = BatchNormalization()(conv_v)
    max_pooling_v = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(batch_v)
    drop_v = Dropout(0.5)(max_pooling_v)     
    
    #concate the two path        
    x = concatenate([drop_u, drop_v], axis=3)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu',name='fea')(x)
    #pre-training model is binary classification problem
    output = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=[input_u,input_v], outputs=output)
    return model



def train(model,data, args):
    """
    Training this pre-training model
    :param model: the pre-training model
    :param data: a tuple containing training and testing data, like ((trn_u,trn_v, trn_y), (val_u,val_v, val_y))
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (trn_u,trn_v, trn_y), (val_u,val_v, val_y) = data
     
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Training
    model.fit([trn_u,trn_v], trn_y, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=([val_u,val_v], val_y), callbacks=[log,checkpoint,lr_decay])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model
        
def load_CK():
    """
    load processed CK+ data,split to 4:1 for train and test set
    @return :train data(u and v),train label and test data(u and v),test label
    """
    train_u_path='data/dynamic_train_u.pkl'
    train_v_path='data/dynamic_train_v.pkl'
    train_label='data/dynamic_train_y.pkl'
    test_u_path='data/dynamic_test_u.pkl'
    test_v_path='data/dynamic_test_v.pkl'
    test_label='data/dynamic_test_y.pkl'
    if os.path.exists(train_u_path) and os.path.exists(test_u_path):
        #if exists ,directly read data
        return pd.read_pickle(train_u_path),pd.read_pickle(train_v_path),pd.read_pickle(train_label),pd.read_pickle(test_u_path),pd.read_pickle(test_v_path),pd.read_pickle(test_label)
    else:
        #this csv includes three columns: onset(onset frame path),frame(apex frame path) and label, which is formed by the data of processed Ck+ dataset
        df=pd.read_csv('data/ck+.csv')
        u,v=[],[]
        for i in tqdm(df.index):
            tmp_u,tmp_v=TV_L1_os2(df.loc[i]['onset'],df.loc[i]['frame'])
            u.append(tmp_u)
            v.append(tmp_v)
        u=np.array(u)
        v=np.array(v)
        y=list(df['label'])
        # put u and v to one array
        tmp=[]
        tmp.append(u)
        tmp.append(v)
        tmp=np.array(tmp)
        #split train and test set
        X_train,X_test, y_train, y_test =train_test_split(tmp,y,test_size=0.2, random_state=2020,stratify=y)
        #save u,v and label 
        u_train=np.array(X_train[0])
        v_train=np.array(X_train[1])
        u_test=np.array(X_test[0])
        v_test=np.array(X_test[1])
        y_train=np.array(y_train)
        y_test=np.array(y_test)
        #print(X_train.shape)
        #print(y_test.shape)
        to_pkl(u_train,train_u_path)
        to_pkl(u_test,test_u_path)
        to_pkl(u_train,train_v_path)
        to_pkl(u_test,test_v_path)
        to_pkl(y_train,train_label)
        to_pkl(y_test,test_label)
        return u_train,v_train, y_train,u_test,v_test,y_test
        
if __name__ == "__main__":

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Pre-training Dynamic Model.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--save_dir', default='./pre-training/dynamic_result')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    u_train,v_train, y_train,u_test,v_test, y_test= load_CK()
    
    ########show the Label distribution
    #tmp=pd.DataFrame()
    #tmp['y']=y_test
    #print(tmp['y'].value_counts())
    
    u_train=np.array(u_train,dtype="float")/255.0
    u_test=np.array(u_test,dtype="float")/255.0
    v_train=np.array(u_train,dtype="float")/255.0
    v_test=np.array(u_test,dtype="float")/255.0
    y_train=np_utils.to_categorical(y_train)
    y_test=np_utils.to_categorical(y_test)
    
    # define model
    model = pre_dynamicModule(input_shape=u_train.shape[1:],n_class=7)
    model.summary()
    model=train(model=model, data=((u_train,v_train, y_train), (u_test,v_test,y_test)), args=args)
    
    #Remove the last output layer and save the parameters of the previous layer
    pre = Model(inputs=model.input,outputs=model.get_layer('fea').output)
    pre.save_weights(args.save_dir + '/fea_model.h5')