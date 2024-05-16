"""
the static module pre-training model, using revised Fer2013dataset
"""

import numpy as np
import pandas as pd
import cv2
from keras.utils import np_utils
from keras import backend as K
from keras import Input, Model
from keras.layers import Activation, Add,Dense, Conv2D, MaxPooling2D,Flatten,BatchNormalization,Dropout,concatenate,AveragePooling2D,MaxPool2D,UpSampling2D,Multiply,Lambda
from keras.initializers import RandomNormal
from keras import optimizers
from utils import to_pkl
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
K.set_image_data_format('channels_last')


def pre_staticModule(input_shape, n_class):
    """
    the pre-training static module model
    @param input_shape: the shape of input facial image,shape=(160,160,3)
    @param n_class: the number of categories classified, Fer2013 is 7
    @return: the model
    """
    input_ = Input(shape=input_shape)
    conv1= Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(input_)
    batch1=BatchNormalization()(conv1)
    max_pooling1=MaxPooling2D(pool_size=(3, 3),name='pool1')(batch1)
    drop1=Dropout(0.25)(max_pooling1)
            
    conv2= Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(drop1)
    batch2=BatchNormalization()(conv2)
    conv3= Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(batch2)
    batch3=BatchNormalization()(conv3)
    max_pooling2=MaxPooling2D(pool_size=(2, 2),name='pool2')(batch3)
    drop2=Dropout(0.25)(max_pooling2)
            
    conv4= Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(drop2)
    batch4=BatchNormalization()(conv4)
    conv5= Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(batch4)
    batch5=BatchNormalization()(conv5)
    max_pooling3=MaxPooling2D(pool_size=(2, 2),name='pool3')(batch5)
    drop3=Dropout(0.25)(max_pooling3)
            
    x4 = Flatten()(drop3)
    x = Dense(1024, activation='relu',name='fea')(x4)
    #pre-training model is multi classification problem
    output = Dense(n_class, activation='softmax', name='output')(x)
    model = Model(inputs=input_, outputs=output)
    return model



def train(model,data, args):
    """
    Training this pre-training model
    :param model: the pre-training model
    :param data: a tuple containing training and testing data, like ((x_train, y_train), (x_test, y_test))
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
     
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Training
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(x_test, y_test), callbacks=[log,checkpoint,lr_decay])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model
        
def load_Fer():
    """
    load revised Fer2013 data,split to 4:1 for train and test set
    @return :train data,train label and test data,test label
    """
    train_path='data/static_train_img.pkl'
    train_label='data/static_train_y.pkl'
    test_path='data/static_test_img.pkl'
    test_label='data/static_test_y.pkl'
    if os.path.exists(train_path) and os.path.exists(test_path):
        #if exists ,directly read data
        return pd.read_pickle(train_path),pd.read_pickle(train_label),pd.read_pickle(test_path),pd.read_pickle(test_label)
    else:
        #this csv includes two columns: pic_name and label, which is formed by the data selected by Fer2013 plus
        df=pd.read_csv('data/FerV2.csv')
        X=[]
        for i in tqdm(df.index):
            img = cv2.imread(df.loc[i]['pic_name'])
            X.append(img)
        y=list(df['label'])
        #split train and test set
        X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=2020,stratify=y)
        X_train=np.array(X_train)
        X_test=np.array(X_test)
        y_train=np.array(y_train)
        y_test=np.array(y_test)
        #print(X_train.shape)
        #print(y_test.shape)
        to_pkl(X_train,train_path)
        to_pkl(X_test,test_path)
        to_pkl(y_train,train_label)
        to_pkl(y_test,test_label)
        return X_train, y_train,X_test,y_test
        
if __name__ == "__main__":

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Pre-training Static Model.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--save_dir', default='./pre-training/static_result')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    x_train, y_train,x_test, y_test= load_Fer()
    tmp=pd.DataFrame()
    tmp['y']=y_test
    #print(tmp['y'].value_counts())
    x_train=np.array(x_train,dtype="float")/255.0
    x_test=np.array(x_test,dtype="float")/255.0
    y_train=np_utils.to_categorical(y_train)
    y_test=np_utils.to_categorical(y_test)
    
    #print(x_train.shape)
    #print(y_train.shape)
    
    # define model
    model = pre_staticModule(input_shape=x_train.shape[1:],n_class=7)
    model.summary()
    model=train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    
    #Remove the last output layer and save the parameters of the previous layer
    pre = Model(inputs=model.input,outputs=model.get_layer('fea').output)
    pre.save_weights(args.save_dir + '/fea_model.h5')