"""
the Network architecture of TWNet,and the training function of TWNet
"""

import numpy as np
from keras import backend as K
from keras import Input, Model
from keras.layers import Activation, Add,Dense, Conv2D, MaxPooling2D,Flatten,BatchNormalization,Dropout,concatenate,AveragePooling2D,MaxPool2D,UpSampling2D,Multiply,Lambda
from keras.initializers import RandomNormal
from keras import optimizers
from utils import load_samm_train_data,load_casme_train_data
import os
import argparse
from keras import callbacks
import h5py
K.set_image_data_format('channels_last')


def static_module(input_):
    """
    static module on TWNet.
    @param input_: the facial image,3d, shape=(160,160,3)
    @return: the 1024 channels static features
    """
    #Conv-1 Layer
    conv1= Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(input_)
    batch1=BatchNormalization()(conv1)
    max_pooling1=MaxPooling2D(pool_size=(3, 3),name='pool1')(batch1)
    drop1=Dropout(0.25)(max_pooling1)
    #Conv-2,3 Layer      
    conv2= Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(drop1)
    batch2=BatchNormalization()(conv2)
    conv3= Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(batch2)
    batch3=BatchNormalization()(conv3)
    max_pooling2=MaxPooling2D(pool_size=(2, 2),name='pool2')(batch3)
    drop2=Dropout(0.25)(max_pooling2)
    #Conv-4,5 Layer     
    conv4= Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(drop2)
    batch4=BatchNormalization()(conv4)
    conv5= Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(batch4)
    batch5=BatchNormalization()(conv5)
    max_pooling3=MaxPooling2D(pool_size=(2, 2),name='pool3')(batch5)
    drop3=Dropout(0.25)(max_pooling3)
            
    x = Flatten()(drop3)
    x = Dense(1024, activation='relu',name='Sf')(x)
    return x

def dynamic_module(input_u,input_v):
    """
    dynamic module on TWNet
    @param input_u: the u image of optical flow,shape=(160,160,1)
    @param input_v: the v image of optical flow,shape=(160,160,1)
    @return: the 1024 channels dynamic features
    """
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
    x = Dense(1024, activation='relu',name='Df')(x)
    return x

def TWNet(input_shape1=(160,160,3),input_shape2=(160,160,1)):
    """
    the Network architecture of TWNet
    @param input_shape1: the input shape of the static module ,default to (160,160,3), the same size in the actual experiment
    @param input_shape2: the input shape of the dynamic module ,default to (160,160,1), the same size in the actual experiment
    @return: the TWNet model
    """
    input_=Input(shape=input_shape1)
    input_u= Input(shape=input_shape2)
    input_v=Input(shape=input_shape2)
    
    
    static_output=static_module(input_)
    dynamic_output=dynamic_module(input_u,input_v)
    
    #use static and dynamic features to get the final probability
    x=concatenate([static_output, dynamic_output])
    x = Dense(256, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=[input_u,input_v,input_], outputs=[output])
    
    #model.summary()    
    return model

def train(model, data, args,fold):
    """
    the training function
    @param model: the TWNet model
    @param data: a tuple containing training and testing data, like `(trn_u,trn_v,trn_fea,trn_y), (val_u,val_v,val_fea,val_y)`
    @param args: arguments
    @param fold: the 5-fold cross-validation
    @return: The trained model
    """
    (trn_u,trn_v,trn_fea,trn_y), (val_u,val_v,val_fea,val_y) = data
    
    save_path=args.save_dir+'/fold'+str(fold)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # callbacks
    log = callbacks.CSVLogger(save_path + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(save_path + '/weights-{epoch:02d}.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Training without data augmentation:
    model.fit([trn_u,trn_v,trn_fea], trn_y, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=([val_u,val_v,val_fea], val_y), callbacks=[log,checkpoint,lr_decay])
    
    #save the whole model
    model.save(save_path + '/trained_model_f'+str(fold)+'.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    return model

def transfer(model,path,module='static'):
    """
    add the pre-training weights to static and dynamic module
    @param model: the initial  TWNet model
    @param path: the pre-training weights file path
    @param module: str,'static' or 'dynamic' denotes the static or dynamic module
    @return: the model after adding pre-training weights 
    """
    #read the pre-training weights file
    f=h5py.File(path)
    #layer_index is the position of the corresponding module in the model
    if module=='static':
        layer_index=[0,1,2,3,4,5,6,7,8,9,12,15,18,21,24,26,28,30,32]
    elif module=='dynamic':
        layer_index=[10,11,13,14,16,17,19,20,22,23,25,27,31,33,34,35,36]
    #add pre-training weights to model before training    
    l=f.attrs['layer_names']
    for i,element in enumerate(layer_index):
        if i!=0:
            weight_names=f[l[i]].attrs['weight_names']
            weights=[f[l[i]][j] for j in weight_names]
            model.layers[element].set_weights(weights)
            
    return model
    
if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="TWNet Network on SAMM and CASME2.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--dataset', default='SAMM', type=str,
                        help="The evaluationed dataset")
    parser.add_argument('--static_is_transfer', default=True, type=bool,
                        help="do static module use pre training parameters")
    parser.add_argument('--dynamic_is_transfer', default=True, type=bool,
                        help="do dynamic module use pre training parameters")
    #if samm,path to be output/samm_result
    parser.add_argument('--save_dir', default='./output/casme_result',
                        help="the save path of model saved")
    args = parser.parse_args()
    print(args)
    
    #pre-training weights file path
    static_pre_training_file='static_pre_training_weights.h5'
    dynamic_pre_training_file='dynamic_pre_training_weights.h5'
    
    #if not, create the save path
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.dataset=='SAMM':
        u,v,img,Y,fold_idx=load_samm_train_data()
    elif args.dataset=='CASME2':
        u,v,img,Y,fold_idx=load_casme_train_data()
    
    #handle the data
    u=np.array(u,dtype="float")/255.0
    v=np.array(v,dtype="float")/255.0
    img=np.array(img,dtype="float")/255.0
    Y=np.asarray(Y).astype('float32')
    
    #5 fold to train dataset
    accuracy=[]
    for fold,(trn_idx,val_idx) in enumerate(fold_idx):
        print('fold'+str(fold)+' start!')
        trn_u,trn_v,trn_fea,trn_y=u[trn_idx,:,:,:],v[trn_idx,:,:,:],img[trn_idx],Y[trn_idx]
        val_u,val_v,val_fea,val_y=u[val_idx,:,:,:],v[val_idx,:,:,:],img[val_idx],Y[val_idx]
            
        #model
        model = TWNet(input_shape1=(160,160,3),input_shape2=(160,160,1))
        #add transfer weights
        if args.static_is_transfer==True:
            model=transfer(model,static_pre_training_file,'static')
        elif args.dynamic_is_transfer==True:
            model=transfer(model,dynamic_pre_training_file,'dynamic')
        print('fold'+str(fold)+'add transfer weights success!')
        #train
        model=train(model=model, data=((trn_u,trn_v,trn_fea,trn_y), (val_u,val_v,val_fea,val_y)), args=args,fold=fold)