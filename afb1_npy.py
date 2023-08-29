from __future__ import print_function
## Package
import tensorflow as tf

import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sn
import scipy.io.wavfile
py.init_notebook_mode(connected=True)

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

import scipy.io as sio
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.io import arff
## Python
import random as rn
import sys
from sklearn import preprocessing
import glob
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# imports
# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import my_models

model_list1=['mini_nvidia_model','nvidia_model','sVGG','VGG16','VGG19','ResNet v2','ResNet v1','ConvSpeechModel','RNNSpeechModel','attRNNSpeechModel']

alist = pd.DataFrame(columns=['Model','Train', 'Test', 'Acc', 'Loss','Precision','Recall','F1','Kappa','ROC'])

nrows=40  # number of rows of feature vectors, columns are auto-adjusted
batch_size1 = 32
epoch1 = 100
test_size1=0.30
lr1=0.001;
factor1=0.90
patience1=20
acf='relu'
n1=12 # for resnet
modelstr1='cnn_'+str(lr1)+'_'+str(factor1)+'_'+str(patience1)+'_'+str(nrows)+'_'+str(epoch1)+'_'+str(batch_size1)+'_'+acf

i=1
# np.random.seed(1234)
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1 )
# sess = tf.compat.v1.Session( graph=tf.compat.v1.get_default_graph(), config=session_conf )
# tf.compat.v1.keras.backend.set_session(sess)

#-----------------------------Keras reproducibility------------------#
# SEED = 1234

# tf.set_random_seed(SEED)
# os.environ['PYTHONHASHSEED'] = str(SEED)
# np.random.seed(SEED)
# rn.seed(SEED)

# session_conf = tf.ConfigProto(
#     intra_op_parallelism_threads=1, 
#     inter_op_parallelism_threads=1
# )
# sess = tf.Session(
#     graph=tf.get_default_graph(), 
#     config=session_conf
# )
# K.set_session(sess)
#-----------------------------------------------------------------#
#list1=glob.glob("E:\\PhD\\thesis\\efb\\spc\\x\\*.npy")
#list1=glob.glob("E:\\PhD\\thesis\\emo\\emo2\\xx\\*.npy")
#list1=glob.glob("E:\\PhD\\thesis\\emo\\efb2\\x\\*.npy")
#list1=glob.glob("E:\\PhD\\thesis\\afb\\afb\\400-160\\x\\x\\*.npy")
#list1=glob.glob("E:\\PhD\\thesis\\emo\\emo1\\x\\*.npy")
#list1=glob.glob("E:\\PhD\\articles\\Arabian\\converted labels\\*.npy")
list1=glob.glob("E:\\PhD\\thesis\\afb\\afb\\400-160\\x\\x\\*.npy")

path1="E:\\PhD\\articles\\EFB\\springer\\data\\iemo impro aug\\"
path1="E:\\PhD\\articles\\EFB\\springer\\data\\downsampled\\"
path1="E:\\PhD\\articles\\EFB\\springer\\data\\iemo impro aug\\"
path1="E:\\PhD\\thesis\\afb\\afb\\400-160\\"

list1=[
 # 'spc digits_mel___emp3_d_400_160_100_norm_type_1.npy',
# 'efb2 spc all_efb_b11_x1c4__emp3_d_400_160_100_norm_type_1.npy',
# 'emo efb b100xt17 _emp3 _d -_cons_400_160__100_norm_type_1.npy',
# 'spc all_efb_b6_x1__emp3_d_400_160_100_norm_type_1.npy',
'spc all_efb_b6_x2__emp3_d_400_160_100_norm_type_1.npy',
 # 'spc all_efb_b7_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b8_x1c__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b8_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b9_x1c1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b9_x2__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b10_x1c3__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b10_x1c4__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b11_x1c__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b11_x1c1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b11_x1c2__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b11_x1c3__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b11_x1c4__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b11_x1c5__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b11_x1c6__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b12_x2__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b13_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b100xt17__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b14_x2__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b15_x2__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b16_x2__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b17_x2__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b18_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b19_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b20_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b21_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b22_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b23_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b24_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_efb_b25_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_mel___emp3_d_400_160_100_norm_type_1.npy',
# 'spc all_mfcc___emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b6_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b6_x2__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b7_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b8_x1c__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b8_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b9_x1c1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b9_x1c__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b10_x1c3__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b10_x1c4__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b11_x1c__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b11_x1c1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b11_x1c2__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b11_x1c3__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b11_x1c4__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b12_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b100xt17__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b13_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b14_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b15_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b16_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b17_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b18_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b19_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b20_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b21_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b22_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b23_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b24_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_efb_b25_x1__emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_mel___emp3_d_400_160_100_norm_type_1.npy',
# 'timit all_mfcc___emp3_d_400_160_100_norm_type_1.npy'       
]

for cm in range(1):   
    
    for st in list1: 
#    df=arff.loadarff(st)
        print(path1+st)
        x=np.load(path1+st)
        y=x[:,-1]
#    break        
        x=x[:,:-1]
#    break    
        x[np.isnan(x)] = 0

        #    x[~np.all(x == 0, axis=1)]
    #    xx=np.any(np.isnan(x))
    # normalize each column independently between [0,1]     
   
        min_max_scaler = preprocessing.MinMaxScaler()
        x= min_max_scaler.fit_transform(x)    
        xx=x
        # del data
        # feature count must be greater than 40 or so for the cnn
        a1=x.shape[1]
        a2=a1 % nrows
        a2=nrows-a2
        c1=np.zeros((x.shape[0],a2),dtype=int)
        x=np.concatenate((x,c1),axis=1)
        values, counts = np.unique(y, return_counts=True)
        n_classes=len(counts)
        ncols=int(x.shape[1]/nrows)
        x_train, x_test, y_train, y_test  = sklearn.model_selection.train_test_split(x, y, test_size=test_size1, random_state=1)
    #    xxx=x_train
        del x
        del y

        print('x_train shape:', x_train.shape)
        print('x_test shape :', x_test.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
           
        # input image dimensions
        img_rows, img_cols = nrows, ncols
        
        # if K.image_data_format() == 'channels_first':
        #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        #     input_shape = (1, img_rows, img_cols)
        # else:
        #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        #     input_shape = (img_rows, img_cols, 1)   
            
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows,img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows,img_cols)
            input_shape = (1, img_rows,img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows,img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows,img_cols, 1)
            input_shape = (img_rows,img_cols, 1)   

        
        # convert class vectors to binary class matrices
        y_train = y_train-1
        y_test = y_test-1
        y_test=y_test.astype(int)
        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)
        
        print('x_train shape:', x_train.shape)
        print('x_test shape :', x_test.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        
        # np.save("x_train25.npy",x_train)
        # np.save("x_test25.npy",x_test)
        # np.save("y_train25.npy",y_train)
        # np.save("y_test25.npy",y_test)
        
        cm=0
        if cm==0:
            model, str1 = my_models.mini_nvidia_model(input_shape, n_classes, acf)
        if cm==1:
            model, str1 = my_models.nvidia_model(input_shape, n_classes, acf)
        if cm==2:
            model, str1 = my_models.sVGG(input_shape, n_classes, acf)
        if cm==3:
            model, str1 = my_models.VGG16(input_shape, n_classes, acf)
        if cm==4:
            model, str1 = my_models.VGG19(input_shape, n_classes, acf)
        if cm==5:
            model, str1 = my_models.resnet_v2(input_shape, n_classes, acf, n1)

        # if cm==6:
        #     model, str1 = my_models.ConvSpeechModel(input_shape, n_classes, acf)
        # if cm==7:
        #     model, str1 = my_models.RNNSpeechModel(input_shape, n_classes, acf)
        # if cm==8:
        #     model, str1 = my_models.attRNNSpeechModel(input_shape, n_classes, acf)

#        model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
            #  optimizer = Adam(lr=1e-4)
        #    optimizer=keras.optimizers.RMSprop(lr=0.0001) #, rho=0.9, epsilon=None, decay=0.0)
        #    optimizer=keras.optimizers.Adadelta() #, epsilon=None, decay=0.0)
        #  optimizer=keras.optimizers.Adagrad(lr=0.001) #, epsilon=None, decay=0.0)
        #  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #  optimizer=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        #  optimizer=keras.optimizers.Nadam(lr=0.0001) #, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        # Model Training
        #    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=factor1, patience=patience1, min_lr=0.000000001, verbose=1)
        # Please change the model name accordingly.
        #    mcp_save = ModelCheckpoint('model/aug_noiseNshift_2class2_np.h5', save_best_only=True, monitor='val_loss', mode='min')
        #optimizer = Adam(lr=1e-4)
        # model.summary()

        # Prepare model saving directory.
#        save_dir = os.path.join(os.getcwd(), 'saved_models')
#        model_name = 'best_model'+ modelstr1
#        if not os.path.isdir(save_dir):
#            os.makedirs(save_dir)
#        filepath = os.path.join(save_dir, model_name)
        # Prepare callbacks for model saving and for learning rate adjustment.
#        checkpoint = ModelCheckpoint(filepath=filepath,
#                                     monitor='val_acc',
#                                     verbose=1,
#                                     save_best_only=True)
        
        optimizer1 = keras.optimizers.SGD(lr=lr1, momentum=0.0, decay=0.0, nesterov=True)
        optimizer1=keras.optimizers.Adam(lr=lr1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])

        opt_name='Adam';    
        #lr_scheduler = LearningRateScheduler(lr_schedule)
        #lr_reduce2 = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=factor1, patience=patience1, min_lr=0.000000001, verbose=1)
        #callbacks = [checkpoint, lr_reduce, lr_scheduler]
        callbacks = [lr_reduce]
        callbacks = []
        
        history=model.fit(x_train, y_train,
                      batch_size=batch_size1,
                      epochs=epoch1,
                      validation_data=(x_test, y_test),
                      shuffle=True,
                      callbacks=callbacks)
        score = model.evaluate(x_test, y_test, verbose=1)
        #    history=model.fit(x_train, y_train, batch_size = batch_size1, epochs=epoch1,  validation_data=(x_test, y_test), callbacks=[lr_reduce])
        #    score = model.evaluate(x_test, y_test, verbose=0)
        modelstr2=model_list1[cm]+'_'+modelstr1+'_'+opt_name

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        head1, tail1 = os.path.split(path1+st)

    # Plotting the Train Valid Loss Graph
        r1=8
        c1=6
        f1=16
        linew1=3
        plt.clf()    
        plt.figure(figsize=(r1,c1))
        plt.rcParams.update({'font.size': f1})       
        plt.plot(history.history['loss'],linewidth=linew1)
        plt.plot(history.history['val_loss'],linewidth=linew1)
        plt.title('Model Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'test'])
    #    plt.legend(['training', 'test'], loc='upper left')
    #    plt.show()
        plt.savefig(head1+'\\aloss_'+tail1+'_'+modelstr2+'.png')
        np.save(head1+'\\aloss_'+tail1+'_'+modelstr2+'.npy', history.history['loss'])
        np.save(head1+'\\aloss_val_'+tail1+'_'+modelstr2+'.npy', history.history['val_loss'])
        
        plt.clf()
        plt.figure(figsize=(r1,c1))
        plt.rcParams.update({'font.size': f1})       
        plt.plot(history.history['accuracy'],linewidth=linew1)
        plt.plot(history.history['val_accuracy'],linewidth=linew1)
        plt.title('Model Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train','test'])
    #    plt.show()
        plt.savefig(head1+'\\acc_'+tail1+'_'+modelstr2+'.png')
        np.save(head1+'\\acc_'+tail1+'_'+modelstr2+'.npy', history.history['accuracy'])
        np.save(head1+'\\acc_val_'+tail1+'_'+modelstr2+'.npy', history.history['val_accuracy'])
    #    plt.clf()
        print(head1+'\\aloss_'+tail1+'_'+modelstr2+'.png')
        
        y_pred = model.predict(x_test, batch_size = batch_size1)
        y_test1=np.argmax(y_test, axis=1)
    #    y_test1=y_test1.tolist()
        y_pred1 = np.argmax(y_pred, axis=1)
    #    y_pred1 = y_pred1.tolist()
        # predict probabilities for test set
        yhat_probs = y_pred
        # predict crisp classes for test set
        # yhat_classes = model.predict_classes(x_test, verbose=0)
        predict_x=model.predict(x_test) 
        yhat_classes=np.argmax(predict_x,axis=1)            
        
        # reduce to 1d array
        yhat_probs = yhat_probs[:, 0]
    #    yhat_classes = yhat_classes[:, 0]
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(y_test1, yhat_classes)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(y_test1, yhat_classes,average='weighted')
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(y_test1, yhat_classes,average='weighted')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test1, yhat_classes,average='weighted')
        print('F1 score: %f' % f1)
        # kappa
        kappa = cohen_kappa_score(y_test1, yhat_classes)
        print('Cohens kappa: %f' % kappa)
        # ROC AUC
    #    auc = roc_auc_score(y_test1, yhat_probs)
    #    print('ROC AUC: %f' % auc)
        # confusion matrix
        conf_mat = confusion_matrix(y_test1, yhat_classes)
        df=pd.DataFrame(data=conf_mat[0:,0:], index=[i for i in range(conf_mat.shape[0])], columns=['f'+str(i) for i in range(conf_mat.shape[1])])
        df.to_excel(head1+'\\aconf_mat_'+tail1+'_'+modelstr2+'.xlsx')        
        print(conf_mat)
#        np.savetxt(head1+'\\conf_mat_'+tail1+'_'+modelstr1+'.csv', conf_mat, '%s', delimiter=",")    
        
#        labels=['ANGRY','HAPPY','NEUTRAL','SAD']
#        print('_test_data_class: test function with y_test (actual values) and predictions (predict)')
#        _test_data_class(y_test1,y_pred1)   
        alist.loc[i]=[str1,tail1,'',score[1],score[0],precision,recall,f1,kappa,'']
#        alist.loc[i]=[str1,tail1,'',score[1],score[0],'','','','','']
        i=i+1
    #    np.savetxt(head1+'\\alist_'+'_'+modelstr1+'.csv', alist, '%s', delimiter="/t")
        alist.to_excel(head1+'\\alist_'+'_'+modelstr2+'.xlsx')