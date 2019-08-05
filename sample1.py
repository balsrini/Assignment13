# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 13:56:06 2019

@author: Balaji
"""

from keras import layers
from keras.layers import Input,add, Dense,SeparableConv2D,BatchNormalization,Activation,Dropout,Conv2D,concatenate,MaxPooling2D,Lambda,AveragePooling2D,Flatten
import tensorflow as tf
from keras.models import Model
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import time

#For use in add/concatenate
def Identity(x,f):    
    return add([x,f])

#For use in projection shortcuts
def Projection(x,output_channel,f):    
    return Identity(f,Conv2D(output_channel, (1,1), strides=2, use_bias=False)(x))    

#Typical of a Block containing BN,Activation and Relu
def Block(input,num_channel,kernel_size=(3,3),pad='same',stride=1):        
    bnOutput = BatchNormalization()(input)
    act = Activation('relu')(bnOutput)
    return Conv2D(num_channel, kernel_size, strides=stride, padding=pad, use_bias=False)(act)
    
def configBlock_Identity(input,num_channel,kernel_size=(3,3),pad='same'):            
    inputBlk = Block(input,num_channel)        
    inputBlk2 = Block(inputBlk,num_channel)
    return Identity(inputBlk2,input)

def configBlock_Projection(input,num_channel,kernel_size=(3,3),pad='same'):                
    inputBlk = Block(input,num_channel,stride=2)        
    inputBlk2 = Block(inputBlk,num_channel)     
    return Projection(input,num_channel,inputBlk2)

#First block in Resnet
def BlockA(count,input,num_channel,kernel_size=(3,3),pad='same'):
    for i in range(count):
        input = configBlock_Identity(input,num_channel)
        i = i + 1

    return input
        
#Remaining block in ResNet. The difference of stride of 2.         
def BlockB_to_D(count,input,num_channel,kernel_size=(3,3),pad='same'):    
    projBlk = configBlock_Projection(input,num_channel)    
    for i in range(count -1):
       projBlk = configBlock_Identity(projBlk,num_channel)
    return projBlk

#Input processing conv + maxpooling
def processInput(input,num_channel=64,kernel_size=(7,7),pad='same',stride=2):
    conv =  Conv2D(num_channel, kernel_size, strides=stride, padding=pad,  use_bias=False)(input)    
    return MaxPooling2D(pool_size=(2, 2))(conv) 

#Defining the Restnet18 model
def Restnet18(img_height, img_width, channel):
    #Define input
    input_layer = Input(shape=(img_height, img_width, channel))
    #Process input -- conv + max pool
    output = Lambda(processInput)(input_layer)   
    
    #defining the block count -- can be extended for future resnet    
    blockCount = [2,2,2,2]    
    
    #numchannel in each block
    numChannels = [64,128,256,512]        
    
    #First block is different from others.(stride + projection being used)
    for i in range(len(blockCount)):
        if i == 0:
            output = BlockA(blockCount[i],output,numChannels[i])
        else:
            output = BlockB_to_D(blockCount[i],output,numChannels[i])
    
    model = Model(inputs=[input_layer], outputs=[output])
    model.summary()    
    return model

#Reinit of the model
def reinit(normalize = True):
    from keras.datasets import cifar10
    (train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
    num_train, img_rows, img_cols,img_channels =  train_features.shape
    num_test, _, _, _ =  test_features.shape
    num_classes = len(np.unique(train_labels))
    if normalize:
        train_features = train_features.astype('float32')/255
        test_features = test_features.astype('float32')/255
        # convert class labels to binary class labels
        train_labels = np_utils.to_categorical(train_labels, num_classes)
        test_labels = np_utils.to_categorical(test_labels, num_classes)
    return (num_classes,train_features,train_labels,test_features,test_labels)

#Plot the model accuracy
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

#define the accuracy    
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)    




datagen = ImageDataGenerator(zoom_range=0.0, horizontal_flip=False)
(num_classes,train_features,train_labels,test_features,test_labels) = reinit()
model = Restnet18(224,224,3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""
# train the model
start = time.time()
# Train the model
model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128),
                                 samples_per_epoch = train_features.shape[0], nb_epoch = 100, 
                                 validation_data = (test_features, test_labels), verbose=1)
end = time.time()
print ("Model took %0.2f seconds to train"%(end - start))
# plot model history
plot_model_history(model_info)
# compute test accuracy
print ("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model))
"""
    


    


