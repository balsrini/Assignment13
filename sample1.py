# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 13:56:06 2019

@author: Balaji
"""

from keras import layers
from keras.layers import Input, Dense,SeparableConv2D,BatchNormalization,Activation,Dropout,Conv2D,concatenate,MaxPooling2D,Lambda,AveragePooling2D,Flatten
import tensorflow as tf
from keras.models import Model
import numpy as np
from keras.utils import np_utils

def Identity(x,f):
    return concatenate(x,f)

def Projection(input,output_channel,f):
    conv =  Conv2D(output_channel, (1,1), strides=2, use_bias=False)(input)
    return Identity(f,conv)

def Block(input,num_channel,kernel_size=(3,3),pad='same',stride=1):        
    bnOutput = BatchNormalization()(input)
    act = Activation('relu')(bnOutput)
    return Conv2D(num_channel, kernel_size, strides=stride, padding=pad, use_bias=False)(act)
    

def configBlock(blkIndex,count,input,num_channel,kernel_size=(3,3),pad='same'):            
    inputShape = input.shape
    for i in range(count):
        stride = 1
        if blkIndex > 0 and i == 0:
            stride = 2            
        inputBlk = Block(input,num_channel,stride=stride)        
        inputBlk2 = Block(inputBlk,num_channel)
        if inputShape == inputBlk2.shape :
            input = Identity(inputBlk2,input)
        else:
           input = Projection(input,num_channel,inputBlk2)
     
    return input
 
    

def processInput(input,num_channel=64,kernel_size=(7,7),pad='same',stride=2):
    conv =  Conv2D(num_channel, kernel_size, strides=stride, padding=pad,  use_bias=False)(input)    
    return MaxPooling2D(pool_size=(2, 2))(conv) 


def Restnet18(img_height, img_width, channel):
    input_layer = Input(shape=(img_height, img_width, channel))
    blockCount = [2,2,2,2]
    numChannels = [64,128,256,512]    
    output = input_layer
    for i in range(len(blockCount)):
        output = configBlock(i,blockCount[i],output,numChannels[i])
    
    model = Model(inputs=[input_layer], outputs=[output])
    model.summary()    
    return model

model = Restnet18(224,224,3)

    


    


