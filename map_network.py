# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 08:43:52 2020

@author: boonping
"""

#Using tensorflow.keras for neural network.
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import Conv2D,Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation,ThresholdedReLU
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import add,subtract,Lambda,Concatenate,Multiply,Dot,Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras import optimizers
from tensorflow.keras import backend


#input does not has fixed shape. Changed to fixed shape only for layer illustration purpose.
#inp=Input(shape=(256,256,1,))
inpp=Input(shape=(None,None,1,))
#negInp=Input(shape=(256,256,1,))


#preprocessing layers - 
#maintaining scan points by creating mask. Masking output prediction for scanned point and re-insert back the values
#clip the max value to 1
#Using threshold Relu for low lying points.
mask=Lambda(lambda x: 1-x)(inpp)
mask1=ThresholdedReLU(theta=11/256)(mask)
mask1=Lambda(lambda x: x*25 )(mask1)
mask1=Lambda(lambda x: backend.clip(x,0,1) )(mask1)
mask=Lambda(lambda x: 1-x)(mask1)
inp0=Multiply()([mask,inpp])
inp1=Multiply()([mask1,inpp])
inp=inpp

#Stage 1-5 : Area identifier
#Stage 1
# Stage 1 MAx Pooling2D layer
rul=MaxPooling2D(pool_size=(64, 64), strides=(64,64),name="max_layer1")(inp)
xa0=rul
#preprocessing layers - 
#maintaining scan points by creating mask. Masking output prediction for scanned point and re-insert back the values
#clip the max value to 1
#Using threshold Relu for low lying points.
maska=Lambda(lambda x: 1-x)(rul)
maska1=ThresholdedReLU(theta=11/256)(maska)
maska1=Lambda(lambda x: x*25 )(maska1)
maska1=Lambda(lambda x: backend.clip(x,0,1) )(maska1)
maska=Lambda(lambda x: 1-x)(maska1)
xap0=Multiply()([maska,xa0])
xap1=Multiply()([maska1,xa0])

#Stage 1 Conv2D layer
xa0=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu',name="Conv2D_Layer1")(xa0)

#REverting prediction on scan point location and add back the scan value.
xa0=Multiply()([xa0,maska1])
xa0=add([xa0,xap0])

#Upsampling to fit next stage
xa0=UpSampling2D(size=(2, 2),name="UpSampling_Layer1")(xa0)
rul=UpSampling2D(size=(2, 2))(rul)
xa0=Lambda(lambda x: backend.clip(x,0,1) )(xa0)

###################################

#Stage 2
# Stage 2 MAx Pooling2D layer
xa=MaxPooling2D(pool_size=(32, 32), strides=(32,32),name="max_layer2")(inp)

#preprocessing layers - 
#maintaining scan points by creating mask. Masking output prediction for scanned point and re-insert back the values
#clip the max value to 1
#Using threshold Relu for low lying points.
maska=Lambda(lambda x: 1-x)(xa)
maska1=ThresholdedReLU(theta=11/256)(maska)
maska1=Lambda(lambda x: x*25 )(maska1)
maska1=Lambda(lambda x: backend.clip(x,0,1) )(maska1)
maska=Lambda(lambda x: 1-x)(maska1)
xap0=Multiply()([maska,xa])
xap1=Multiply()([maska1,xa])

#Concatenate with stage 1 output
xa=Concatenate()([xa0,xa])

#Stage 2 Conv2D layer
xa=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu',name="Conv2D_Layer2")(xa)

#REverting prediction on scan point location and add back the scan value.
xa=Multiply()([xa,maska1])
xa=add([xa,xap0])

#Upsampling to fit next stage
xa=UpSampling2D(size=(2, 2),name="UpSampling_Layer2")(xa)
rul=UpSampling2D(size=(2, 2))(rul)
xa=Lambda(lambda x: backend.clip(x,0,1) )(xa)

###################################
#Stage 3
# Stage 3 MAx Pooling2D layer
xb=MaxPooling2D(pool_size=(16, 16), strides=(16,16))(inp)

#preprocessing layers - 
#maintaining scan points by creating mask. Masking output prediction for scanned point and re-insert back the values
#clip the max value to 1
#Using threshold Relu for low lying points.
maskb=Lambda(lambda x: 1-x)(xb)
maskb1=ThresholdedReLU(theta=11/256)(maskb)
maskb1=Lambda(lambda x: x*25 )(maskb1)
maskb1=Lambda(lambda x: backend.clip(x,0,1) )(maskb1)
maskb=Lambda(lambda x: 1-x)(maskb1)
xbp0=Multiply()([maskb,xb])

#Concatenate with stage 2 output
xb=Concatenate()([xb,xa])
#Stage 3 Conv2D layer
xb=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(xb)

#REverting prediction on scan point location and add back the scan value.
xb=Multiply()([xb,maskb1])
xb=add([xb,xbp0])

#Upsampling to fit next stage
xb=UpSampling2D(size=(2, 2))(xb)
rul=UpSampling2D(size=(2, 2))(rul)
xb=Lambda(lambda x: backend.clip(x,0,1) )(xb)
#xb=ThresholdedReLU(theta=63/256)(xb)

###################################
#Stage 4
# Stage 4 MAx Pooling2D layer
rulc=MaxPooling2D(pool_size=(8, 8), strides=(8,8))(inp)
xc=rulc
#preprocessing layers - 
#maintaining scan points by creating mask. Masking output prediction for scanned point and re-insert back the values
#clip the max value to 1
#Using threshold Relu for low lying points.
maskc=Lambda(lambda x: 1-x)(xc)
maskc1=ThresholdedReLU(theta=11/256)(maskc)
maskc1=Lambda(lambda x: x*25 )(maskc1)
maskc1=Lambda(lambda x: backend.clip(x,0,1) )(maskc1)
maskc=Lambda(lambda x: 1-x)(maskc1)
xcp0=Multiply()([maskc,xc])
xcp1=Multiply()([maskc1,xc])

#Concatenate with stage 3 output
xc=Concatenate()([xc,xb])
#Stage 4 Conv2D layer
xc=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(xc)

#REverting prediction on scan point location and add back the scan value.
xc=Multiply()([xc,maskc1])
xc=add([xc,xcp0])

#Upsampling to fit next stage
xc=UpSampling2D(size=(2, 2))(xc)
rul=UpSampling2D(size=(2, 2))(rul)
xc=Lambda(lambda x: backend.clip(x,0,1) )(xc)

###################################
#Stage 5
# Stage 5 MAx Pooling2D layer
xd=MaxPooling2D(pool_size=(4, 4), strides=(4,4))(inp)

#preprocessing layers - 
#maintaining scan points by creating mask. Masking output prediction for scanned point and re-insert back the values
#clip the max value to 1
#Using threshold Relu for low lying points.
maskd=Lambda(lambda x: 1-x)(xd)
maskd1=ThresholdedReLU(theta=11/256)(maskd)
maskd1=Lambda(lambda x: x*25 )(maskd1)
maskd1=Lambda(lambda x: backend.clip(x,0,1) )(maskd1)
maskd=Lambda(lambda x: 1-x)(maskd1)
xdp0=Multiply()([maskd,xd])
xdp1=Multiply()([maskd1,xd])

#Concatenate with stage 4 output
xd=Concatenate()([xd,xc])
#Stage 5 Conv2D layer
xd=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(xd)

#REverting prediction on scan point location and add back the scan value.
xd=Multiply()([xd,maskd1])
xd=add([xd,xdp0])

#Upsampling to fit next stage
xd=UpSampling2D(size=(4, 4))(xd)

#Clipping value to prevent overflow
xd=Lambda(lambda x: backend.clip(x,0,1) )(xd)

###########################################

#Stage 6: Refining value layers

xa1=xd

#1st Conv2D layer
xa1=Conv2D(32,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(xa1)

#REverting prediction on scan point location and add back the scan value.
xa1=Multiply()([xa1,mask1])
xa1=add([xa1,inp0])
xa1=Lambda(lambda x: backend.clip(x,0,1) )(xa1)

#2nd Conv2D layer
xa1=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(xa1)

#REverting prediction on scan point location and add back the scan value.
xa1=Multiply()([xa1,mask1])
xa1=add([xa1,inp0])
xa1=Lambda(lambda x: backend.clip(x,0,1) )(xa1)

#3rd Conv2D layer
xa1=Conv2D(8,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(xa1)

#REverting prediction on scan point location and add back the scan value.
xa1=Multiply()([xa1,mask1])
xa1=add([xa1,inp0])
xa1=Lambda(lambda x: backend.clip(x,0,1) )(xa1)

#4th Conv2D layer
xa1=Conv2D(4,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(xa1)

#REverting prediction on scan point location and add back the scan value.
xa1=Multiply()([xa1,mask1])
xa1=add([xa1,inp0])
xa1=Lambda(lambda x: backend.clip(x,0,1) )(xa1)


#5th Conv2D layer
xa1=Conv2D(2,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(xa1)

#REverting prediction on scan point location and add back the scan value.
xa1=Multiply()([xa1,mask1])
xa1=add([xa1,inp0])
xa1=Lambda(lambda x: backend.clip(x,0,1) )(xa1)

xa1=ThresholdedReLU(theta=63/256)(xa1)



outp=xa1

model=Model(inpp,outp)
model.summary()
#raise
#model.compile(loss='mean_squared_error', optimizer = optimizers.RMSprop(), metrics=['mean_squared_error'])
model.compile(loss='mean_absolute_error', optimizer = optimizers.RMSprop(), metrics=['mean_squared_error','accuracy'])
#model.compile(loss='mean_squared_error', optimizer = optimizers.Adam(), metrics=['mean_squared_error','accuracy'])
#model.compile(loss='mean_squared_error', optimizer = optimizers.Adam(), metrics=['accuracy'])
#model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam() ,metrics=['accuracy'] )
#model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam() ,metrics=['accuracy'] )

modelname="map_network_new"

def lrSchedule(epoch):
    lr  = 1.5e-3
    lr=1.5e-3*(0.98**epoch)
    #if epoch<2: lr=0.8e-2

    
    if epoch > 195:
        lr  *= 1e-4
    elif epoch > 180:
        lr  *= 1e-3
        
    elif epoch > 160:
        lr  *= 1e-2
        
    elif epoch > 140:
        lr  *= 1e-1
        
    elif epoch > 120:
        lr  *= 2e-1
    elif epoch > 60:
        lr  *= 0.5
        
    print('Learning rate: ', lr)
    
    return lr

#6.2 For autoencoder classfier
def lrSchedule2(epoch):
    lr  = 0.15e-3
    if epoch > 59:
        lr  *= 1e-3
    elif epoch > 55:
        lr  *= 1e-2
        
    elif epoch > 52:
        lr  *= 2.5e-2
        
    elif epoch > 50:
        lr  *= 5e-2
        
    elif epoch > 48:
        lr  *= 0.1
    elif epoch > 45:
        lr  *= 0.4
    elif epoch > 30:
        lr  *= 0.7
    elif epoch > 25:
        lr  *= 0.8

        
        
    print('Learning rate: ', lr)
    
    return lr

#general setting for autoencoder training model
LRScheduler     = LearningRateScheduler(lrSchedule)

filepath        = modelname+"_classifier" + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger,LRScheduler]

reso=32

#Fitting 2D lidar map data compressed in the csv files.
with open("input_b.csv", "r") as f:
    samp=np.loadtxt(f)
with open("output_b.csv", "r") as f:
    result=np.loadtxt(f)

samp=samp.reshape(int(samp.shape[0]/256/256),256,256,1)
result=result.reshape(int(result.shape[0]/256/256),256,256,1)

samp/=256
result/=256

#9.3 Set the epoch to 60. As the encoding part is already been trained, it should converge faster to user defined classes.  
model.fit( samp, result,
           validation_data=(samp, result), 
           epochs=100, 
           batch_size=1,
           callbacks=callbacks_list)

#Save weights for testing usage
model.save_weights(modelname + "_weight.hdf5")
#print( model.predict(np.expand_dims(samp[1],axis=0)) )

#Training result illustration
res_model=model.predict( np.expand_dims(samp[20],axis=0) ) 
print(samp[20].shape,res_model[0].shape,result[20].shape)
print(np.nonzero(samp[20]))
print(np.nonzero(res_model[0]))

xxk,yyk,zzk=np.nonzero(samp[20])
print(res_model[0][xxk,yyk,zzk])
print(samp[20][xxk,yyk,zzk])

x=np.arange(256)
y=np.arange(256)

ax1 = plt.subplot(211)
ax1.set_aspect('equal')

# equivalent but more general
#ax1.pcolormesh(x,y,samp[1].reshape(256,256))
#ax2.pcolormesh(x,y,result[1].reshape(256,256))


ax2 = plt.subplot(212)
ax2.set_aspect('equal')
# add a subplot with no frame
ax2.pcolormesh(x,y,result[20].reshape(256,256))

ax3 = plt.subplot(222)
ax3.set_aspect('equal')
# add a subplot with no frame
ax3.pcolormesh(x,y,res_model[0][:,:,0].reshape(256,256))

plt.show()

