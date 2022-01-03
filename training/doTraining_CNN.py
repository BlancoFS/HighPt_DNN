import os, sys, stat
import random
import pandas as pd
import numpy as np 
from keras import layers
from keras import models
from keras import backend as K
from keras import callbacks
from keras import optimizers
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from PIL import Image

# sys.argv[1]: nNeurons first layer
# sys.argv[2]: nhidden layers
# sys.argv[3]: learning rate 0.00005
# sys.argv[4]: batch size 1000

#train = pd.read_csv('train.csv')
#train = pd.read_csv('train_fortesting.csv')
train = pd.read_csv('train_binnedPt.csv')

#variablesTrain = ["Muon_InnerTrack_eta", "Muon_InnerTrack_phi", "Muon_InnerTrack_charge", "Muon_InnerTrack_pt",  "Muon_TunePTrack_pt", "Muon_DT_s1_nhits","Muon_DT_s1_x_mean","Muon_DT_s1_y_mean","Muon_DT_s1_z_mean","Muon_DT_s1_x_std","Muon_DT_s1_y_std","Muon_DT_s1_z_std","Muon_DT_s1_x_skew","Muon_DT_s1_y_skew","Muon_DT_s1_z_skew","Muon_DT_s1_x_kurt","Muon_DT_s1_y_kurt","Muon_DT_s1_z_kurt","Muon_DT_s2_nhits","Muon_DT_s2_x_mean","Muon_DT_s2_y_mean","Muon_DT_s2_z_mean","Muon_DT_s2_x_std","Muon_DT_s2_y_std","Muon_DT_s2_z_std","Muon_DT_s2_x_skew","Muon_DT_s2_y_skew","Muon_DT_s2_z_skew","Muon_DT_s2_x_kurt","Muon_DT_s2_y_kurt","Muon_DT_s2_z_kurt","Muon_DT_s3_nhits","Muon_DT_s3_x_mean","Muon_DT_s3_y_mean","Muon_DT_s3_z_mean","Muon_DT_s3_x_std","Muon_DT_s3_y_std","Muon_DT_s3_z_std","Muon_DT_s3_x_skew","Muon_DT_s3_y_skew","Muon_DT_s3_z_skew","Muon_DT_s3_x_kurt","Muon_DT_s3_y_kurt","Muon_DT_s3_z_kurt","Muon_DT_s4_nhits","Muon_DT_s4_x_mean","Muon_DT_s4_y_mean","Muon_DT_s4_x_std","Muon_DT_s4_y_std","Muon_DT_s4_x_skew","Muon_DT_s4_y_skew","Muon_DT_s4_x_kurt","Muon_DT_s4_y_kurt"]
variablesTrain = ["Muon_InnerTrack_eta", "Muon_InnerTrack_phi", "Muon_InnerTrack_charge", "Muon_InnerTrack_pt",  "Muon_TunePTrack_pt", "Muon_DT_s1_nhits","Muon_DT_s1_x_mean","Muon_DT_s1_y_mean","Muon_DT_s1_z_mean","Muon_DT_s1_x_std","Muon_DT_s1_y_std","Muon_DT_s1_z_std","Muon_DT_s1_x_skew","Muon_DT_s1_y_skew","Muon_DT_s1_z_skew","Muon_DT_s1_x_kurt","Muon_DT_s1_y_kurt","Muon_DT_s1_z_kurt","Muon_DT_s2_nhits","Muon_DT_s2_x_mean","Muon_DT_s2_y_mean","Muon_DT_s2_z_mean","Muon_DT_s2_x_std","Muon_DT_s2_y_std","Muon_DT_s2_z_std","Muon_DT_s2_x_skew","Muon_DT_s2_y_skew","Muon_DT_s2_z_skew","Muon_DT_s2_x_kurt","Muon_DT_s2_y_kurt","Muon_DT_s2_z_kurt","Muon_DT_s3_nhits","Muon_DT_s3_x_mean","Muon_DT_s3_y_mean","Muon_DT_s3_z_mean","Muon_DT_s3_x_std","Muon_DT_s3_y_std","Muon_DT_s3_z_std","Muon_DT_s3_x_skew","Muon_DT_s3_y_skew","Muon_DT_s3_z_skew","Muon_DT_s3_x_kurt","Muon_DT_s3_y_kurt","Muon_DT_s3_z_kurt","Muon_DT_s4_nhits","Muon_DT_s4_x_mean","Muon_DT_s4_y_mean","Muon_DT_s4_x_std","Muon_DT_s4_y_std","Muon_DT_s4_x_skew","Muon_DT_s4_y_skew","Muon_DT_s4_x_kurt","Muon_DT_s4_y_kurt", "binnedPt"]

image_index = train['Muon_Eventid'].to_numpy()

images = []

for i in image_index:
    image = np.array(Image.open('/gpfs/users/blancose/HighPT/CMSSW_11_0_3/src/HighPt_DNN/processing/train_' + str(i) + '.png').getdata())
    images.append(image)
    
  

genpT = ["Muon_Genpt"]

K.clear_session()


# Two Firsts Layer: CNN for Images and MLP for data 

#
# MLP:
#

def create_mpl(neurons):
    
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation='relu',input_dim=len(variablesTrain))) #, kernel_regularizer=l2(0.001)
    return model

#
# CNN
#

def create_cnn(dim):
    
    inputs = Input(shape=(256, 256, 4))
    
    model_cnn = Sequential()
    model_cnn.add(layers.Conv2D(256, (4, 4), activation='relu', inputs))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    model_cnn.add(layers.Flatten())
    model_cnn.add(layers.Dense(64, activation='relu'))
    
    model_2 = Model(inputs, model_cnn)
    
    return model_cnn
    


split = train_test_split(train[variablesTrain], images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split


mlp = models.create_mlp(int(sys.argv[1]))
cnn = models.create_cnn(256)

combinedInput = concatenate([mlp.output, cnn.output])


first = True

# Hidden layers
for i in range(0, int(sys.argv[2])):
    
    if i == 0:
        currentNeurons = int(sys.argv[1])
        continue
    
    if currentNeurons!=2:
        if random.choice([True, False]):
            if First:
                x = layers.Dense(currentNeurons, activation='relu')(combinedInput)
                First = False
                continue
            else:
                x = layers.Dense(currentNeurons, activation='relu')(x)
                continue
        else:
            currentNeurons = currentNeurons/2
            x = layers.Dense(currentNeurons, activation='relu')(x)
            continue
    else:
         x = layers.Dense(currentNeurons, activation='relu')(x)
         continue
            

x = layers.Dense(1, activation='linear')(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)


opt = optimizers.Adam(float(sys.argv[3]))
model.compile(loss="mean_squared_error", optimizer=opt)


history = model.fit(x=[trainAttrX, trainImagesX], y=train[genpT],validation_split=0.1, epochs=1000, batch_size=int(sys.argv[4]), verbose=0, callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=50,verbose=1)])


# Save the model

model.save('model_nFirstNeurons' + sys.argv[1] + '_nHiddenLayers_' + sys.argv[2] + '_LearningRate_' + sys.argv[3] + '_BatchSize_' + sys.argv[4] + '.h5')

# # Summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.ylabel('MSE',fontsize=10)
# plt.xlabel('Epoch',fontsize=10)
# plt.legend(['train', 'validation'], loc='upper right',fontsize=11)
# plt.tick_params(axis='x', labelsize=10)
# plt.tick_params(axis='y', labelsize=7)
# plt.yscale('log')


# plt.savefig('history/model_loss_nFirstNeurons' + sys.argv[1] + '_nHiddenLayers_' + sys.argv[2] + '_LearningRate_' + sys.argv[3] + '_BatchSize_' + sys.argv[4] + '.png')
