#!/usr/bin/env python
# coding: utf-8

# # Code for diagnosing diabetic retinopathy using CNN


# This tells matplotlib not to try opening a new window for each plot.
#get_ipython().run_line_magic('matplotlib', 'inline')

# Import a bunch of libraries.
import time
import random
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import os
import sys
#from matplotlib.ticker import MultipleLocator
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

import pandas as pd
import cv2
import datetime
#import seaborn as sn

# Set the randomizer seed so results are the same each time.
np.random.seed(0)


# In[178]:


from keras.models import Sequential,Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import model_from_json
from keras import applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping



from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,RMSprop,SGD
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#from pyimagesearch.smallervggnet import SmallerVGGNet


# In[111]:


file_dir = os.path.dirname(os.path.abspath('__file__'))
print(file_dir)
sys.path.append(file_dir)


# In[110]:


os.path.dirname("__file__")


# In[3]:


# Load the csv data
orig_label = pd.read_csv("./Retinal-Images/trainLabels.csv")
orig_label.head(5)


# In[6]:


orig_label.count()


# In[5]:


orig_label.groupby(['level']).count()


# In[113]:


# In[135]:


image_list = list()
IMAGE_SAMPLE = 35000
# We create a toy dataset of 'N' images, maintaining the split of the original 
images_percent = [73.6,6.9,15.1,2.4,2]
for level in range(5):
    # Get respective number of images in each level
    number_of_images = int(images_percent[level]*IMAGE_SAMPLE/100)
    sample_images = orig_label[orig_label.level==level].sample(n=number_of_images,axis=0,random_state=42)
    for i in range(len(sample_images)):
        image_name = sample_images.iloc[i].image
        #image_name = sample_images.iloc[i].image+'.jpeg'
        image_list.append(image_name)
len(image_list)


# In[136]:


random.shuffle(image_list)
# 80% is trained
n_train = int(IMAGE_SAMPLE*0.94)
n_test = IMAGE_SAMPLE-n_train
train_images = image_list[:n_train]
test_images = image_list[n_train:]


def create_data_labels_aug(image_list,orientation):
    """ This function is to be used if one of the images is flipped, to maintain same orientation"""
    image_dir = os.getcwd()+'/Retinal-Images'+'/train_resize_224_flop/'
    data = []
    labels = []
    augmentation = "-flop"
    for filename in image_list:
        image_orientation = filename.split('.')[0].split('_')[1]
        if(image_orientation == orientation):
            imagePath = image_dir + filename + augmentation + ".jpeg"
        else:
            imagePath = image_dir + filename + ".jpeg"
        if os.path.isfile(imagePath):
            image = cv2.imread(imagePath)
            image = img_to_array(image)
            data.append(image)
            # Get the class label from the image and update the
            # labels list
            label = orig_label[orig_label.image==filename].level.values[0]
            # Change the label to binary
            if label>0:
                label=1
            labels.append(label)

    # Convert the values between 0-1
    #data = np.array(data, dtype="float") / 255.0
    data = np.array(data, dtype="int")
    labels = np.array(labels)
    return(data,labels)

# Load the data with image augmentation
# 128x128 data to start with
def create_data_labels_pp(image_list,pre_process=True):
    """ Function to load the data if augmentation is preprocessed"""
    image_dir = os.getcwd()+'/Retinal-Images'+'/train_resize_224/'
    data = []
    labels = []
    #aug_list = ["-flip","-flop","-rs"]
    aug_list = ["-flop"]
    for filename in image_list:
        if(pre_process):
            for aug_n in range(len(aug_list)):
                if aug_n!=0:
                    aug = aug_list[aug_n-1]
                    imagePath = image_dir + filename + aug + ".jpeg"
                else:
                    imagePath = image_dir + filename + ".jpeg"
                #print(imagePath)
                # load the image, pre-process it, and store it in the data list
                if os.path.isfile(imagePath):
                    image = cv2.imread(imagePath)
                    image = img_to_array(image)
                    data.append(image)
                    # Get the class label from the image and update the
                    # labels list
                    label = orig_label[orig_label.image==filename].level.values[0]
                    # Change the label to binary
                    if label>0:
                        label=1
                    labels.append(label)
        else:
            # For test, we do not need augmentation
            imagePath = image_dir + filename + ".jpeg"
            #print(imagePath)
            # load the image, pre-process it, and store it in the data list
            if os.path.isfile(imagePath):
                image = cv2.imread(imagePath)
                image = img_to_array(image)
                data.append(image)
                # Get the class label from the image and update the
                # labels list
                label = orig_label[orig_label.image==filename].level.values[0]
                # Change the label to binary
                if label>0:
                    label=1
                labels.append(label)

    # Convert the values between 0-1
    #data = np.array(data, dtype="float") / 255.0
    data = np.array(data, dtype="int")
    labels = np.array(labels)
    return(data,labels)


# In[152]:

trainX,trainY = create_data_labels_pp(train_images,pre_process=False)
testX,testY = create_data_labels_pp(test_images,pre_process=False)

#trainX,trainY = create_data_labels_aug(train_images,orientation='left')
#testX,testY = create_data_labels_aug(test_images,orientation='left')


# In[153]:


print(trainX.shape)
print(testX.shape)





length = trainX.shape[1]
width = trainX.shape[2]
depth = trainX.shape[3]
print(length,width,depth)
# number of classes is 5 if default. 2 if binary
num_classes = 2


# Change labels to categorical
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)


# In[157]:


test_Y = np.argmax(testY,axis=1)


# In[86]:


aug = ImageDataGenerator(rotation_range=25,shear_range=0.2,zoom_range=0.2)

#aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
#                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#                         horizontal_flip=True, fill_mode="nearest")


# In[87]:


aug.fit(trainX)


# In[129]:


def RMSE(actual, predict):
    diff = actual - predict
    diff = sum(diff**2) / len(actual)
    return np.sqrt(diff)


# In[147]:


def save_model(model,vgg=True):
    # saving model
    json_model = model.model.to_json()
    # Get today's date. We will use this as string for filename
    now = datetime.datetime.now()
    day = str(now)[:10]
    if vgg:
        arch_name = 'model_architecture_vgg_'+day+'35K_aug'+'.json'
        model_weights = 'model_weights_vgg_'+day+'35K_aug'+'.h5'
    else:
        arch_name = 'model_architecture_s_cnn'+day+'.json'
        model_weights = 'model_weights_s_cnn'+day+'.h5'
        
    open(arch_name, 'w').write(json_model)
    # saving weights
    model.model.save_weights(model_weights, overwrite=True)

def load_model():
    # loading model
    model = model_from_json(open('model_architecture.json').read())
    model.load_weights('model_weights.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# In[179]:

def VGG_16_TL():
    model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (length, width, depth))
    for layer in model.layers[:9]:
        layer.trainable = False
    
    #Adding custom Layers 
    x = model.output
    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    adam_opt = Adam(lr=0.01)
    rms_opt = RMSprop(lr=0.01)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    # creating the final model 
    model_final = Model(input = model.input, output = predictions)
    model_final.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

    return model_final

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),input_shape=(length,width,depth),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    adam_opt = Adam(lr=0.01)
    rms_opt = RMSprop(lr=0.01)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    #print(model.summary())
    return model

def CNN_FF():
    """ CNN with Feed Forward NN """
    model_conv = Sequential()
    model_conv.add(Conv2D(32, kernel_size=(5, 5), 
                                          input_shape=(length,width,depth),activation='relu'))
    model_conv.add(MaxPooling2D(pool_size=(2, 2)))
    model_conv.add(Conv2D(64, (5, 5), activation='relu'))
    model_conv.add(MaxPooling2D(pool_size=(2, 2)))
    model_conv.add(Flatten())
    model_conv.add(Dense(100))
    model_conv.add(Dropout(0.1))
    model_conv.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model_conv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    #print(model_conv.summary())
    return model_conv



# With Augmentation done during pre-process
#estimator_vgg_aug_pp = KerasClassifier(build_fn=VGG_16, epochs=10, batch_size=100)
#estimator_vgg_aug_pp.fit(trainX, trainY)

estimator_vgg = KerasClassifier(build_fn=VGG_16)
epochs = 10
batch_size = 100
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in aug.flow(trainX, trainY, batch_size=batch_size):
        estimator_vgg.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(trainX) / 10:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break


# In[183]:

# Save the model
save_model(estimator_vgg)

# Predict for test data
prediction_vgg_aug_pp_proba=estimator_vgg.predict_proba(testX)
prediction_vgg_aug_pp=estimator_vgg.predict(testX)


# In[184]:


print(prediction_vgg_aug_pp_proba)
print(prediction_vgg_aug_pp)


# In[185]:

acc_score = np.mean(test_Y==prediction_vgg_aug_pp)

rmse_val_vgg = RMSE(test_Y,prediction_vgg_aug_pp)
# Find the overall RMSE value.
print("Accuracy: ",acc_score)
print("RMSE-VGG: ",rmse_val_vgg)
print("Cohen-VGG: ",cohen_kappa_score(test_Y,prediction_vgg_aug_pp,weights='quadratic'))


# In[186]:


#rounded_prediction_s_cnn = np.around(prediction_nn)
predict_df = pd.DataFrame([test_Y,prediction_vgg_aug_pp]).transpose()
predict_df.columns = ['Orig_Score','Pred_Score_VGG']



# In[188]:


#cnf = confusion_matrix(test_Y, prediction_vgg_aug_pp)
#df_cm = pd.DataFrame(cnf, index = [i for i in range(2)],
#                  columns = [i for i in range(2)])
#plt.figure(figsize = (10,7))
#sn.heatmap(df_cm, annot=True,cmap="YlGnBu",fmt='g')


# In[190]:






