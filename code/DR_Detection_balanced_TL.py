#!/usr/bin/env python
# coding: utf-8

# # DrEyeBot Retinal Image Analysis using CNN

# In[6]:


# This tells matplotlib not to try opening a new window for each plot.

# Import a bunch of libraries.
import datetime
import time
import random
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import os
import sys
import imutils
#from matplotlib.ticker import MultipleLocator
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.utils.multiclass import unique_labels
from shutil import copyfile

import pandas as pd
import cv2
#import seaborn as sn

# Set the randomizer seed so results are the same each time.
np.random.seed(0)


# In[7]:


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
from keras.initializers import glorot_normal
from keras import losses

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,RMSprop,SGD
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#from pyimagesearch.smallervggnet import SmallerVGGNet


# In[3]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# ## Common Functions

# ### Helper Functions

# In[8]:


def levelset(x):
    """ Sets Levels 1-4 to level 1"""
    if x>0:
        return 1
    else:
        return 0

def RMSE(actual, predict):
    diff = actual - predict
    diff = sum(diff**2) / len(actual)
    return np.sqrt(diff)

def select_toy_images(image_label,N=-1,images_percent=list()):
    """ Selects number of images from each class. By default it is ALL images"""
    image_list = list()
    final_images = pd.DataFrame(columns = image_label.columns)
    # We create a toy dataset of 'N' images, maintaining the split of the original 
    if N==-1:
            # We need to pick all the images. No need to sample
            # We can ignore the percentage here
            final_images = image_label
    else:
        for level in range(5):
        # Get respective number of images in each level
            if len(images_percent)==5:
                number_of_images = int(images_percent[level]*N/100)
            else:
                # We have no percentage of images. Setting the default safe percentage
                images_percent = [73.6,6.9,15.1,2.4,2]
                number_of_images = int(images_percent[level]*N/100)
            sample_images = image_label[image_label.level==level].sample(n=number_of_images,axis=0)
            frames = [final_images,sample_images]
            final_images = pd.concat(frames).reset_index(drop=True)
    return final_images 

def preprocess(image,scale=224):
    """ preprocess the test image and covert to array """
    inter=cv2.INTER_AREA
    (h, w) = image.shape[:2]
    dW = 0
    dH = 0

    width = scale
    height = scale

    # if the width is smaller than the height, then resize
    # along the width (i.e., the smaller dimension) and then
    # update the deltas to crop the height to the desired
    # dimension
    if w < h:
        image = imutils.resize(image, width=width,
            inter=inter)
        dH = int((image.shape[0] - height) / 2.0)

    # otherwise, the height is smaller than the width so
    # resize along the height and then update the deltas
    # crop along the width
    else:
        image = imutils.resize(image, height=height,
            inter=inter)
        dW = int((image.shape[1] - width) / 2.0)

    # now that our images have been resized, we need to
    # re-grab the width and height, followed by performing
    # the crop
    (h, w) = image.shape[:2]
    image = image[dH:h - dH, dW:w - dW]

    # finally, resize the image to the provided spatial
    # dimensions to ensure our output image is always a fixed
    # size

    image =  cv2.resize(image, (width, height),
        interpolation=inter)
    image_arr = img_to_array(image)
    image_arr = image_arr/255.0
    return img_to_array(image_arr)

def selector(x):
    """ Function to select the class"""
    if x[0] > x[1]:
        return 0
    else:
        return 1


# ### Model related functions



# In[10]:


def OrdinalLoss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)

def OrdinalLoss_new(y_true,y_pred,train_out):
    """ Custom loss Function for Ordinal Data"""
    num_classes=5
    dx = np.ones((num_classes,1)) * np.arange(num_classes)
    dy = dx.transpose()
    d = np.sqrt(abs(dx - dy))
    overestimate_penalty = np.triu(d[:,1:]) / (np.spacing(1) + 
                                               (np.sum(np.triu(d[:,1:]), axis=1)/
                                                (np.arange(num_classes)[::-1]+np.spacing(1))).reshape((5,1)))
    underestimate_penalty = overestimate_penalty[::-1, ::-1]
    overestimate_penalty = tf.convert_to_tensor(overestimate_penalty, np.float32)
    underestimate_penalty = tf.convert_to_tensor(underestimate_penalty, np.float32)
    #loss_train = -K.mean(K.sum((underestimate_penalty[y_pred])*K.log(train_out) + (overestimate_penalty[y_pred])*K.log(1-train_out), axis=1))
    loss_train = K.binary_crossentropy(y_true, y_pred)
    loss_val = K.binary_crossentropy(y_true, y_pred)
    return K.in_train_phase(loss_train, loss_val)

def LossWrapper(train_out):
    def customLoss(y_true, y_pred):
        return OrdinalDataLoss_new(y_true, y_pred, train_out)
    return customLoss


# In[11]:


def VGG_16_TL(input_shape,layers_to_skip=None,weights=None,include_top=True):
    """ VGG 16 with Transfer Learning. Using Keras built in function"""
    model = applications.VGG16(weights = weights, include_top=include_top, input_shape = input_shape)
    if layers_to_skip:
        if weights==None:
            print("ERROR: You cannot have weights as none if layers_to_skip is non-zero")
        else:
            for layer in model.layers[:layers_to_skip]:
                layer.trainable = False
            #Adding custom Layers 
            x = model.output
            x = Flatten()(x)
            x = Dense(4096, activation="relu")(x)
            x = Dropout(0.5)(x)
            x = Dense(4096, activation="relu")(x)
            x = Dropout(0.5)(x)
    else:
        x = model.output
    
    predictions = Dense(num_classes, activation="softmax")(x)
    adam_opt = Adam(lr=0.01)
    rms_opt = RMSprop(lr=0.01)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    # creating the final model 
    model_final = Model(input = model.input, output = predictions)
    model_final.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    return model_final

def VGG_16(weights_path=None):
    initializer = glorot_normal()
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),input_shape=(length,width,depth),
                     activation='relu',kernel_initializer=initializer))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu',kernel_initializer=initializer))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu',kernel_initializer=initializer))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',kernel_initializer=initializer))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    adam_opt = Adam(lr=0.01)
    rms_opt = RMSprop(lr=0.01)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    #x = model.output
    #custom_loss = LossWrapper(train_out=x)
    # compile model
    model.compile(loss=OrdinalLoss,optimizer = sgd, metrics = ['accuracy'])
    #model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
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


def save_model(model,vgg=True):
    # saving model
    json_model = model.to_json()
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
    model.save_weights(model_weights, overwrite=True)

def load_model(model_arch,model_weights):
    # loading model
    model = model_from_json(open(model_arch).read())
    model.load_weights(model_weights)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# ## LOAD DATA

# In[12]:


# Load the csv data
orig_label = pd.read_csv("./Retinal-Images/trainLabels.csv")
# Load the test csv data
test_label = pd.read_csv("./Retinal-Images/retinopathy_solution.csv")
test_label.drop('Usage',inplace=True,axis=1)



# ### DATA PREPERATION

# In[15]:


l4_orig_df = orig_label[orig_label.level==4].sample(200,random_state=42)
l3_orig_df = orig_label[orig_label.level==3].sample(500,random_state=42)
l2_orig_df = orig_label[orig_label.level==2].sample(500,random_state=42)
l1_orig_df = orig_label[orig_label.level==1].sample(500,random_state=42)
l0_orig_df = orig_label[orig_label.level==0].sample(3300,random_state=42)
rest_orig_df = pd.concat([orig_label, l4_orig_df,l3_orig_df,
                          l2_orig_df,l1_orig_df,l0_orig_df]).drop_duplicates(keep=False)


# In[16]:


l4_test_df = test_label[test_label.level==4].sample(200,random_state=42)
l3_test_df = test_label[test_label.level==3].sample(500,random_state=42)
l2_test_df = test_label[test_label.level==2].sample(500,random_state=42)
l1_test_df = test_label[test_label.level==1].sample(500,random_state=42)
l0_test_df = test_label[test_label.level==0].sample(3300,random_state=42)
rest_test_df = pd.concat([test_label, l4_test_df,l3_test_df,
                          l2_test_df,l1_test_df,l0_test_df]).drop_duplicates(keep=False)


# ### Save the dataframes so that same sets can be loaded at a later point

# In[15]:



# In[17]:


val_list = [l0_orig_df,l1_orig_df,l2_orig_df,l3_orig_df,l4_orig_df]
test_list = [l0_test_df,l1_test_df,l2_test_df,l3_test_df,l4_test_df]
train_list = [rest_orig_df,rest_test_df]


# In[18]:


test_labels=list()
test_labels.append(list(l0_test_df.level.values))
test_labels.append(list(l1_test_df.level.values))
test_labels.append(list(l2_test_df.level.values))
test_labels.append(list(l3_test_df.level.values))
test_labels.append(list(l4_test_df.level.values))
test_labels = [item for sublist in test_labels for item in sublist]
test_labels=np.asarray(test_labels)


# In[19]:


test_image_list=list()
test_image_list.append(list(l0_test_df.image.values))
test_image_list.append(list(l1_test_df.image.values))
test_image_list.append(list(l2_test_df.image.values))
test_image_list.append(list(l3_test_df.image.values))
test_image_list.append(list(l4_test_df.image.values))
test_image_list = [item for sublist in test_image_list for item in sublist]
test_image_list=np.asarray(test_image_list)


# In[20]:


val_labels=list()
val_labels.append(list(l0_orig_df.level.values))
val_labels.append(list(l1_orig_df.level.values))
val_labels.append(list(l2_orig_df.level.values))
val_labels.append(list(l3_orig_df.level.values))
val_labels.append(list(l4_orig_df.level.values))
val_labels = [item for sublist in val_labels for item in sublist]
val_labels=np.asarray(val_labels)


# In[21]:


val_image_list=list()
val_image_list.append(list(l0_orig_df.image.values))
val_image_list.append(list(l1_orig_df.image.values))
val_image_list.append(list(l2_orig_df.image.values))
val_image_list.append(list(l3_orig_df.image.values))
val_image_list.append(list(l4_orig_df.image.values))
val_image_list = [item for sublist in val_image_list for item in sublist]
val_image_list=np.asarray(val_image_list)


# In[22]:




# In[23]:


test_labels[test_labels>0]=1


# In[24]:


val_labels[val_labels>0]=1


# In[52]:


# Copy train images
# OVERSAMPLING LOGIC:
# CLASS-1 IMAGES 10X
# CLASS-2 IMAGES 5X
# CLASS-3 IMAGES 10X
# CLASS-4 IMAGES 10X

# Set this if we are using binary classification
binary_class = 1
oversample = {0:1,1:3,2:1,3:14,4:10}
#oversample = {0:1,1:10,2:5,3:30,4:25}


for i in range(2):
    if i==0:
        image_dir = os.getcwd()+'/Retinal-Images/train_resize_224_new/'
        for j in range(len(rest_orig_df)):        
            image_class = rest_orig_df.iloc[[j],[1]].values[0][0]
            image_name = rest_orig_df.iloc[[j],[0]].values[0][0]
            image_loc = image_dir+image_name+'.png'
            # Get the oversample percent for the given class
            class_oversample = oversample[image_class]
            image_class_str = image_class
            if binary_class:
                if image_class_str>0:
                    image_class_str=1
            for samples in range(class_oversample):
                class_dir = 'class_'+str(image_class_str)
                out_dir = os.getcwd()+'/Retinal-Images/train_new/'+class_dir+'/'
                copy_loc = out_dir+image_name+'-'+str(samples)+'.png'
                copyfile(image_loc,copy_loc)               
    else:
        image_dir = os.getcwd()+'/Retinal-Images/test_resize_224_new/'
        for j in range(len(rest_test_df)):        
            image_class = rest_test_df.iloc[[j],[1]].values[0][0]
            image_name = rest_test_df.iloc[[j],[0]].values[0][0]
            image_loc = image_dir+image_name+'.png'
            # Get the oversample percent for the given class
            class_oversample = oversample[image_class]
            image_class_str = image_class
            if binary_class:
                if image_class_str>0:
                    image_class_str=1
            for samples in range(class_oversample):
                class_dir = 'class_'+str(image_class_str)
                out_dir = os.getcwd()+'/Retinal-Images/train_new/'+class_dir+'/'
                copy_loc = out_dir+image_name+'-'+str(samples)+'.png'
                copyfile(image_loc,copy_loc)               

# Copy val images
for df in val_list:
    image_dir = os.getcwd()+'/Retinal-Images/train_resize_224_new/'
    for j in range(len(df)):
        image_class = df.iloc[[j],[1]].values[0][0]
        image_name = df.iloc[[j],[0]].values[0][0]
        image_loc = image_dir+image_name+'.png'
        # Get the oversample percent for the given class
        class_oversample = oversample[image_class]
        image_class_str = image_class
        if binary_class:
            if image_class_str>0:
                image_class_str=1
        for samples in range(class_oversample):
            class_dir = 'class_'+str(image_class_str)
            out_dir = os.getcwd()+'/Retinal-Images/val/'+class_dir+'/'
            copy_loc = out_dir+image_name+'-'+str(samples)+'.png'
            copyfile(image_loc,copy_loc)  

# Copy test images
for df in test_list:
    image_dir = os.getcwd()+'/Retinal-Images/test_resize_224_new/'
    for j in range(len(df)):
        image_name = df.iloc[[j],[0]].values[0][0]
        image_loc = image_dir+image_name+'.png'
        out_dir = os.getcwd()+'/Retinal-Images/test_new/test/'
        copy_loc = out_dir+image_name+'.png'
        copyfile(image_loc,copy_loc)


# Initialize parameters
length = 224
width = 224
depth = 3
num_classes = 2
input_shape = (224,224,3)

batch_size = 100


# In[26]:


val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
         'Retinal-Images/test_new',
         target_size=(224, 224),
         batch_size=batch_size)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)

train_generator = train_datagen.flow_from_directory(
        'Retinal-Images/train_new',
        target_size=(224,224),
        classes = ['class_0','class_1'],
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
         'Retinal-Images/val',
         target_size=(224, 224),
         classes = ['class_0','class_1'],
         batch_size=batch_size,
         class_mode='categorical')


# In[27]:


#n_train = len(rest_orig_df) + len(rest_test_df)
# Finding n_train with oversampling needs better coding. Hardcode for now
n_train = 116869
n_val = 14300
n_test = len(l0_test_df)+len(l1_test_df)+len(l2_test_df)+len(l3_test_df)+len(l4_test_df)
n_train,n_val,n_test


# In[28]:


steps_per_epoch = int(n_train/batch_size)
validation_steps = int(n_val/batch_size)
test_steps = int(n_test/batch_size)


# In[29]:




# In[30]:


callbacks = [EarlyStopping(monitor='val_loss', patience=6),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]


# In[31]:


model = VGG_16_TL(input_shape,layers_to_skip=7,weights="imagenet",include_top=False)
#model = VGG_16()
print(model.summary())
H=model.fit_generator(
        train_generator,
        callbacks=callbacks,
        validation_data=validation_generator,
        nb_epoch=20,steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps)


# In[31]:


save_model(model)


# ## Prediction without using generator

# In[48]:


testList = list(test_image_list)
valList = list(val_image_list)


# In[33]:


# Process each image one at a time for test data
image_class_proba_list = []
image_class_list = []
for image in testList:
    image_loc = './Retinal-Images/test_resize_224/' + image + '.jpeg'
    image_test = cv2.imread(image_loc)
    image_test_mod = preprocess(image_test)
    image_test_pp_mod = np.expand_dims(image_test_mod,0)
    image_class_proba = model.predict(image_test_pp_mod).flatten()
    image_class = model.predict(image_test_pp_mod).flatten()
    image_class_proba_list.append(image_class_proba)
    image_class_list.append(image_class)


# In[34]:


image_class_proba_arr = np.vstack(image_class_proba_list)


# In[35]:


# In[36]:


image_class_arr = np.argmax(image_class_proba_arr,axis=1)


# In[37]:


df_pred_class = pd.DataFrame(test_image_list,columns=['Image'])
df_pred_class['Actual'] = test_labels
df_pred_class['Pred'] = image_class_arr
df_class_1 = df_pred_class[(df_pred_class.Pred>0)&(df_pred_class.Actual==df_pred_class.Pred)]
df_class_0 = df_pred_class[(df_pred_class.Pred==0)&(df_pred_class.Actual==df_pred_class.Pred)]


# In[38]:


df_pred_class.groupby(['Pred']).count()


# In[39]:


df_class_0.head(5)


# In[40]:

print(df_class_0.count())

print(df_class_1.count())


# In[46]:


cnf = confusion_matrix(test_labels, image_class_arr)
tn,fp,fn,tp = cnf.ravel()
print("TN: {},FP: {}, FN: {}, TP:{}".format(tn,fp,fn,tp))


# In[47]:


# In[50]:


# Process each image one at a time for validation data
val_image_class_proba_list = []
val_image_class_list = []
for image in valList:
    image_loc = './Retinal-Images/train_resize_224/' + image + '.jpeg'
    image_test = cv2.imread(image_loc)
    image_test_mod = preprocess(image_test)
    image_test_pp_mod = np.expand_dims(image_test_mod,0)
    image_class_proba = model.predict(image_test_pp_mod).flatten()
    image_class = model.predict(image_test_pp_mod).flatten()
    val_image_class_proba_list.append(image_class_proba)
    val_image_class_list.append(image_class)


# In[51]:


val_image_class_proba_arr = np.vstack(val_image_class_proba_list)
val_image_class_arr = np.argmax(val_image_class_proba_arr,axis=1)
df_pred_val_class = pd.DataFrame(val_image_list,columns=['Image'])
df_pred_val_class['Actual'] = val_labels
df_pred_val_class['Pred'] = val_image_class_arr
df_val_class_1 = df_pred_val_class[(df_pred_val_class.Pred>0)&(df_pred_val_class.Actual==df_pred_val_class.Pred)]
df_val_class_0 = df_pred_val_class[(df_pred_val_class.Pred==0)&(df_pred_val_class.Actual==df_pred_val_class.Pred)]


# In[57]:


cnf_val = confusion_matrix(val_labels, val_image_class_arr)
tn,fp,fn,tp = cnf_val.ravel()
print("TN: {},FP: {}, FN: {}, TP:{}".format(tn,fp,fn,tp))



# ## Load the best model and repeat

# In[61]:


model_vgg = load_model('./model_architecture_vgg_2019-03-2535K_aug.json',
                           './best_model.h5')


# In[62]:


# Process each image one at a time for test data
image_class_proba_list = []
image_class_list = []
for image in testList:
    image_loc = './Retinal-Images/test_resize_224/' + image + '.jpeg'
    image_test = cv2.imread(image_loc)
    image_test_mod = preprocess(image_test)
    image_test_pp_mod = np.expand_dims(image_test_mod,0)
    image_class_proba = model_vgg.predict(image_test_pp_mod).flatten()
    image_class = model_vgg.predict(image_test_pp_mod).flatten()
    image_class_proba_list.append(image_class_proba)
    image_class_list.append(image_class)


# In[63]:


image_class_proba_arr = np.vstack(image_class_proba_list)
image_class_arr = np.argmax(image_class_proba_arr,axis=1)
df_pred_class = pd.DataFrame(test_image_list,columns=['Image'])
df_pred_class['Actual'] = test_labels
df_pred_class['Pred'] = image_class_arr
df_class_1 = df_pred_class[(df_pred_class.Pred>0)&(df_pred_class.Actual==df_pred_class.Pred)]
df_class_1 = df_pred_class[(df_pred_class.Pred==0)&(df_pred_class.Actual==df_pred_class.Pred)]


# In[64]:


df_pred_class.groupby(['Pred']).count()


# In[65]:


cnf = confusion_matrix(test_labels, image_class_arr)
tn,fp,fn,tp = cnf.ravel()
print("TN: {},FP: {}, FN: {}, TP:{}".format(tn,fp,fn,tp))


# In[67]:


# Process each image one at a time for validation data
val_image_class_proba_list = []
val_image_class_list = []
for image in valList:
    image_loc = './Retinal-Images/train_resize_224/' + image + '.jpeg'
    image_test = cv2.imread(image_loc)
    image_test_mod = preprocess(image_test)
    image_test_pp_mod = np.expand_dims(image_test_mod,0)
    image_class_proba = model_vgg.predict(image_test_pp_mod).flatten()
    image_class = model_vgg.predict(image_test_pp_mod).flatten()
    val_image_class_proba_list.append(image_class_proba)
    val_image_class_list.append(image_class)


# In[68]:


val_image_class_proba_arr = np.vstack(val_image_class_proba_list)
val_image_class_arr = np.argmax(val_image_class_proba_arr,axis=1)
df_pred_val_class = pd.DataFrame(val_image_list,columns=['Image'])
df_pred_val_class['Actual'] = val_labels
df_pred_val_class['Pred'] = val_image_class_arr
df_val_class_1 = df_pred_val_class[(df_pred_val_class.Pred>0)&(df_pred_val_class.Actual==df_pred_val_class.Pred)]
df_val_class_0 = df_pred_val_class[(df_pred_val_class.Pred==0)&(df_pred_val_class.Actual==df_pred_val_class.Pred)]


# In[69]:


cnf_val = confusion_matrix(val_labels, val_image_class_arr)
tn,fp,fn,tp = cnf_val.ravel()
print("TN: {},FP: {}, FN: {}, TP:{}".format(tn,fp,fn,tp))





