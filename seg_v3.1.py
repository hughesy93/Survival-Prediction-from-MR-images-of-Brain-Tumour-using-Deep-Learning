"""
Sam Hughes
20/10/2019

This code initialises a skeleton encoder-decoder convulutional neural
network for image segmentation, in application to the MICCAI Brain Tumor
Segmentation Challenge.

REFERENCES
"MICCAI BRATS - The Multimodal Brain Tumour Segmentation Challenge",
Braintumoursegmentation.org, 2019. [Online]. Available:
http://braintumoursegmentation.org. [Accessed: 17- Oct- 2019].

M. Brett, NiBabel. MIT, 2019.

F. Chollet, Keras. 2015. Software available at https://keras.io

"Home - Keras Documentation", Keras.io, 2019. [Online]. Available:
https://keras.io. [ccessed: 25- Oct- 2019].

M. Abadi et al. “TensorFlow: Large-scale machine learning on
heterogeneous systems”,
2015. Software available from tensorflow.org.

"""
import os
import numpy as np
import csv
import nibabel as nib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import MaxPooling2D, Conv2D, Dropout,\
Activation, Input, UpSampling2D, concatenate
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from math import ceil   

#____________________________ File Paths ______________________________#
#set working directory
WDir = 'work_directory'
#os.chdir(WDir)

#directory of data
dataDir1 = 'root of image folders'

#data names
datalist = 'name_mapping.csv'

channel = ['flair', 't1', 't1ce', 't2', 'seg']
#______________________________________________________________________#

#_______________________________Parameters_____________________________#

batch_size = 5

epochnum = 5

#______________________________________________________________________#


#_______________________________train test split_______________________#
##train_list = pd.DataFrame()
##test_list = pd.DataFrame()
##
##for j, i in enumerate(name_list['BraTS_2019_subject_ID']):
##    if os.path.isdir(os.path.join(dataDir1, 'train', name_list['Grade'][j], i)):
##        train_list = train_list.append(name_list.iloc[j])
##    elif os.path.isdir(os.path.join(dataDir1, 'validate', name_list['Grade'][j], i)):
##        test_list = test_list.append(name_list.iloc[j])
##    else:
##        print('Directory ', i,' not found. \n')
#______________________________________________________________________#

# data_gen is a generator that yeilds a batch of input data. Each
# iteration yeilds a stack of slices for all 4 channels and the corresponding
# segmentation data for training. The data has been zero - padded out to
# (256, 256, batch_size, 4). 

def data_gen(batch_size, directory, name_list):
    while True:
        for i in range(0, len(name_list)):
            img = np.zeros((155, 256, 256, 4))
            seg = np.zeros((155,256,256,1))
            for k in range(0,5):
                file_path = os.path.join(directory, name_list['Grade'].iloc[i],
                                         name_list['BraTS_2019_subject_ID'].iloc[i],
                                         name_list['BraTS_2019_subject_ID'].iloc[i]\
                                         + '_' + channel[k] + '.nii.gz')
                if k < 4:
                    img[:, 8:248, 8:248, k] = np.swapaxes(nib.load(file_path).get_fdata(), 0, 2)
                else:
                    seg[:, 8:248, 8:248, 0] = np.swapaxes(nib.load(file_path).get_fdata(), 0, 2)
            img[img == 4] = 3
            seg[seg == 4] = 3            
    
            for j in range(0, int(155/batch_size)):
                batch_train = img[j * batch_size : (j+1) * batch_size, :, :, :]
                batch_test = seg[j * batch_size : (j+1) * batch_size, :, :, :]
                yield (batch_train, batch_test)


# Define layers of the U-Net.

n_filters = 16

dropout = 0.1

inputs = Input(batch_shape  = (batch_size, 128, 128, 4))

# 2 x convolution layers
c1 = Conv2D(filters = n_filters, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(inputs)
c2 = Conv2D(filters = n_filters, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(c1)

# Maxpooling
p1 = MaxPooling2D(pool_size = (2,2))(c2)

# Dropout layer 
d1 = Dropout(dropout)(p1)

# 2 x convolution layers
c3 = Conv2D(filters = n_filters*2, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(d1)
c4 = Conv2D(filters = n_filters*2, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(c3)

# Maxpooling
p2 = MaxPooling2D(pool_size = (2,2))(c4)

# Dropout layer 
d2 = Dropout(dropout)(p2)

# 2 x convolution layers
c5 = Conv2D(filters = n_filters*4, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(d2)
c6 = Conv2D(filters = n_filters*4, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(c5)

# Maxpooling
p3 = MaxPooling2D(pool_size = (2,2))(c6)

# Dropout layer 
d3 = Dropout(dropout)(p3)

# 2 x convolution layers
c7 = Conv2D(filters = n_filters*8, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(d3)
c8 = Conv2D(filters = n_filters*8, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(c7)

# Maxpooling
p4 = MaxPooling2D(pool_size = (2,2))(c8)

# Dropout layer 
d4 = Dropout(dropout)(p4)

# 2 x convolution layers
c9 = Conv2D(filters = n_filters*16, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(d4)
c10 = Conv2D(filters = n_filters*16, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(c9)

# Upsampling
u1 = UpSampling2D(size = (2,2))(c10)

# Concatenate
j1 = concatenate([u1, c8])

# Dropout layer 
d5 = Dropout(dropout)(j1)

# 2 x convolution layers
c11 = Conv2D(filters = n_filters*8, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(d5)
c12 = Conv2D(filters = n_filters*8, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(c11)

# Upsampling
u2 = UpSampling2D(size = (2,2))(c12)

# Concatenate
j2 = concatenate([u2, c6])

# Dropout layer 
d6 = Dropout(dropout)(j2)

# 2 x convolution layers
c13 = Conv2D(filters = n_filters*4, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(d6)
c14 = Conv2D(filters = n_filters*4, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(c13)

# Upsampling
u3 = UpSampling2D(size = (2,2))(c14)

# Concatenate
j3 = concatenate([u3, c4])

# Dropout layer 
d7 = Dropout(dropout)(j3)

# 2 x convolution layers
c15 = Conv2D(filters = n_filters*2, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(d7)
c16 = Conv2D(filters = n_filters*2, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(c15)

# Upsampling
u4 = UpSampling2D(size = (2,2))(c16)

# Concatenate
j4 = concatenate([u4, c2])

# Dropout layer 
d8 = Dropout(dropout)(j4)

# 2 x convolution layers
c17 = Conv2D(filters = n_filters, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(d8)
c18 = Conv2D(filters = n_filters, kernel_size = (3,3), padding = 'same',
            activation = 'relu')(c17)


outputs = Conv2D(filters = 4, kernel_size = (1,1), activation = 'sigmoid')(c18)

model = Model(inputs, outputs = [outputs])



model.compile(optimizer = Adam(), loss = 'sparse_categorical_crossentropy' ) 

model.summary()

Train model

train_dir = os.path.join(dataDir1, 'train')
test_dir = os.path.join(dataDir1, 'validate')

model.fit_generator(generator = data_gen(batch_size, train_dir, train_list),
                    steps_per_epoch = ceil(len(train_list)/batch_size),
                    epochs = epochnum, verbose = 2,
                    validation_data = data_gen(batch_size, test_dir, test_list),
                    validation_steps = ceil(len(test_list)/batch_size))


model_dir = os.path.join(WDir, 'seg_model.h5')
model.save(model_dir)


