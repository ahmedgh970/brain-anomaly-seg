# Imports

import json
import os

import cv2
from skimage import filters
from einops import rearrange
import statistics
import seaborn as sns; sns.set_theme()

import random
import traceback
import nibabel as nib
import scipy 

import numpy as np
from numpy import save
import matplotlib.pyplot as plt
import time
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers
from plot_keras_history import plot_history

from sklearn.model_selection import ParameterGrid
from sklearn import metrics

from scripts.evalresults import *
from scripts.utils import *


# Configure the hyperparameters

model_name = 'Variational Autoencoder'
numEpochs = 50
learning_rate = 0.00001
rate = 0.  
image_size = 256
num_channels = 1
batch_size = 1
latent_dim = 512
intermediate_dim = 2048


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        

def VAE():

  inputs = tf.keras.Input(shape=(image_size, image_size, num_channels))
  
  x = layers.Conv2D(32 , 5, activation=layers.LeakyReLU(), strides=2, padding="same")(inputs)
  x = layers.Conv2D(64 , 5, activation=layers.LeakyReLU(), strides=2, padding="same")(x)
  x = layers.Conv2D(128, 5, activation=layers.LeakyReLU(), strides=2, padding="same")(x)    
  x = layers.Conv2D(128, 5, activation=layers.LeakyReLU(), strides=2, padding="same")(x)   
  x = layers.Conv2D(16 , 1, activation=layers.LeakyReLU(), strides=1, padding="same")(x)   
  x = layers.Flatten()(x)
  encoded = layers.Dense(intermediate_dim, activation=layers.LeakyReLU())(x)   
  
  z_mean = layers.Dense(latent_dim, name="z_mean")(encoded)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoded)
  z = Sampling()([z_mean, z_log_var])
  
  x = layers.Dense(16 * 16 * 16, activation=layers.LeakyReLU())(z)
  x = layers.Reshape((16, 16, 16))(x)
  x = layers.Conv2D(128, 1, strides=1, activation=layers.LeakyReLU(), padding="same")(x)    
  x = layers.Conv2DTranspose(128, 5, strides=2, activation=layers.LeakyReLU(), padding="same")(x) 
  x = layers.Conv2DTranspose(64 , 5, strides=2, activation=layers.LeakyReLU(), padding="same")(x)
  x = layers.Conv2DTranspose(32 , 5, strides=2, activation=layers.LeakyReLU(), padding="same")(x)
  x = layers.Conv2DTranspose(32 , 5, strides=2, activation=layers.LeakyReLU(), padding="same")(x)
  decoder_outputs = layers.Conv2D(num_channels, 1, activation=layers.LeakyReLU(), padding='same')(x)
  
  model = tf.keras.Model(inputs, decoder_outputs)
  
  return model



# Configure training and testing on MOOD Datasets 

saved_dir = './saved/'
data_dir  = './data/OASIS/'

test_healthy_path = './data/OASIS_all/OASIS_Flair_Test.npy'

test_path = './data/BraTS/s0/BraTS_Flair.npy'
brainmask_path = './data/BraTS/s0/BraTS_Brainmask.npy'
x_prior_path = './data/BraTS/s0/BraTS_prior_52.npy.npy' 
label_path = './data/BraTS/s0/BraTS_GT.npy'

'''
#-- If using MSLUB as test-set

test_path = './data/MSLUB/MSLUB_Flair.npy'
brainmask_path = './data/MSLUB/MSLUB_Brainmask.npy'
x_prior_path = './data/MSLUB/MSLUB_prior_57.npy'    
label_path = './data/MSLUB/MSLUB_GT.npy'
'''

list_len = [4805, 3078]    #-- BraTS and MSLUB test-set sizes, respectively.
len_testset = list_len[0]  #-- 0 for BraTS and 1 for MSLUB

train_paths = list_of_paths(data_dir)

nb_train_files = 66
data_gen = data_generator(train_paths[:nb_train_files], batch_size)
training_steps = (256 / batch_size) * nb_train_files

nb_val_files = 5
val_gen = data_generator(train_paths[-nb_val_files:], batch_size)
validation_steps = (256 / batch_size) * nb_val_files

# Checkpoints dir
date = datetime. now(). strftime("%Y_%m_%d-%I:%M:%S_%p")
ckpts_dir = os.path.join(saved_dir, f'Ckpts_{date}')
os.makedirs(ckpts_dir)
     
ckpts_path = os.path.join(ckpts_dir, 'Model_Ckpts.h5')
params_path = os.path.join(ckpts_dir, 'Parameters.txt')
results_path = os.path.join(ckpts_dir, 'Results.txt')
fig_path = os.path.join(ckpts_dir, 'History_plot.png')
dice_plot_path = os.path.join(ckpts_dir, 'Dice_plot.png')
predicted_path = os.path.join(ckpts_dir, 'Predicted.npy')
residual_path = os.path.join(ckpts_dir, 'Residuals.npy')
residual_BP_path = os.path.join(ckpts_dir, 'Residuals_BP.npy')

      
# Configure the training
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
           
calbks = tf.keras.callbacks.ModelCheckpoint(filepath=ckpts_path, monitor='loss', save_best_only=True, save_weights_only=True, verbose=2)
tqdm_callback = tfa.callbacks.TQDMProgressBar() 

model = VAE()
model.summary()
model.compile(optimizer=opt, loss='mae', metrics=['mse', SSIMLoss, MS_SSIMLoss])


# Print & Write model Parameters
parameters = (f'\nSelected model "{model_name}" with :\n - {batch_size}: Batche(s)\n - {numEpochs}: Epochs\n - {intermediate_dim}: Intermediate dim\n - {latent_dim}: Bottelneck size\n')
print(parameters)

  
# TRAIN 
print('\nTrain =>\n')
history = model.fit(x = data_gen,
                    steps_per_epoch = training_steps,
                    validation_data = val_gen,
                    validation_steps = validation_steps,
                    verbose = 0,
                    epochs = numEpochs,
                    callbacks = [calbks, tqdm_callback]
                    )

                          
# Get training and test loss histories                   
plot_history(history, path=fig_path)
plt.close()
time.sleep(2)


# Test       
print('\nTest ===>\n')
my_test = np.load(test_path)
brainmask = np.load(brainmask_path)
x_prior = np.load(x_prior_path)
my_labels = np.load(label_path)

healthy_test = np.load(test_healthy_path)
steps = healthy_test.shape[0]

score = model.evaluate(x=healthy_test, y=healthy_test, verbose=0, steps=steps, callbacks = [tqdm_callback])    
score_out = (f'\nTest on healthy unseen data :\n - Test loss (MAE): {score[0]},\n - Test MSE : {score[1]},\n - Test SSIM : {score[2]}\n - Test MS_SSIM : {score[3]}\n')
print(score_out)


#-- Predict
print('\nPredict =====>\n')
predicted = model.predict(x=my_test, verbose=1, steps=len_testset)
np.save(predicted_path, predicted)
time.sleep(4)


#-- Calculate, Post-process and Save Residuals
print('\nCalculate, Post-process and Save Residuals =====>\n')     
residual_BP = calculate_residual_BP(my_test, predicted, brainmask)  #-- You can use either brainmask or x_prior
np.save(residual_BP_path, residual_BP)
        
residual = calculate_residual(my_test, predicted, brainmask)  #-- You can use either brainmask or x_prior
np.save(residual_path, residual)
        

#-- Evaluation
print('\nEvaluate =========>\n')        
[AUROC, AUPRC, AVG_DICE, MAD, STD], DICE = eval_residuals(my_labels, residual)     
results = (f'\nResults after median_filter :\n - AUROC = {AUROC}\n - AUPRC = {AUPRC}\n - AVG_DICE = {AVG_DICE}\n - MEDIAN_ABSOLUTE_DEVIATION = {MAD}\n - STANDARD_DEVIATION = {STD}')
print(results)
                      
plt.figure()
hor_axis = [x for x in range(len_testset)]
plt.scatter(hor_axis, DICE, s = 5, marker = '.', c = 'blue')
plt.ylabel('Dice Score')
plt.xlabel('N° Samples')
plt.title('Dice scores')
plt.savefig(dice_plot_path)
time.sleep(2)


#-- Save
print('\nSave Results and Parameters =============>\n')
f = open(results_path, "w")
f.write(results)       
f.close()   
                       
f = open(params_path, "w")
f.write(parameters)
f.write(score_out)
f.close()

      
#-- End
print('\nEnd !\n')