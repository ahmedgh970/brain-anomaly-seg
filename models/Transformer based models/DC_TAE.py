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



"""# Implement patch creation as a layer"""

class Patches(layers.Layer):

    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size
        })
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID")
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


"""# Implement the patch encoding layer"""

class PatchEncoder(layers.Layer):

    def __init__(self, embed_shape):
        super(PatchEncoder, self).__init__()
        self.embed_shape = embed_shape
        self.position_embedding = layers.Embedding(
            input_dim = embed_shape[0],
            output_dim = embed_shape[1]
            )
        self.projection = layers.Dense(units=embed_shape[1])
        
    def get_config(self):
      config = super().get_config().copy()
      config.update({
          'embed_shape': self.embed_shape,
          'position_embedding': self.position_embedding,
          'projection': self.projection
        })
      return config

    def call(self, patches):
        positions = tf.range(start=0, limit=self.embed_shape[0], delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded


"""# Implement image reconstruction from patches as a layer"""

class Images(layers.Layer):
  
    def __init__(self, image_size, patch_size, num_channels):
        super(Images, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_channels': self.num_channels
          })
        return config

    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        reconstructed = tf.reshape(patches, [batch_size, self.image_size, self.image_size, num_channels])
        rec_new = tf.nn.space_to_depth(reconstructed, self.patch_size) 
        image = tf.reshape(rec_new, [batch_size, self.image_size, self.image_size, self.num_channels])   
        return image


"""# Dense Layer """    

class TruncatedDense(layers.Dense):
    def __init__(self, units, use_bias=True, initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)):
        super().__init__(units, use_bias=use_bias, kernel_initializer=initializer)


"""# Mlp Head """

class Mlp(layers.Layer):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=layers.Activation(tf.nn.gelu), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TruncatedDense(hidden_features)
        self.act = act_layer
        self.fc2 = TruncatedDense(out_features)
        self.drop = layers.Dropout(drop)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fc1': self.fc1,
            'act': self.act,
            'fc2': self.fc2,
            'drop': self.drop
          })
        return config

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


"""# Dense Convolutional Autoencoder"""

def DCAE(input_image):

    x = layers.Conv2D(32 , 3, activation=layers.LeakyReLU(), strides=2, padding="same")(input_image)
    x = layers.Conv2D(64 , 3, activation=layers.LeakyReLU(), strides=2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(), strides=2, padding="same")(x)    
    x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(), strides=2, padding="same")(x)   
    x = layers.Conv2D(16 , 1, activation=layers.LeakyReLU(), strides=1, padding="same")(x)
    
    x = layers.Flatten()(x)
    encoded = layers.Dense(intermediate_dim, activation=layers.LeakyReLU())(x)

    #-- BOTTELNECK SIZE : 512 = 4096/8   ==> choice (128 = 1024/8)
    
    x = layers.Dense(16 * 16 * 16, activation=layers.LeakyReLU())(encoded)
    x = layers.Reshape((16, 16, 16))(x)

    x = layers.Conv2D(128, 1, strides=1, activation=layers.LeakyReLU(), padding="same")(x)    
    x = layers.Conv2DTranspose(128, 3, strides=2, activation=layers.LeakyReLU(), padding="same")(x) 
    x = layers.Conv2DTranspose(64 , 3, strides=2, activation=layers.LeakyReLU(), padding="same")(x)
    x = layers.Conv2DTranspose(32 , 3, strides=2, activation=layers.LeakyReLU(), padding="same")(x)
    x = layers.Conv2DTranspose(32 , 3, strides=2, activation=layers.LeakyReLU(), padding="same")(x)
    
    decoded = layers.Conv2D(num_channels, 1, activation=layers.LeakyReLU(), padding='same')(x)

    return decoded


# Model implementation : Vision Autoencoder Transformer

def transformer_autoencoder(encoded_patches):

    for _ in range(transformer_layers):
        query = key = encoded_patches
        attn_encoded_patches = layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embed_shape[1],
            dropout = rate)(query=query, value=encoded_patches, key=key)   
        attn_encoded_patches = layers.Dropout(rate)(attn_encoded_patches) 
        encoded_patches += attn_encoded_patches  
        encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        ffn_out = Mlp(
            in_features=embed_shape[1],
            hidden_features=4*embed_shape[1],
            drop=rate)(encoded_patches)
        encoded_patches += ffn_out   
        encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    ## Convolutional AutoEncoder
    encoded_img = Images(image_size, patch_size, num_channels)(encoded_patches)
    decoded_patches = DCAE(encoded_img)  
    encoded_patches = Patches(patch_size)(decoded_patches)
    encoded_patches = PatchEncoder(embed_shape)(encoded_patches)

    positions = tf.range(start=0, limit = embed_shape[0], delta=1)
    pos_embed = layers.Embedding(
        input_dim = embed_shape[0],
        output_dim = embed_shape[1])(positions)
    target = encoded_patches

    for _ in range(transformer_layers):
        query_tgt = key_tgt = target + pos_embed    
        attn_target1 = layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embed_shape[1],
            dropout=rate)(query=query_tgt, value=target, key=key_tgt)   
        attn_target1 = layers.Dropout(rate)(attn_target1)
        target += attn_target1
        target = layers.LayerNormalization(epsilon=1e-6)(target)
        query_tgt = target + pos_embed
        attn_target2 = layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embed_shape[1],
            dropout=rate)(query=query_tgt, value=encoded_patches + pos_embed, key=encoded_patches)
        attn_target2 = layers.Dropout(rate)(attn_target2)
        target += attn_target2
        target = layers.LayerNormalization(epsilon=1e-6)(target)
        ffn_out = Mlp(
            in_features=embed_shape[1],
            hidden_features=4*embed_shape[1],
            drop=rate)(encoded_patches)
        target +=ffn_out
        target = layers.LayerNormalization(epsilon=1e-6)(target)
    return target

# Build the TAE Model

def TAE():
  
    input_img = tf.keras.Input(shape=(image_size, image_size, num_channels))

    patches = Patches(patch_size)(input_img)    
    enc_patches = PatchEncoder(embed_shape)(patches)

    decoded_patches = transformer_autoencoder(enc_patches)
  
    reconstructed = Images(image_size, patch_size, num_channels)(decoded_patches)

    model = tf.keras.Model(input_img, reconstructed)
    
    return model
 
    
# Configure the hyperparameters

model_name = 'DCAE Inside Transformer'
numEpochs = 50
learning_rate = 0.00001
rate = 0.1  
image_size = 256
num_channels = 1
batch_size = 1
intermediate_dim = 512


param_grid = {
              'transformer_layers': [8],      
              'patch_size' : [16],
              'num_heads' : [4]
             }
             
            
PARAMS = ParameterGrid(param_grid)
list_PARAMS = list(PARAMS)


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


# Train, Evaluate and Test

for param in list_PARAMS:
   try:

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
      

        # Configure the parameters
        transformer_layers = param['transformer_layers']
        patch_size = param['patch_size']
        num_heads = param['num_heads']
        input_resolution = image_size // patch_size
        num_patches = input_resolution ** 2
        embed_shape = (num_patches, patch_size*patch_size*num_channels)

        # Configure the training
        opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            
        calbks = tf.keras.callbacks.ModelCheckpoint(filepath=ckpts_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=2)
        tqdm_callback = tfa.callbacks.TQDMProgressBar() 

        model = TAE()
        model.summary()        
        model.compile(optimizer=opt, loss='mae', metrics=['mse', SSIMLoss, MS_SSIMLoss])       
        
        # Print & Write model Parameters
        parameters = (f'\nSelected model "{model_name}" with :\n - {num_heads}: Heads,\n - {patch_size}: Patch size,\n - {transformer_layers}: Transformer Layer,\n - ({embed_shape[0]}, {embed_shape[1]}): Embedding Shape,\n - {batch_size}: Batche(s)\n - {numEpochs}: Epochs\n')
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
        score_out = (f'\nTest on healthy unseen data :\n - Test loss (MSE): {score[0]},\n - Test MAE : {score[1]},\n - Test SSIM: {score[2]},\n - Test MS_SSIMLoss: {score[3]},\n')
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
        print('\nEnd of step !\n')
   

   except:
        print('Error encountered !')
        tf.keras.backend.clear_session()
        continue
        
        
