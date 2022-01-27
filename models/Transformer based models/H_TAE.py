#-- Imports

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


#--- Dense Layer   

class TruncatedDense(layers.Dense):
    def __init__(self, units, use_bias=True, initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)):
        super(TruncatedDense, self).__init__(units, use_bias=use_bias, kernel_initializer=initializer)


#--- Patch Merging Layer

class PatchMerging(layers.Layer):

    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim    #--- refer to the projection_dim (nC)
        self.reduction = TruncatedDense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_resolution': self.input_resolution,
            'dim': self.dim,
            'reduction': self.reduction,
            'norm': self.norm
          })
        return config    

    def call(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.input_resolution
        B = tf.shape(x)[0]

        x = tf.reshape(x, [B, H, W, self.dim])
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = tf.reshape(x, [B, (H//2)*(W//2), 4 * self.dim])  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        return x

#--- Patch Expanding Layer  

class PatchExpanding(layers.Layer):

    def __init__(self, input_resolution, dim):
        super(PatchExpanding, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = TruncatedDense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_resolution': self.input_resolution,
            'dim': self.dim,
            'expand': self.expand,
            'norm': self.norm
          })
        return config

    def call(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.input_resolution
        x = self.expand(x)
        B = tf.shape(x)[0]

        x = tf.reshape(x, [B, H, W, self.dim * 2])
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=self.dim // 2)

        x = tf.reshape(x, [B, (H*2)*(W*2), self.dim//2])
        x = self.norm(x)

        return x


#--- Final Patch Expanding Layer

class FinalPatchExpand_X4(layers.Layer):

    def __init__(self, input_resolution, dim):
        super(FinalPatchExpand_X4, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = TruncatedDense(16 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_resolution': self.input_resolution,
            'dim': self.dim,
            'expand': self.expand,
            'norm': self.norm
          })
        return config

    def call(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.input_resolution
        x = self.expand(x)
        B = tf.shape(x)[0]

        x = tf.reshape(x, [B, H, W, self.dim * 16])
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=4, p2=4, c=self.dim)
        x = tf.reshape(x, [B, (H*4)*(W*4), self.dim])

        x = self.norm(x)

        return x


#--- Mlp Head

class Mlp(layers.Layer):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=layers.Activation(tf.nn.gelu), drop=0.):
        super(Mlp, self).__init__()
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

#--- Transformer Encoder Layer

class TransformerEnc(layers.Layer):
  
  def __init__(self, transformer_layers, num_heads, embed_shape, rate):
    super(TransformerEnc, self).__init__()
    num_heads = num_heads
    embed_shape = embed_shape
    rate = rate
    self.transformer_layers = transformer_layers
    self.mlp = Mlp(
        in_features=embed_shape[1],
        hidden_features=4*embed_shape[1],
        drop=rate)
    self.norm = layers.LayerNormalization(epsilon=1e-6)
    self.drop = layers.Dropout(rate)
    self.mha = layers.MultiHeadAttention(
        num_heads = num_heads,
        key_dim = embed_shape[1],
        dropout = rate)
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'transformer_layers': self.transformer_layers,
        'mlp': self.mlp,
        'norm': self.norm,
        'drop': self.drop,
        'mha': self.mha
        })
    return config
  
  def call(self, encoded_patches):
    for _ in range(self.transformer_layers):
      query = key = encoded_patches
      attn_encoded_patches = self.mha(query=query, value=encoded_patches, key=key)
      attn_encoded_patches = self.drop(attn_encoded_patches)
      encoded_patches += attn_encoded_patches
      encoded_patches = self.norm(encoded_patches)
      ffn_out = self.mlp(encoded_patches)
      encoded_patches += ffn_out
      encoded_patches = self.norm(encoded_patches)
    return encoded_patches


#--- Transformer Decoder Layer

class TransformerDec(layers.Layer):
  
  def __init__(self, transformer_layers, num_heads, embed_shape, rate):
    super(TransformerDec, self).__init__()
    num_heads = num_heads
    embed_shape = embed_shape
    rate = rate
    self.transformer_layers = transformer_layers
    self.mlp = Mlp(
        in_features=embed_shape[1],
        hidden_features=4*embed_shape[1],
        drop=rate)
    self.norm = layers.LayerNormalization(epsilon=1e-6)
    self.drop = layers.Dropout(rate)
    self.mha = layers.MultiHeadAttention(
        num_heads = num_heads,
        key_dim = embed_shape[1],
        dropout=rate)
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'transformer_layers': self.transformer_layers,
        'positions': self.positions,
        'mlp': self.mlp,
        'norm': self.norm,
        'drop': self.drop,
        'mha': self.mha
        })
    return config
  
  def call(self, encoded_patches):
    target = encoded_patches
    for _ in range(self.transformer_layers):
      query_tgt = key_tgt = target
      attn_target1 = self.mha(query=query_tgt, value=target, key=key_tgt)
      attn_target1 = self.drop(attn_target1)
      target += attn_target1
      target = self.norm(target)
      query_tgt = target
      attn_target2 = self.mha(query=query_tgt, value=encoded_patches, key=encoded_patches)
      attn_target2 = self.drop(attn_target2)
      target += attn_target2
      target = self.norm(target)
      ffn_out = self.mlp(encoded_patches)
      target += ffn_out
      target = self.norm(target)
    return target




#--- Hierarchical Transformer Encoder Layer

class HTransformerEnc(layers.Layer):
  
  def __init__(self, iterations, transformer_layers, num_heads, input_resolution, embed_shape, rate):
    super(HTransformerEnc, self).__init__()
    self.iterations = iterations
    self.transformer_layers = transformer_layers
    self.num_heads = num_heads   
    self.input_resolution = input_resolution
    self.embed_shape = embed_shape
    self.rate = rate
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'iterations': self.iterations,
        'transformer_layers': self.transformer_layers,
        'num_heads': self.num_heads,
        'input_resolution': self.input_resolution,
        'embed_shape': self.embed_shape,
        'rate': self.rate
        })
    return config
  
  def call(self, encoded_patches):   
    for iter in range(self.iterations):     
      #-- Transformer Encoder Block x2
      encoded_patches = TransformerEnc(self.transformer_layers, self.num_heads, self.embed_shape, self.rate)(encoded_patches)   
      #-- Patch Merging           
      encoded_patches = PatchMerging(
          input_resolution = self.input_resolution,
          dim = self.embed_shape[1])(encoded_patches)
      self.input_resolution = self.input_resolution // 2
      self.embed_shape = (self.input_resolution*self.input_resolution, self.embed_shape[1]*2)  
    #-- Transformer Encoder Block x2
    encoded_patches = encoded_patches = TransformerEnc(self.transformer_layers, self.num_heads, self.embed_shape, self.rate)(encoded_patches)     
    return encoded_patches



#--- Hierarchical Transformer Decoder Layer

class HTransformerDec(layers.Layer):
  def __init__(self, iterations, transformer_layers, num_heads, input_resolution, embed_shape, rate):
    super(HTransformerDec, self).__init__()
    self.iterations = iterations
    self.transformer_layers = transformer_layers
    self.num_heads = num_heads   
    self.input_resolution = input_resolution
    self.embed_shape = embed_shape
    self.rate = rate
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'iterations': self.iterations,
        'transformer_layers': self.transformer_layers,
        'num_heads': self.num_heads,
        'input_resolution': self.input_resolution,
        'embed_shape': self.embed_shape,
        'rate': self.rate
        })
    return config

  def call(self, encoded_patches):           
    #-- Transformer Decoder Block x2
    decoded_patches = TransformerDec(self.transformer_layers, self.num_heads, self.embed_shape, self.rate)(encoded_patches)
    for iter in range(self.iterations):
      #-- Patch Expanding
      decoded_patches = PatchExpanding(input_resolution = self.input_resolution,
                                       dim = self.embed_shape[1])(decoded_patches)
      self.input_resolution = self.input_resolution * 2
      self.embed_shape = (self.input_resolution*self.input_resolution, self.embed_shape[1]//2)
      #-- Transformer Decoder Block x2   
      decoded_patches = TransformerDec(self.transformer_layers, self.num_heads, self.embed_shape, self.rate)(decoded_patches)
    return decoded_patches



#--- Build the HTAE Model

def HTransformerAE(input_resolution, embed_shape):
   
  input_img = tf.keras.Input(shape=(image_size, image_size, num_channels))

  projection = layers.Conv2D(filters=filters, kernel_size=patch_size, strides=patch_size)(input_img)
  projection = layers.LayerNormalization(epsilon=1e-5)(projection)
  B = tf.shape(projection)[0]
  enc_patches = tf.reshape(projection, [B, input_resolution*input_resolution, filters])

  encoded_patches = HTransformerEnc(iterations, transformer_layers, num_heads, input_resolution, embed_shape, rate)(enc_patches) 
  
  embed_shape = (encoded_patches.shape[1], encoded_patches.shape[2])
  input_resolution = int(math.sqrt(encoded_patches.shape[1]))

  decoded_patches = HTransformerDec(iterations, transformer_layers, num_heads, input_resolution, embed_shape, rate)(encoded_patches)
  
  decoded_patches = FinalPatchExpand_X4(input_resolution, filters)(decoded_patches)  
  B = tf.shape(decoded_patches)[0]
  decoded_patches = tf.reshape(decoded_patches, [B, image_size, image_size, filters])  
  reconstructed = layers.Conv2D(filters=num_channels, kernel_size=1, use_bias=False)(decoded_patches)

  model = tf.keras.Model(input_img, reconstructed)
  
  return model
  
    
#-- Configure the hyperparameters

model_name = 'Hierarchical Transformer Autoencoder'
numEpochs = 50
learning_rate = 0.00001
batch_size = 1
  
image_size = 256
num_channels = 1
rate = 0.


#-- Transformer parameters variation

param_grid = {
              'transformer_layers': [8],
              'patch_size' : [4],
              'num_heads' : [4],
              'filters' : [96]
             }
             
            
PARAMS = ParameterGrid(param_grid)
list_PARAMS = list(PARAMS)


#-- Configure training and testing Datasets 

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


#-- Train, Test and Evaluate

for param in list_PARAMS:
   try:

        #-- Checkpoints dir

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


        #-- Configure the parameters

        transformer_layers = param['transformer_layers']
        patch_size = param['patch_size']
        num_heads = param['num_heads']
        filters = param['filters']
        input_resolution = image_size // patch_size
        num_patches = input_resolution ** 2
        embed_shape = (num_patches, filters)


        #-- Configure the training

        opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)

        calbks = tf.keras.callbacks.ModelCheckpoint(filepath=ckpts_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=2)
        tqdm_callback = tfa.callbacks.TQDMProgressBar()

        model = TAE()
        model.summary()
        model.compile(optimizer=opt, loss='mse', metrics=['mae', SSIMLoss, MS_SSIMLoss])


        #-- Print & Write model Parameters

        parameters = (f'\nSelected model "{model_name}" with :\n - {num_heads}: Heads,\n - {patch_size}: Patch size,\n - {transformer_layers}: Transformer Layer,\n - ({embed_shape[0]}, {embed_shape[1]}): Embedding Shape,\n - {batch_size}: Batche(s)\n - {numEpochs}: Epochs\n')
        print(parameters)

        
        #-- Train

        print('\nTrain =>\n')
        history = model.fit(x = data_gen,
                            steps_per_epoch = training_steps,
                            validation_data = val_gen,
                            validation_steps = validation_steps,
                            verbose = 0,
                            epochs = numEpochs,
                            callbacks = [calbks, tqdm_callback]
                            )
        

        #-- Get training and test loss historie
        plot_history(history, path=fig_path)
        plt.close()
        time.sleep(2)               
        
        
        #-- Test 

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

        print('\nEnd !\n')


   except:
        print('Error encountered !')
        tf.keras.backend.clear_session()
        continue
