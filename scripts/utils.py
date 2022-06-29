# Imports

import os
import tensorflow as tf
from skimage import filters
import scipy 
import numpy as np
from sklearn import metrics


# Loss Function SSIM

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
  
  
def MS_SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))
  
  
# Data generator

def data_generator(data_file, batch_size):
  
    num_samples = len(data_file)

    while True:

      for offset in range(0, num_samples):

        # Get the samples you'll use in this batch
        batch_samples = data_file[offset : offset+1]
        
        # Initialise X_train and y_train arrays for this batch
        X_train = np.load(batch_samples[0][0])
        d = X_train.shape[0]
          
        for l in range(0, d, batch_size):
          X = X_train[l:l+batch_size, :, :]
          yield X, X


def list_of_paths(data_dir):

    list_paths = []
    for path in os.listdir(data_dir):
      full_path = os.path.join(data_dir, path)
      if os.path.isfile(full_path):
        list_paths += [[full_path]]
        
    list_paths.sort()

    return list_paths


# Post-processing

def calculate_residual(x, x_rec, x_prior):
    # x_prior threshold for BraTS : 0.52, for MSLUB : 0.57
    # you can use either x_prior or brainmask
    # r is referring to the residual 

    r = np.multiply(np.squeeze(x_prior), np.squeeze(x - x_rec))
    r[r < 0] = 0
    
    #r = np.multiply(np.squeeze(x_prior), np.squeeze(np.absolute(x - x_rec)))
    
    r = apply_3d_median_filter(r)        
    mask = squash_intensities(r)

    return mask

        
def calculate_residual_BP(x, x_rec, brainmask):  # Before Post-processing
    
    r = np.multiply(np.squeeze(brainmask), np.squeeze(x - x_rec))
    r[r < 0] = 0
    
    #r = np.multiply(np.squeeze(brainmask), np.squeeze(np.absolute(x - x_rec)))

    return r


def apply_3d_median_filter(volume, kernelsize=5):
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume

def add_colorbar(img):
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        img[i][j, -1] = 0.0
        img[i][j,  0] = 0.0
        img[i][-1, j] = 0.0
        img[i][0,  j] = 0.0

    return img

'''
def squash_intensities(img):
    k = 100
    offset = 0.5
    return 2.0 * ((1.0 / (1.0 + np.exp(-k * img))) - offset)
'''  

def squash_intensities(img):
    k = 100
    offset = 0.5
    my_thresh = 0.5
    squash_img = 2.0 * ((1.0 / (1.0 + np.exp(-k * img))) - offset)
    return cv2.threshold(squash_img, my_thresh, 1, cv2.THRESH_BINARY) #-- to binarize the residual
    
