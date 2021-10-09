import numpy as np
import cv2
import tensorflow as tf
import scipy


# x_prior threshold for BraTS : 0.52, for MSLUB : 0.57

#-- x_prior for BraTS ----------------------------

flair = np.load('./data/BraTS/s0/BraTS_Flair.npy')
seuil = 0.52
x_prior = 1 - np.squeeze(flair < seuil)
x_prior = scipy.ndimage.filters.median_filter(x_prior, (5, 5, 5))
np.save('./data/BraTS/s0/BraTS_prior_52.npy.npy', x_prior) 
print('BraTS x_prior Saved !')

#-------------------------------------------------

#-- x_prior for MSLUB ----------------------------

flair = np.load('./data/MSLUB/MSLUB_Flair.npy')
seuil = 0.57
x_prior = 1 - np.squeeze(flair < seuil)
x_prior = scipy.ndimage.filters.median_filter(x_prior, (5, 5, 5))
np.save('./data/MSLUB/MSLUB_prior_57.npy', x_prior)
print('MSLUB x_prior Saved !')

#-------------------------------------------------


