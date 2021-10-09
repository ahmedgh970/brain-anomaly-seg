import numpy as np
from sklearn import metrics
from scipy import stats


def dice_coef(y_true, y_pred, smooth=1):

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice


def eval_residuals(labels, residuals):
 
  res_list = []
  label_list = []
  
  shape = labels.shape

  for i in range(shape[0]):  
    r = residuals[i].flatten()
    res_list.append(r)   
    l = labels[i].flatten()   
    label_list.append(l) 
       
  res_vals = np.concatenate(res_list, axis=0)  
  label_vals = np.concatenate(label_list, axis=0)    
  
  AUPRC = metrics.average_precision_score(label_vals, res_vals)
  AUROC = metrics.roc_auc_score(label_vals, res_vals)            
  
  DICE = []
    
  for i in range(shape[0]):
    dice_i = dice_coef(label_list[i], res_list[i])
    DICE.append(dice_i)
  
  AVG_DICE = np.mean(DICE)
  STD = np.std(DICE)
  MAD = stats.median_absolute_deviation(DICE) 

  return [AUROC, AUPRC, AVG_DICE, MAD, STD], DICE
