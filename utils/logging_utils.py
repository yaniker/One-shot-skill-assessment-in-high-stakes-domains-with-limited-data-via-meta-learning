"""
VBANet and associated model code
Copyright © 2024 Erim Yanik
Licensed under the GNU General Public License v3.0
You must retain this notice and attribute the original author (Erim Yanik).
Full license: https://www.gnu.org/licenses/gpl-3.0.en.html
"""

import os, sys
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve
from trustpy import CNTS

def save_results(save_path, k, y_actual, y_pred, y_pred_sm, val_dataset, cl, seed_value, dev):
    """
    Logs performance metrics, trustworthiness scores, and prediction details for k-shot learning experiments. 

    Inputs:
        save_path (str): Directory path for saving results.
        k (int): Number of adaptation samples indicating one-shot or few-shot learning.
        y_actual (list): Actual labels for the dataset.
        y_pred (list): Predicted labels using hardmax.
        y_pred_sm (list): Predicted labels probabilities using softmax.
        val_dataset (str): Identifier for the validation dataset.
        cl (list): Class names for trustworthiness analysis.
        seed_value (int): Seed value for reproducibility.
    """
    
    #The data is filtered out from 'na'. Please do not change.
    y_actual, y_pred, y_pred_argmax, y_pred_sm = _filter_valid_data(y_actual, y_pred, y_pred_sm)

    if val_dataset not in ['JST', 'JNP', 'JKT']:
        #Compute the confusion matrix and the metrics based on the outcome.
        conf_mat = confusion_matrix(y_actual, y_pred)
        TN, FP, FN, TP = conf_mat[0][0], conf_mat[0][1], conf_mat[1][0], conf_mat[1][1]
        ACC      = (TP+TN)/(TP+FP+FN+TN).squeeze()
        ROC_AUC  = roc_auc_score(y_actual, np.array(y_pred_argmax))
        lr_precisionb, lr_recallb, _ = precision_recall_curve(y_actual,
                                                              np.array(y_pred_argmax))

    else:
        #JIGSAWS has 3 classes so requires its own metric computation.
        ACC = _classification_JIGSAWS(y_actual, y_pred)

    print(f'\n########### For K-SHOT: k = {k} ###########\n')
    print('## PERFORMANCE ##')
    print('Accuracy    = {:.3f}'.format(ACC))
    if val_dataset not in ['JST', 'JNP', 'JKT']:
        print('ROC AUC     = {:.3f}'.format(ROC_AUC))
    
    #Compute the trustworthiness of the model based on the results.
    NTS = CNTS(oracle = np.array(y_actual), predictions = np.array(y_pred_sm),
               show_summary = False, export_summary = False).compute()
            
    if not dev:
        df         = pd.DataFrame({'Seed' : seed_value}, index = [0])
        save_name  = 'Seed.xlsx'
        _save_this_location (save_path, save_name, df)

        if val_dataset not in ['JST', 'JNP', 'JKT']:
            df         = pd.DataFrame({'ROC_AUC' : ROC_AUC, 'Accuracy'  : ACC,
                                       'NTS_0'   : NTS['class_0'], 
                                       'NTS_1'   : NTS['class_1'],
                                       'NTS_TN'  : NTS['class_0_correct'],
                                       'NTS_FP'  : NTS['class_0_incorrect'],
                                       'NTS_TP'  : NTS['class_0_correct'],  
                                       'NTS_FN'  : NTS['class_1_incorrect']
                                      }, index = [0])
            save_name  = 'Results_for_k{}.xlsx'.format(k)
            _save_this_location (save_path, save_name, df)

            df         = pd.DataFrame({ 
                                        'Softmax_0'   : np.array(y_pred_sm)[:,0],
                                        'Softmax_1'   : np.array(y_pred_sm)[:,1]
                                       })
            save_name  = 'Scores_Softmax_for_k{}.xlsx'.format(k)
            _save_this_location (save_path, save_name, df)

        elif val_dataset in ['JST', 'JNP', 'JKT']:
            df         = pd.DataFrame({'Accuracy' : ACC,
                                       'NTS_00'   : NTS[3],
                                       'NTS_01'   : NTS[4], 'NTS_02'   : NTS[5],
                                       'NTS_11'   : NTS[6], 'NTS_10'   : NTS[7],
                                       'NTS_12'   : NTS[8], 'NTS_22'   : NTS[9],
                                       'NTS_20'   : NTS[10],'NTS_21'   : NTS[11],
                                      }, index = [0])
            save_name  = 'Results_for_k{}.xlsx'.format(k)
            _save_this_location (save_path, save_name, df)

            df         = pd.DataFrame({ 
                                       'Softmax_0'   : np.array(y_pred_sm)[:,0],
                                       'Softmax_1'   : np.array(y_pred_sm)[:,1],
                                       'Softmax_2'   : np.array(y_pred_sm)[:,2]
                                      })
            save_name  = 'Scores_Softmax_for_k{}.xlsx'.format(k)
            _save_this_location (save_path, save_name, df)


        df = pd.DataFrame({ 
                           'Binary_Predicted' : np.array(y_pred),
                           'Binary_Actual'    : np.array(y_actual),
                          })
        save_name  = 'Scores_for_k{}.xlsx'.format(k)
        _save_this_location (save_path, save_name, df)
            
def save_results_cholec(save_path, k, y_actual, y_pred, y_pred_sm, val_dataset, cl, seed_value, dev):
    """
    Logs results for cholecystectomy datasets, including performance metrics and trustworthiness analysis.

    Inputs:
        save_path (str): Path to save the results.
        k (int): Number of adaptation samples for one-shot or few-shot learning.
        y_actual (list): Actual class labels.
        y_pred (list): Predicted class labels (hardmax).
        y_pred_sm (list): Predicted class probabilities (softmax).
        val_dataset (str): Name of the validation dataset.
        cl (list): Class names for trustworthiness analysis.
        seed_value (int): Seed value for reproducibility.
    """

    #The data is filtered out from 'na'. Please do not change.
    y_actual, y_pred, y_pred_argmax, y_pred_sm = _filter_valid_data(y_actual, y_pred, y_pred_sm)
    
    #Compute the confusion matrix and the metrics based on the outcome.
    conf_mat = confusion_matrix(y_actual, y_pred)
    TN, FP, FN, TP = conf_mat[0][0], conf_mat[0][1], conf_mat[1][0], conf_mat[1][1]
    ACC      = (TP+TN)/(TP+FP+FN+TN).squeeze()
    ROC_AUC  = roc_auc_score(y_actual, np.array(y_pred_argmax))
    lr_precisionb, lr_recallb, _ = precision_recall_curve(y_actual,
                                                          np.array(y_pred_argmax))
    
    #Compute the trustworthiness of the model based on the results.
    NTS = _binary_trustworthiness(y_actual, y_pred_sm, cl = cl).main()

    print(f'\n########### For K-SHOT: k = {k} ###########\n')
    print('## PERFORMANCE ##')
    print('Accuracy    = {:.3f}'.format(ACC))
    print('ROC AUC     = {:.3f}'.format(ROC_AUC))
    
    if not dev:
        df         = pd.DataFrame({'ROC_AUC'      : ROC_AUC, 'Accuracy'     : ACC,     
                                   'NTS_0'        : NTS[0],  'NTS_1'        : NTS[3],
                                   'NTS_TN'       : NTS[1],  'NTS_FP'       : NTS[2],
                                   'NTS_TP'       : NTS[4],  'NTS_FN'       : NTS[5]
                                  }, index = [0])
        save_name  = 'Results_for_k{}.xlsx'.format(k)
        _save_this_location (save_path, save_name, df)

        df         = pd.DataFrame({ 
                                   'Softmax_0'   : np.array(y_pred_sm)[:,0],
                                   'Softmax_1'   : np.array(y_pred_sm)[:,1],
                                  })
        save_name  = 'Scores_Softmax_for_k{}.xlsx'.format(k)
        _save_this_location (save_path, save_name, df)

        df = pd.DataFrame({ 
                           'Binary_Predicted' : np.array(y_pred),
                           'Binary_Actual'    : np.array(y_actual),
                          })
        save_name  = 'Scores_for_k{}.xlsx'.format(k)
        _save_this_location (save_path, save_name, df)

        df         = pd.DataFrame({'Seed' : seed_value}, index = [0])
        save_name  = 'Seed.xlsx'
        _save_this_location (save_path, save_name, df)
                  
def save_hyperparameters_2(save_path, params):  
    """
    Saves hyperparameters to an Excel file for model development tracking.

    Inputs:
        save_path (str): Destination path for saving hyperparameters.
        params (dict): Dictionary containing model development hyperparameters.
    """
    
    df         = pd.DataFrame({'batch_size' : params['batch_size'],
                               'lr' : params['lr'],
                               'lr_inner' : params['lr_inner'],
                               'lr_outer' : params['lr_outer'],
                               'weight_decay' : params['weight_decay'],
                               'num_inner_steps_training' : params['num_inner_steps_training'],
                               'num_inner_steps_testing' : params['num_inner_steps_testing'],
                               'max_epochs' : params['max_epochs'],
                               'min_epochs' : params['min_epochs'],
                               'patience' : params['patience'],
                               'loss_func': params['loss_func']
                               }, index = [0])

    save_name  = 'Hyperparameters.xlsx'
    _save_this_location (save_path, save_name, df)
    
def save_model_parameters(save_path, params):  
    """
    Saves model parameters to an Excel file for easy reference and reproducibility.

    Inputs:
        save_path (str): Destination path for saving model parameters.
        params (dict): Dictionary containing model architecture parameters.
    """
    
    df         = pd.DataFrame({'kernel_size_1' : params['kernel_size_1'],
                               'kernel_size_2' : params['kernel_size_2'],
                               'dilation_1' : params['dilation_1'],
                               'dilation_2' : params['dilation_2'],
                               'stride' : params['stride'],
                               'dropout_p' : params['dropout_p'],
                               'use_dropout': params['use_dropout'],
                               }, index = [0])

    save_name  = 'Model_parameters.xlsx'
    _save_this_location (save_path, save_name, df)
    
def _save_this_location (save_folder, save_name, df):
    """
    Saves a DataFrame to a specified location, handling potential errors.

    Inputs:
        save_folder (str): Folder path where the file will be saved.
        save_name (str): Filename for the saved Excel file.
        df (pandas.DataFrame): Data to be saved into an Excel file.
    """
    
    try:
        save_path = os.path.join(save_folder, save_name)
        writer = pd.ExcelWriter(save_path) 
        df.to_excel(writer, 'Sheet', index=False)
        writer.save()
    
    except:
        print(f"An error occurred while trying to save the file")

def _filter_valid_data(y_actual: list, y_pred: list,
                       y_pred_sm: list) -> Tuple[list, list, list, list]:
    """
    Filters out samples with 'na' in actual labels from given label lists, ensuring only valid data is processed.

    Inputs:
        y_actual (list): The actual labels, where 'na' indicates a sample to be filtered out.
        y_pred (list): The predicted labels in hardmax format.
        y_pred_sm (list): The predicted labels in softmax format.

    Returns:
        y_actual (list): Filtered actual labels, excluding 'na' entries.
        y_pred (list): Corresponding filtered hardmax predicted labels.
        y_pred_argmax (list): Filtered argmax from softmax predicted labels, corresponding to valid actual labels.
        y_pred_sm (list): Filtered softmax predicted labels.
    """

    try:
        valid_indices = [i for i, val in enumerate(y_actual) if val != 'na']
        y_actual = [y_actual[i] for i in valid_indices]
        y_pred   = [y_pred[i] for i in valid_indices]
        y_pred_argmax = [y_pred_sm[i][1] for i in valid_indices]
        y_pred_sm = [y_pred_sm[i] for i in valid_indices]
        
    except:
        print('Filtering the NAs from the actual and predicted arrays did not work')
        
    return y_actual, y_pred, y_pred_argmax, y_pred_sm

def _classification_JIGSAWS (y_actual, y_pred):
    """
    Calculates micro accuracy for the JIGSAWS surgical skill dataset, evaluating model performance across different skill levels.

    Inputs:
        y_actual (list): Actual skill labels.
        y_pred (list): Predicted skill labels.

    Returns:
        float: Micro accuracy across novice, intermediate, and expert classes.
    """


    sum_c = confusion_matrix(np.array(y_actual), np.round(y_pred))
    #Novice
    nTP = sum_c[0][0]
    nTN = sum_c[1][1] + sum_c[1][2] + sum_c[2][1] + sum_c[2][2]
    nFP = sum_c[0][1] + sum_c[0][2]
    nFN = sum_c[1][0] + sum_c[2][0]
    #Intermediate
    iTP = sum_c[1][1]
    iTN = sum_c[0][0] + sum_c[0][2] + sum_c[2][0] + sum_c[2][2]
    iFP = sum_c[1][0] + sum_c[1][2]
    iFN = sum_c[0][1] + sum_c[2][1]
    #Expert
    eTP = sum_c[2][2]
    eTN = sum_c[0][0] + sum_c[0][1] + sum_c[1][0] + sum_c[1][1]
    eFP = sum_c[2][0] + sum_c[2][1]
    eFN = sum_c[0][2] + sum_c[1][2]
    #Total
    TP = nTP + iTP + eTP
    TN = nTN + iTN + eTN
    FP = nFP + iFP + eFP
    FN = nFN + iFN + eFN
    micro_acc = (TP+TN)/(TP+FP+FN+TN)
    return micro_acc

