"""
# VBA_net and associated model code
# Copyright © 2024 Erim Yanik
# Licensed under the GNU General Public License v3.0
# You must retain this notice and attribute the original author.
# Full license: https://www.gnu.org/licenses/gpl-3.0.en.html
"""

"""
There is no model training in this script. It takes the best performing models from the training and validation by the other datasets of this study and uses these models to adapt to Laparoscopic Cholecystectomy for given gradient update and adaptation shots, i.e., k-shots.
"""

import os, sys
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from protomaml import test_model

from utils import (SequentialDataset, convert_to_torch_tensors,
       to_tensor, zero_padder, X_normalize, save_results  
)

from import_datasets import *
from config import *

def main():    
    X_val = X_CHOLEC[0]
    X_val_test_query = X_CHOLEC[1]
    yb_val = yb_CHOLEC[0]
    yb_val_test_query = yb_CHOLEC[1]
        
    #Convert the labels to torch tensor.
    yb_val, yb_val_test_query = convert_to_torch_tensors(yb_val,yb_val_test_query)
    
    #Batch_size = Sample size during testing
    X_val_test_support = to_tensor(zero_padder(X_val))
    
    for i in range(len(X_val_test_support)):
        X_val_test_support[i] = X_normalize(X_val_test_support[i])
        
    X_val_test_query = to_tensor(zero_padder(X_val_test_query)) #To torch tensor
    for i in range(len(X_val_test_query)): 
        X_val_test_query[i] = X_normalize(X_val_test_query[i]) #Normalize the tensors.
    
    #Initialize test subsets for sequential processing.
    val_set_test_support = SequentialDataset(X_val_test_support, yb_val)
    val_set_test_query = SequentialDataset(X_val_test_query, yb_val_test_query)
    
    #Calls the best performing  model on the validation dataset: check_dataset.
    best_model_path = os.path.join(model_path, 'ProtoMAML',
                                  'lightning_logs', 'version_0',
                                  'checkpoints')
    addit = os.listdir(best_model_path)
    best_model_path = os.path.join(best_model_path, addit[0])
    
    #Load the best model.
    protomaml_model = ProtoMAML.load_from_checkpoint(best_model_path)
    protomaml_model.hparams.num_inner_steps = params['num_inner_steps_testing']
        
    #For number of test shots, e.g., one-shot learning or few-shot learning, we test the adaptation performance of the trained model on laparoscopic cholecystectomy.
    for k in params['test_shots']:
        Y_ACT, Y_PRED, Y_PRED_SM = test_model(protomaml_model, val_set_test_support,
                                              val_set_test_query,
                                              seed = model_seed, K_SHOT = k)
        
        save_results_cholec(save_results_path, k, Y_ACT, Y_PRED, Y_PRED_SM,
                            test_dataset, params['cl'], model_seed, dev)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #We fixed N_WAY to 2 and K_SHOT to 1 throughout our study but left it adjustable in case the user would like to experiment with different values.
    parser.add_argument('N_WAY', type = int, help = ': Num_classes per batch')
    parser.add_argument('K_SHOT', type = int, help = ': Few-shot No.')
    parser.add_argument('SS_feat_no', type = int, help = ': Number_of_SS_feats.')
    #The check_dataset is the dataset that the meta learner was validated on during training. The other datasets were used to train the model.
    parser.add_argument('check_dataset', type = str, help = ': PC, STB, JST, JKT, JNP')
    args = parser.parse_args()
    
    #The self-supervised feature size is predefined as given below.
#     assert args.SS_feat_no in [2,4,8,10,16,32,64,128], 'Please pick one of the: 2,4,8,16,32,64,128.'
    assert args.SS_feat_no in [8], 'You can only select 8 as the SS feature size at this moment.' #Remove this line and uncomment the one above for different implementations.
    
    dev = True #This will prevent results from overwriting the provided ones. Change it to False if you would like to save your results.
    
    warnings.filterwarnings('ignore')
    ROOT_DIR = os.path.abspath('')
        
    #Importing the hyperparameters for the model training.
    test_dataset = 'Cholec'
    params = config(test_dataset) #DL model hyperparameters.
    n = params['n']
    
    #Import laparoscopic cholecystectomy labels and feature sets.
    yb_CHOLEC, names_CHOLEC, X_CHOLEC = import_dataset_Cholec(ROOT_DIR, args.SS_feat_no, 
                                                                        test_dataset, n = n).main()
            
    #Path of the best performing model on a given validation dataset: check_dataset.
    model_path_2 = os.path.join(ROOT_DIR, 'output',
                               'Results_' + str(args.SS_feat_no), 
                                args.check_dataset,
                               'NWAY_' + str(args.N_WAY) + '_KSHOT_' + str(args.K_SHOT))
    
    #Directory to save the results.
    save_results_path_2 = os.path.join(ROOT_DIR, 'output',
                                       'Results_' + str(args.SS_feat_no), 
                                       'Cholec', 'Val_' + args.check_dataset,
                                       'NWAY_' + str(args.N_WAY) + '_KSHOT_' + str(args.K_SHOT))
            
    countt = 0
    #Default: Calls all 100 runs of the check dataset. For this version: Only one run.
    for seed_value in range(1):
        if countt < 0:
            countt += 1
            continue

        model_path = os.path.join(model_path_2, 'Seed_' + str(countt))
        save_results_path = os.path.join(save_results_path_2, 'Seed_' + str(countt))
        if not os.path.exists(save_results_path): os.makedirs(save_results_path)
       
        model_seed = pd.read_excel(os.path.join(model_path, 'Seed.xlsx'))['Seed'].squeeze()
        #Fix the randomized parameters with the seed for consistency.
        pl.seed_everything(seed_value)
        torch.manual_seed(seed_value)
        
        #Calls the main function.
        print(f'#################### RUN = {countt+1} ####################')
        main()
        countt += 1
        
                


