"""
# VBA_net and associated model code
# Copyright © 2024 Erim Yanik
# Licensed under the GNU General Public License v3.0
# You must retain this notice and attribute the original author.
# Full license: https://www.gnu.org/licenses/gpl-3.0.en.html
"""

"""
Nomenclature:
    PC     : pattern cutting
    STB    : laparoscopic suturintest_on_g
    ST/JST : robotic suturing
    KT/JKT : knot tying
    NP/JNP : needle passing
    Cholec : laparoscopic chlecystectomy
"""

import os, sys
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from protomaml import train_model, test_model

from utils import (SequentialDataset, convert_to_torch_tensors,
       to_tensor, zero_padder, X_normalize, shuffle_data,
       save_results, save_hyperparameters_2, save_model_parameters  
)

from import_datasets import *
from config import *

def main():   
    """
    The main function that controls the flow from data processing to saving results.
    """
    
    #The imported labels and features are sorted to their respective datasets.
    datasets = {'STB': (X_STB, yb_STB), 'PC': (X_PC, yb_PC), 'JST': (X_JST, yb_JST),
                'JKT': (X_JKT, yb_JKT), 'JNP': (X_JNP, yb_JNP)}
    
    #We decide the validation dataset and assign to the appropriate variables.
    val_dataset_name = args.val_dataset
    X_val_dataset, yb_val_dataset = datasets[val_dataset_name]
    X_val = X_val_dataset[0]
    X_val_test_query = X_val_dataset[1]
    yb_val = yb_val_dataset[0]
    yb_val_test_query = yb_val_dataset[1]
    
    #Remove the validation dataset and send the rest to training.
    del datasets[val_dataset_name]
    X_train = np.concatenate([datasets[key][0] for key in datasets])
    yb_train = np.concatenate([datasets[key][1] for key in datasets])
    
    #We shuffle the training labels and features to prevent bias and convert the labels to tensors.
    X_train, yb_train = shuffle_data(X_train, yb_train)
    yb_train, yb_val, yb_val_test_query = convert_to_torch_tensors(yb_train, yb_val,yb_val_test_query)
    
    #The validation features are assigned to support and quert sets and converted to torch tensors.
    X_val_test_support = to_tensor(zero_padder(X_val)) #To torch tensor
    for i in range(len(X_val_test_support)):
        X_val_test_support[i] = X_normalize(X_val_test_support[i]) #Normalize the tensors.
        
    X_val_test_query = to_tensor(zero_padder(X_val_test_query)) #To torch tensor
    for i in range(len(X_val_test_query)):
        X_val_test_query[i] = X_normalize(X_val_test_query[i]) #Normalize the tensors.
        
    #Initialize training, validation, and test subsets for sequential processing.   
    train_set = SequentialDataset(X_train, yb_train)
    val_set   = SequentialDataset(X_val, yb_val)
    val_set_test_support = SequentialDataset(X_val_test_support, yb_val)
    val_set_test_query = SequentialDataset(X_val_test_query, yb_val_test_query)
    
    #We print the number of unique classes in training and validation sets.
    print(f'\nTraining classes  : {torch.unique(train_set.targets).tolist()}')
    print(f'Validation classes: {torch.unique(val_set.targets).tolist()}\n')
    
    #Calls for the script to train the meta learner via the decided parameters.
    protomaml_model = train_model(train_set, val_set, batch_size = params['batch_size'],
                                  N_WAY = args.N_WAY, K_SHOT = args.K_SHOT, 
                                  loss_type = params['loss_func'], inp_size = args.SS_feat_no,
                                  weight_decay = params['weight_decay'], CHECKPOINT_PATH = save_path,
                                  seed = seed_value, patience = params['patience'],
                                  max_epochs = params['max_epochs'],
                                  min_epochs = params['min_epochs'],
                                  lr = params['lr'], lr_inner = params['lr_inner'],
                                  lr_output = params['lr_outer'], 
                                  num_inner_steps = params['num_inner_steps_training'])
    
    protomaml_model.hparams.num_inner_steps = params['num_inner_steps_testing']
    
    #For number of test shots, e.g., one-shot learning or few-shot learning, we test the adaptation performance of the trained model.
    for k in params['test_shots']:
        Y_ACT, Y_PRED, Y_PRED_SM = test_model(protomaml_model, val_set_test_support,
                                              val_set_test_query, seed = seed_value, K_SHOT = k)
        
        #Calls for the script to calculate the metrics, print them, and save the results
        save_results(save_path, k, Y_ACT, Y_PRED, Y_PRED_SM,
                     args.val_dataset, params['cl'], seed_value, dev)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #We fixed N_WAY to 2 and K_SHOT to 1 throughout our study but left it adjustable in case the user would like to experiment with different values.
    parser.add_argument('N_WAY', type = int, help = ': Num_classes per batch during training')
    parser.add_argument('K_SHOT', type = int, help = ': Few-shot No. during training')
    parser.add_argument('SS_feat_no', type = int, help = ': Number of self-supervised features.')
    parser.add_argument('val_dataset', type = str, help = ': PC, STB, JST, JKT, JNP')
    args = parser.parse_args()
    
    #The self-supervised feature size is predefined as given below.
#     assert args.SS_feat_no in [2,4,8,10,16,32,64,128], 'Please pick one of the: 2,4,8,16,32,64,128.'
    assert args.SS_feat_no in [8], 'You can only select 8 as the SS feature size at this moment.' #Remove this line and uncomment the one above for different implementations.
    
    dev = True #This will prevent results from overwriting the provided ones. Change it to False if you would like to save your results.
     
    warnings.filterwarnings('ignore')
    ROOT_DIR = os.path.abspath('')
    #Importing the hyperparameters for the model training.
    params = config(args.val_dataset) #DL model hyperparameters.
    n = params['n']
    
    #Decides where the results are saved.
    save_path_2 = os.path.join(ROOT_DIR, 'output',
                               'Results_' + str(args.SS_feat_no), args.val_dataset,
                               'NWAY_' + str(args.N_WAY) + '_KSHOT_' + str(args.K_SHOT))
    
    if not os.path.exists(save_path_2): os.makedirs(save_path_2)
    save_hyperparameters_2(save_path_2, params)
    save_model_parameters(save_path_2, config_model())
    
    #Import pattern cutting labels and feature sets.
    yb_PC, names_PC, X_PC = import_dataset_PC(ROOT_DIR, args.SS_feat_no, args.val_dataset, n).main()
    
    #Import laparoscopic suturing labels and feature sets.
    yb_STB, names_STB, X_STB = import_dataset_STB(ROOT_DIR, args.SS_feat_no, args.val_dataset,
                                                  n).main()
    
    #Import robotic suturing labels and feature sets.
    yb_JST, names_JST, X_JST = import_dataset_JIGSAWS(ROOT_DIR, args.SS_feat_no, 'ST', 
                                                      args.val_dataset, n).main()
    
    #Import knot tying labels and feature sets.
    yb_JKT, names_JKT, X_JKT = import_dataset_JIGSAWS(ROOT_DIR, args.SS_feat_no, 'KT', 
                                                     args.val_dataset, n).main()   
    
    #Import neddle passing labels and feature sets.
    yb_JNP, names_JNP, X_JNP = import_dataset_JIGSAWS(ROOT_DIR, args.SS_feat_no, 'NP', 
                                                      args.val_dataset, n).main()
    
    #Randomly select 100 seeds and run the model 100 times with each seed.
    seed_values = np.random.randint(0, 400000, 100)
    countt = 0
    for seed_value in seed_values:
        save_path = os.path.join(save_path_2, 'Seed_' + str(countt))
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        #Fix the randomized parameters with the seed for consistency.
        pl.seed_everything(seed_value)
        torch.manual_seed(seed_value)
        
        #Calls the main function.
        print(f'#################### RUN = {countt+1} ####################')
        main()
        countt += 1
        

