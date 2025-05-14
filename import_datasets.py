"""
Author: Dr. Erim Yanik
Date: 04/05/2024
"""

import pandas as pd
import numpy as np
import os, sys
import torch
from utils import inp_properties, shuffle_data, reorder_data, sample_adjuster, class_name_adjuster, import_feats_SS_JIGSAWS, import_feats_SS

"""
    Classes in this file are used for importing, adjusting, and optionally shuffling the specified dataset features and labels.

    Inputs:
        ROOT_DIR (str): Base directory containing the dataset and related files.
        feature_no (int): Identifier for the specific feature set to use.
        val_dataset (str): Specifies the validation dataset.
        n (int): Desired number of samples from each class for balancing.
        surgery_type (str): Specifies which JIGSAWS dataset is being used.
        
    import_dataset_PC: imports the labels and the input features for the pattern cutting dataset.
    import_dataset_STB: imports the labels and the input features for the laparoscopic suturing dataset.
    import_dataset_Cholec: imports the labels and the input features for the laparoscopic cholecystectomy dataset.
    import_dataset_JIGSAWS: imports the labels and the input features for the tasks of the JIGSAWS dataset, i.e., robotic suturing, needle passing, and knot tying.

    """

class import_dataset_PC:
    def __init__ (self, ROOT_DIR, feature_no, val_dataset, n = 16):
        self.INPUT_DIR = os.path.join(ROOT_DIR, 'input')
        self.feature_no = feature_no
        self.val_dataset = val_dataset
        self.n = n

    def main(self, shuffle = True):
        #Import the labels
        df = pd.read_excel(os.path.join(self.INPUT_DIR, 'meta_PC.xlsx'))
        names, yb = df['Names'].values, df['Scores_Binary'].values
        #Import the feature sets
        X = import_feats_SS(self.INPUT_DIR, self.feature_no, names, 'PC')
        #Decides the class with lowest number of samples and return the number.
        min_numb = inp_properties(yb, 'PC')
        
        #We adjust the number of samples that will be sent to the model based on whether the task is for training or validation. For training, we send the lowest number (n = 16) of all the datasets involved in the study for equal representation for all the classes. For validation, i.e., adapting task, we send the min_numb to have equal representation between classes of the adaptation task (only).
        yb, names, X = sample_adjuster(yb, names, X, 
                                       n = self.n if self.val_dataset != 'PC' else min_numb)
        
        if shuffle: X, yb, names = shuffle_data(X, yb, names) #Shuffle to prevent bias.
        
        #The involved datasets have classes 0,1,2.. so we adjust the class names to prevent classes from different datasets to have the same label.
        yb = class_name_adjuster(yb, [0,1])
        
        #Prints the class names, number of samples in each class etc. for visual validation.
        _ = inp_properties(yb, 'PC', verbose = True) #ybv
        
        if self.val_dataset == 'PC':
            _ = inp_properties(yb, 'PC_Test_Query', verbose=True)
            return [yb, yb], [names, names], [X, X]
        else:
            return yb, names, X
              
class import_dataset_STB:
    def __init__ (self, ROOT_DIR, feature_no, val_dataset, n = 16):
        self.INPUT_DIR = os.path.join(ROOT_DIR, 'input')
        self.feature_no = feature_no
        self.val_dataset = val_dataset
        self.n = n
        
    def main(self, shuffle = True):
        #Import the labels
        df = pd.read_excel(os.path.join(self.INPUT_DIR, 'meta_STB.xlsx'))
        names, yb = df['Names'].values, df['Scores_Binary'].values
        #Import the feature sets
        X = import_feats_SS(self.INPUT_DIR, self.feature_no, names, 'STB')
        #Decides the class with lowest number of samples and return the number.
        min_numb = inp_properties(yb, self.val_dataset)
                
        if self.val_dataset == 'STB':
            #We adjust the sample size.
            ybv, namesv, Xv = sample_adjuster(yb, names, X, n = min_numb)
            
            if shuffle == True: Xv, ybv, namesv = shuffle_data(Xv, ybv, namesv) #Shuffle to prevent bias.

            #Reorders given arrays based on specified row indices, followed by those not included.
            rows = [np.where(i == np.array(names))[0][0] for i in namesv]
            names, yb, X = reorder_data(rows, names, yb, X)    
            
            #We adjust the class names to prevent classes of diff. datasets to have the same label.
            yb = class_name_adjuster(yb, [2,3])
            ybv = class_name_adjuster(ybv, [2,3])
            
            #Prints the class names, number of samples in each class etc. for visual validation.
            _ = inp_properties(ybv, 'STB', verbose = True)
            _ = inp_properties(yb, 'STB_Test_Query', verbose = True)
                
            return [ybv, yb], [namesv, names], [Xv, X]

        elif self.val_dataset != 'STB':
            yb, names, X = sample_adjuster(yb, names, X, n = self.n)
            if shuffle == True: X, yb, names = shuffle_data(X, yb, names)
            yb = class_name_adjuster(yb, [2,3])
            _ = inp_properties(yb, 'STB', verbose = True)
            
            return yb, names, X
           
class import_dataset_Cholec:
    def __init__ (self, ROOT_DIR, feature_no, val_dataset, n = 16):
        self.INPUT_DIR = os.path.join(ROOT_DIR, 'input')
        self.feature_no = feature_no
        self.val_dataset = val_dataset
        self.n = n
    
    def main(self, shuffle = True):
        #Import the labels
        df      = pd.read_excel(os.path.join(self.INPUT_DIR, 'meta_Cholec.xlsx')) #meta
        names    = df['Names'].values
        ybb = df['GRS'].values
        
        #Pick the top OSATS scores as a separate class than the rest.
        yb = np.array([1 if i in [24, 25] else 0 for i in ybb])
        
        #Import the feature sets
        X = import_feats_SS(self.INPUT_DIR, self.feature_no, names, 'Cholec')
        
        if self.val_dataset == 'Cholec':
            #Decides the class with lowest number of samples and return the number.
            min_numb = inp_properties(yb, 'Cholec')
            #We adjust the sample size.
            ybv, namesv, Xv = sample_adjuster(yb, names, X, n = min_numb)
            
            if shuffle == True: 
                X, yb, names = shuffle_data(X, yb, names) #Shuffle to prevent bias.
                Xv, ybv, namesv = shuffle_data(Xv, ybv, namesv) #Shuffle to prevent bias.
            
            #Reorders given arrays based on specified row indices, followed by those not included.
            rows = [np.where(i == np.array(names))[0][0] for i in namesv]
            names, yb, X = reorder_data(rows, names, yb, X) 
           
            #We adjust the class names to prevent classes of diff. datasets to have the same label.
            yb = class_name_adjuster(yb, [14,15])
            ybv = class_name_adjuster(ybv, [14,15])
            
            #Prints the class names, number of samples in each class etc. for visual validation.
            _ = inp_properties(ybv, 'Cholec', verbose = True)
            _ = inp_properties(yb, 'Cholec_Test_Query', verbose = True)    
            return [ybv, yb], [namesv, names], [Xv, X]

        elif self.val_dataset != 'Cholec':
            min_numb = inp_properties(yb, 'Cholec')
            yb, names, X = sample_adjuster(yb, names, X, n = self.n)
            if shuffle == True: X, yb, names = shuffle_data(X, yb, names)
            yb = class_name_adjuster(yb, [14,15])    
            _ = inp_properties(yb, 'Cholec', verbose = True)
            return yb, names, X
                         
class import_dataset_JIGSAWS:
    def __init__ (self, ROOT_DIR, feature_no, surgery_type, val_dataset, n = 16):
        self.surgery_type = surgery_type
        self.INPUT_DIR = os.path.join(ROOT_DIR, 'input')
        self.feature_no = feature_no
        self.val_dataset = val_dataset
        self.n = n
        
    def importer(self, camera_angle):
        #Import the labels
        df = pd.read_excel(os.path.join(self.INPUT_DIR, 'meta_' + self.surgery_type + '.xlsx'))
        names, yb = df['Names'].values, df['Scores_Binary'].values
        #Import the feature sets
        X = import_feats_SS_JIGSAWS(self.INPUT_DIR, self.feature_no, names,
                                    camera_angle, self.surgery_type)
                
        if  self.surgery_type == 'ST':
            row_order = [0,5,10,15,20,25,30,34,
                         1,6,11,16,21,26,35,
                         2,7,12,17,22,27,31,36,
                         3,8,13,18,23,28,32,37,
                         4,9,14,19,24,29,33,38]
            final_order = [0,8,15,23,31,39]

        elif  self.surgery_type == 'NP':
            row_order = [0,4,9,14,18,
                         1,5,10,21,24,
                         2,6,11,15,19,25,
                         3,7,12,16,20,22,26,
                         8,13,17,23,27]
            final_order = [0,5,10,16,23,28]

        elif  self.surgery_type == 'KT':
            row_order = [0,4,9,14,19,24,32,
                         1,5,10,15,20,25,33,
                         2,6,11,16,21,26,29,34,
                         3,7,12,17,22,27,30,
                         8,13,18,23,28,31,35]
            final_order = [0,7,14,22,29,36]
                        
        return yb[row_order], names[row_order], X[row_order], final_order
    
    def fix_temporal_mismatch(self, X, X2, surgery_type = 'ST'):
        #Ensures that the two separate views of the JIGSAWS tasks do not get mismatched temporally.
        adjustment_map = {
                            ('ST'): {'X': [0, 32], 'X2': [7]},
                            ('NP'): {'X': [0, 5, 22, 24, 25, 26]},
                            ('KT'): {'X': [13]},
                         }

        indices_to_adjust = adjustment_map.get((surgery_type), {})

        for i in indices_to_adjust.get('X', []):
            for j in range(X.shape[1]):
                X[i][j] = X[i][j][:-1]

        for i in indices_to_adjust.get('X2', []):
            for j in range(X.shape[1]):
                X2[i][j] = X2[i][j][:-1]

        return X, X2  

    def main(self, shuffle = True):
        #Import the labels and features for the first camera view.
        yb, names, X1, final_order = self.importer('capture1')
        #Import the labels and features for the second camera view.
        _, _, X2, final_order      = self.importer('capture2')
        #Fixes the temporal mismatch between the views.
        X1, X2 = self.fix_temporal_mismatch(X1, X2, self.surgery_type)
        #Merges the separate views under one dataset.
        X = np.concatenate((X1, X2), axis = 0)
        yb = np.concatenate((yb, yb), axis = 0)
        
        names2 = np.array([i + '_2' for i in names])
        names = np.concatenate((names, names2))
                       
        surgery_mapping = {'JST': [4, 5, 6], 'JNP': [7, 8, 9], 'JKT': [10, 11, 12]}
        listt = surgery_mapping.get(f'J{self.surgery_type}', [])
        
        if self.val_dataset == 'J' + self.surgery_type:
            #Decides the class with lowest number of samples and return the number.
            min_numb = inp_properties(yb, 'J' + self.surgery_type)
            #We adjust the sample size.
            ybv, namesv, Xv = sample_adjuster(yb, names, X, n = min_numb)
            
            
            if shuffle == True: Xv, ybv, namesv = shuffle_data(Xv, ybv, namesv) #Shuffle to prevent bias.
            
            #Reorders given arrays based on specified row indices, followed by those not included.
            rows = [np.where(i == np.array(names))[0][0] for i in namesv]    
            names, yb, X = reorder_data(rows, names, yb, X)
            
            #We adjust the class names to prevent classes of diff. datasets to have the same label.
            yb = class_name_adjuster(yb, listt)
            ybv = class_name_adjuster(ybv, listt)
            
            #Prints the class names, number of samples in each class etc. for visual validation.
            _ = inp_properties(ybv, 'J' + self.surgery_type, verbose = True)
            _ = inp_properties(yb, 'J' + self.surgery_type + '_Test_Query', verbose = True)  
            
            return [ybv, yb], [namesv, names], [Xv, X]

        elif self.val_dataset != 'J' + self.surgery_type:
            min_numb = inp_properties(yb, 'J' + self.surgery_type)
            yb, names, X = sample_adjuster(yb, names, X, n = self.n)
            if shuffle == True: X, yb, names = shuffle_data(X, yb, names)
             
            yb = class_name_adjuster(yb, listt)    
            _ = inp_properties(yb, 'J' + self.surgery_type, verbose = True)
            return yb, names, X
                
