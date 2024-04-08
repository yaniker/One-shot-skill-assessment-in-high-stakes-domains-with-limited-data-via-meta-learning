"""
Author: Dr. Erim Yanik
Date: 04/05/2024
"""

import os
import torch
import numpy as np

def import_feats_SS(input_dir, feature_no, names, dataset_type):
    """
    Loads feature files into a numpy array from a specified directory and dataset type.

    Inputs:
        input_dir (str): Path to the directory containing feature files.
        feature_no (int): Identifier for the subdirectory corresponding to specific features.
        names (list of str): Filenames to import, without the extension.
        dataset_type (str): Category of the dataset ('PC', 'STB', 'Cholec'), which affects the path.

    Returns:
        numpy.ndarray: Array containing loaded features for each specified file.
    """

    X = []
    for name in names:
        # Adjust name based on dataset type if necessary. Please do not change this.
        if dataset_type == 'PC': name = name.replace("S", "L")

        file_path = os.path.join(input_dir, f'SS_feats_{dataset_type}_{feature_no}', f'{name}.pt')
        
        if os.path.isfile(file_path):
            a = torch.load(file_path).numpy().squeeze()
            X.append(list(a))

    return np.array(X, dtype=object)

def import_feats_SS_JIGSAWS(input_dir, feature_no, names, camera_angle, dataset_type):
    """
    Imports and resamples feature sequences for the JIGSAWS dataset from specified directories, adjusting for dataset variants and camera angles.

    Inputs:
        input_dir (str): Path to the directory containing feature files.
        feature_no (int): Identifier for the subdirectory corresponding to specific features.
        names (list of str): Filenames to import, without the extension.
        camera_angle (str): Specific camera angle identifier, part of the file name.
        dataset_type (str): General category of the dataset, affecting subdirectory naming.

    Returns:
        numpy.ndarray: Array containing loaded features for each specified file.
    """

    from scipy import signal
    X = []
    for name in names:
        file_path = os.path.join(input_dir, f'SS_feats_{dataset_type}_{feature_no}', 
                                 f'{name}_{camera_angle}.pt')
        if os.path.isfile(file_path):
            a = torch.load(file_path).numpy().squeeze()
            # Resample each sequence in 'a' to 1/30th of its original length. Resampling.
            resampled = [signal.resample(seq, len(seq)//30) for seq in a]
            X.append(resampled)

    return np.array(X, dtype=object)

def class_name_adjuster(yb, new_labels):
    """
    Adjusts class names to new labels provided by the user and returns them as a numpy array.
 
    Inputs:
        yb (list): The original list of class indices.
        new_labels (list): The new class labels to assign based on the indices in yb.
    
    Returns:
        numpy.ndarray: An array containing the new class labels corresponding to the original indices.
    """
    
    yb_new = [new_labels[i] for i in yb]
    return np.array(yb_new)  

def sample_adjuster(yb, names, X, n = 16):
    """
    Uniformly samples a fixed number of instances from each class within the dataset.
        
    Inputs:
        yb (numpy.ndarray): Class labels.
        names (numpy.ndarray): Names associated with each sample.
        X (numpy.ndarray): Feature sets for each sample.
        n (int): Number of samples to retain from each class.
        
    Returns:
        yb (numpy.ndarray): Adjusted class labels.
        names (numpy.ndarray): Adjusted names.
        X (numpy.ndarray): Adjusted feature sets.
    """
    
    assert all(len(arr) == len(yb) for arr in [yb, names, X]), print('All input arrays must have the same length.')
    
    unique = np.unique(yb)
    yyb, namess, XX = [], [], []
    for j in range(len(unique)):
        idxx = np.where(np.array(yb) == unique[j])
        yyyb = yb[idxx]
        select_idxx = np.random.choice(yyyb.shape[0], n, replace=False)  
        yyb.extend(yb[idxx][select_idxx])
        namess.extend(names[idxx][select_idxx])
        XX.extend(X[idxx][select_idxx])
        
    yb, names, X = np.array(yyb), np.array(namess), np.array(XX)
    return yb, names, X

def inp_properties(yb, name, verbose = False):  
    """
    Calculates and optionally prints the distribution of class samples in a given dataset.

    Inputs:
        yb (list): The list of class labels from the dataset.
        name (str): A descriptive name for the dataset, used in verbose output.
        verbose (bool, optional): A flag to enable printing of class distribution details. 

    Returns:
        The numpy array of minimum number of samples found across all classes.
    """

    unique, numb = np.unique(yb, return_counts = True)
    if verbose == True:
        print(f'\nFor {name}\nTotal sample no.: {yb.shape[0]}')
        print(f'Classes: {unique} / No. of samples in each: {numb}')
    return np.min(numb)

def shuffle_data(*args):
    """
    Randomly shuffles multiple arrays in unison, maintaining the correspondence between elements of
    each array.
       
    Inputs:
        args: Variable number of arrays to be shuffled. All arrays must have the same length.
    
    Returns:
        A tuple of arrays, each shuffled in the same order.
    """
    assert all(len(arr) == len(args[0]) for arr in args)
    permutation = np.random.permutation(len(args[0]))
    return tuple(arr[permutation] for arr in args)
    
def reorder_data(rows, *args):
    """
    Reorders elements in given arrays based on specified row indices, followed by those not included.

    Inputs:
        rows (list): List of indices to prioritize in the reordering.
        args: Arguments, where each argument is an array to be reordered.

    Returns:
        tuple: A tuple of arrays, each reordered according to the specified priority indices.
    """

    unique_rows = np.unique(rows)
    reordered_arrays = []
    for arg in args:
        all_indices = np.arange(len(arg))
        not_in_rows = np.setdiff1d(all_indices, unique_rows)
        reordered = np.concatenate((arg[unique_rows], arg[not_in_rows]), axis=0)
        reordered_arrays.append(reordered)

    return tuple(reordered_arrays)
