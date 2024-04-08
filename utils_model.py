"""
Author: Dr. Erim Yanik
Date: 04/05/2024
Licence: MIT Licence
"""

import os
import torch
import numpy as np

def convert_to_torch_tensors(*args):
    """
    Accepts any number of numpy array arguments and converts each to its corresponding PyTorch tensor. 
Parameters:

    Inputs:    
        *args: Variable length argument list of numpy arrays to be converted.

    Returns:
        list: A list containing the converted PyTorch tensors for each input numpy array.
    """
    
    torch_tensors = [torch.from_numpy(array) for array in args]
    return torch_tensors

def to_tensor(arg):
    """
    Converts a single numpy array into a PyTorch tensor. 

    Inputs:
        arg (numpy.ndarray): An array to be converted into a tensor.

    Returns:
        torch.Tensor: The converted tensor from the numpy array.
    """

    arg = torch.from_numpy(np.array(arg.tolist()))
    return arg
              
def split_batch(Xs, ys):
    """
    Splits a batch of input features and labels into support and query sets for few-shot learning tasks.

    Inputs:
        Xs (torch.Tensor): The input features batch to be divided.
        ys (torch.Tensor): The corresponding labels batch to be divided.

    Returns:
        support_Xs (torch.Tensor): The support set input features.
        query_Xs (torch.Tensor): The query set input features.
        support_ys (torch.Tensor): The support set labels.
        query_ys (torch.Tensor): The query set labels.
    """

    support_Xs, query_Xs = Xs.chunk(2, dim=0)
    support_ys, query_ys = ys.chunk(2, dim=0)
    return support_Xs, query_Xs, support_ys, query_ys

def zero_padder(arg):
    """
    Zero-pads each sequence within a numpy array to ensure they all have the same length.

    Inputs:
        arg (numpy.ndarray): An array of sequences to be zero-padded.

    Returns:
        numpy.ndarray: An array with sequences zero-padded to uniform length.
"""

    
    max_length = max(len(item) for row in arg for item in row)
    
    for i in range(arg.shape[0]):
        if len(arg[i][0]) != max_length:
            for j in range(arg.shape[1]):
                arg[i][j] = np.pad(arg[i][j], 
                                   (0, max_length - len(arg[i][j])),
                                   'constant') 
    return np.array(arg)
    
def X_normalize(arg):
    """
    Applies min-max normalization to the given PyTorch tensor.
    
    Inputs:
        arg (torch.Tensor): A tensor to be normalized.

    Returns:
        torch.Tensor: A tensor with its elements normalized between 0 and 1.
    """

    for i in range(arg.shape[0]):      
        arg[i,:] = (arg[i,:] - arg[i,:].min())/(arg[i,:].max() - arg[i,:].min())
    return arg
             