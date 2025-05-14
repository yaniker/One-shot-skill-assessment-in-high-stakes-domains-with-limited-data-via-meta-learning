"""
Copyright © 2024 Erim Yanik
Licensed under the GNU General Public License v3.0
You must retain this notice and attribute the original author (Erim Yanik).
Full license: https://www.gnu.org/licenses/gpl-3.0.en.html
"""

import os
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset

class SequentialDataset(Dataset):
    def __init__(self, Xs: np.ndarray, targets: torch.Tensor) -> None:
        """
        Creates a dataset for sequential data processing in PyTorch.

        Parameters:
            Xs (numpy.ndarray): An array of input features.
            targets (torch.Tensor): A tensor of corresponding labels.
    
        Returns:
            tuple: A tuple containing an input feature and its corresponding label for a given index.
        """
        
        super().__init__()
        self.Xs = Xs
        self.targets = targets
        
    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, torch.Tensor]:
        x, y = self.Xs[idx], self.targets[idx]
        return x, y

def convert_to_torch_tensors(*args) -> torch.Tensor:
    """
    Accepts any number of numpy array arguments and converts each to its corresponding PyTorch tensor. 
Parameters:

    Inputs:    
        *args: Variable length argument list of numpy arrays to be converted.

    Returns:
        list: A list containing the converted PyTorch tensors for each input numpy array.
    """
    
    return [torch.from_numpy(array) for array in args]

def to_tensor(arg: np.ndarray) -> torch.Tensor:
    """
    Converts a single numpy array into a PyTorch tensor. 

    Inputs:
        arg (numpy.ndarray): An array to be converted into a tensor.

    Returns:
        torch.Tensor: The converted tensor from the numpy array.
    """
    
    return torch.from_numpy(np.array(arg.tolist()))
              
def split_batch(Xs: torch.Tensor, ys: torch.Tensor) -> Tuple:
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

def zero_padder(arg: np.ndarray) -> np.ndarray:
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
                arg[i][j] = np.pad(arg[i][j], (0, max_length - len(arg[i][j])), 'constant') 
    return np.array(arg)
    
def X_normalize(arg: torch.Tensor) -> torch.Tensor:
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
             
