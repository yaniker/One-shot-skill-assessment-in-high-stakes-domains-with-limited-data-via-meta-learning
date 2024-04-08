"""
Author: Dr. Erim Yanik
Date: 04/05/2024
Licence: MIT Licence
"""

from torch.utils.data import Dataset
# import torchvision.transforms as T
# from torchvision.transforms import ToTensor
# from PIL import Image

class SequentialDataset(Dataset):
    def __init__(self, Xs, targets, X_transform = None, y_transform = None):
        """
        Creates a dataset for sequential data processing in PyTorch.

        Parameters:
            Xs (numpy.ndarray): An array of input features.
            targets (torch.Tensor): A tensor of corresponding labels.
            X_transform (callable, optional): A function to apply to each input feature. 
                                              Defaults to None.
            y_transform (callable, optional): A function to apply to each label. Not used in this 
                                              study but included for extensibility. Defaults to None.
    
        Returns:
            tuple: A tuple containing an input feature and its corresponding label for a given index.
        """
        
        super().__init__()
        self.X_transform = X_transform
        self.y_transform = y_transform
        self.Xs = Xs
        self.targets = targets
        
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        x, y = self.Xs[idx], self.targets[idx]
        if self.X_transform == 'Normalize': x = self.X_normalize(x)
        return x, y

    

