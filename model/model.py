"""
© 2025 Erim Yanik
Licensed under CC BY-SA 4.0
You must attribute the original author and share any derivatives under the same terms.
"""

import torch
import torch.nn as nn
from config import config_model

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def conv_padder (in_channels, out_channels, k, s, d):
    """
    Calculates the padding needed for a convolutional layer to maintain specific output dimensions.
    
    Inputs:
        in_channels (int): The number of channels in the input tensor.
        out_channels (int): The desired number of channels in the output tensor.
        k (int): The size of the kernel to be used in the convolution operation.
        s (int): The stride of the convolution operation, which specifies the movement of the kernel across the input tensor.
        d (int): The dilation size for the convolution operation, which indicates the spacing between kernel elements.
        
    Returns:
        int: The calculated padding value, rounded down to the nearest integer, necessary to achieve the desired output dimensions.
    """
    
    pad = (s*(out_channels-1)-in_channels+1+d*(k-1))/2
    return int(pad)

class attention (nn.Module):
    """
    An attention mechanism for 1D inputs using global average pooling and linear transformations. These weights are then applied to the original input to emphasize important features.
    
    Inputs:
        inc (int): Number of input channels.
    """
    
    def __init__(self, inc):
        super(attention, self).__init__()
        self.gap = torch.nn.AdaptiveAvgPool1d(1)       
        self.linear1 = nn.Linear(inc, int(inc/2), bias = False)
        self.linear2 = nn.Linear(int(inc/2), inc, bias = False)
        self.act = nn.Sigmoid()
        self.inc = inc
        
    def forward(self, x):
        x1 = self.gap(x)
        x1 = torch.squeeze(x1)
        x1 = self.linear1(x1)
        x1 = self.linear2(x1)
        x1 = self.act(x1)
        x1 = torch.reshape(x1, [-1, self.inc, 1])
        xx1 = torch.multiply(x, x1)

        return xx1
        
class ResBlock(nn.Module):
    """
    Constructs a residual block with two convolutional layers, each followed by an attention mechanism. It's designed for 1D input data, suitable for time-series or sequence data processing.
    
    Inputs:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor after the second convolution.
        k (int): Kernel size for the convolutional layers.
        s (int): Stride for the convolutional layers.
        d (int): Dilation for the convolutional layers.
    """
    
    def __init__(self, in_channels, out_channels, k, s, d):
        super(ResBlock, self).__init__()
        self.conv1d1 = nn.Conv1d(in_channels, in_channels, kernel_size = k, stride = s,
                                 dilation = d,
                                 padding = conv_padder(in_channels, in_channels, k, s, d))
        self.attention1 = attention(in_channels)
        self.conv1d2 = nn.Conv1d(in_channels, out_channels, kernel_size = k, stride = s,
                                 dilation = d,
                                 padding = conv_padder(in_channels, out_channels, k, s, d))
        self.attention2 = attention(out_channels)
        self.act = nn.ReLU()
    
    def batch_norm_1d(self, inp_shape): 
        """
        Creates a 1D batch normalization layer for a specified input shape.
        
        Inputs:
            inp_shape (int): The size of the input shape.

        Returns:
            A batch normalization layer configured for 1D input.
    """
        return nn.BatchNorm1d(inp_shape).to(device)

    def forward(self, x):
        x1 = self.conv1d1(x)
        x1 = self.act(x1)
        x1 = self.batch_norm_1d(x1.shape[1])(x1)
        x1 = self.attention1(x1)
        x1 = self.conv1d2(x1)
        xc = torch.add(x1, x)
        xc = self.act(xc)
        xc = self.batch_norm_1d(xc.shape[1])(xc)
        xc = self.attention2(xc)
        return xc
    
class model(nn.Module):
    """
    Defines model, a neural network model.

    Parameters:
        inp_size (int): The size of the input features/channels.
    """
    
    def __init__(self, inp_size):
        super(model, self).__init__()
        params = config_model()
        if inp_size in [2,4,8,10]: channel_size = 16; out_dim = 512
        elif inp_size in [16,32]: channel_size = 64; out_dim = 512
        elif inp_size in [64,128]: channel_size = 256; out_dim = 1024
            
        self.use_dropout = params['use_dropout']
        self.dropout = nn.Dropout(p=params['dropout_p'])       
        self.res_block1 = ResBlock(in_channels = inp_size, out_channels = inp_size,
                                   k = params['kernel_size_1'],
                                   s = params['stride'],
                                   d = params['dilation_1'])
        
        self.conv_1x1_1   = nn.Conv1d(inp_size, channel_size, kernel_size = 1,
                                    stride = 1, dilation = 1,
                                    padding = conv_padder(inp_size, channel_size, 1, 1, 1))
        
        self.res_block2 = ResBlock(in_channels = channel_size, out_channels = channel_size,
                                   k = params['kernel_size_2'],
                                   s = params['stride'],
                                   d = params['dilation_2'])
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channel_size, out_dim)
            
    def forward(self, x):
        x = self.res_block1(x)
        if self.use_dropout: x = self.dropout(x)
        x = self.conv_1x1_1(x)
        x = self.res_block2(x)
        if self.use_dropout: x = self.dropout(x)
        x = self.gap(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
                  
