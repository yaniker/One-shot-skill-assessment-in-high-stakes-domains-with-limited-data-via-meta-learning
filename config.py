"""
Author: Dr. Erim Yanik
Date: 04/05/2024
"""

def config(val_dataset):
    """
    Configures hyperparameters tailored to the specified validation dataset.
    
    Input:
        val_dataset (str): Identifier for the validation dataset.
    
    Returns:
        dict: A dictionary containing model hyperparameters and training settings specific to the validation dataset.
    """
    
    if val_dataset in ['JST', 'JKT']:
        params = {'batch_size': 8, 'lr' : 0.01, 'n' : 16, 'cl' : ['Novice', 'Intermediate', 'Expert'],
                  'lr_inner':  0.1, 'lr_outer': 0.1,
                  'weight_decay': 0, 'loss_func': 'CS',
                  'num_inner_steps_training': 1, 'num_inner_steps_testing': 20,
                  'test_shots': [1,2,4,8,16], 'max_epochs': 500, 'min_epochs': 40, 'patience': 10}
        return params
    
    elif val_dataset in ['STB']:
        params = {'batch_size': 8, 'lr' : 0.01, 'n' : 16, 'cl' : ['Novice', 'Expert'],
                  'lr_inner':  0.1, 'lr_outer': 0.1,
                  'weight_decay': 0, 'loss_func': 'CS',
                  'num_inner_steps_training': 1, 'num_inner_steps_testing': 20,
                  'test_shots': [1,2,4,8,16], 'max_epochs': 500, 'min_epochs': 40, 'patience': 10}
        return params
    
    elif val_dataset in ['JNP']:
        params = {'batch_size': 8, 'lr' : 0.01, 'n' : 20, 'cl' : ['Novice', 'Intermediate', 'Expert'],
                  'lr_inner':  0.1, 'lr_outer': 0.1,
                  'weight_decay': 0, 'loss_func': 'CS',
                  'num_inner_steps_training': 1, 'num_inner_steps_testing': 20,
                  'test_shots': [1,2,4,8,16], 'max_epochs': 500, 'min_epochs': 40, 'patience': 10}
        return params

    elif val_dataset in ['Cholec']:
        params = {'batch_size': 1, 'lr' : 0.01, 'n' : 16, 'cl' : ['Unsuccessful', 'Successful'],
                  'lr_inner':  0.1, 'lr_outer': 0.1,
                  'weight_decay': 0, 'loss_func': 'CS',
                  'num_inner_steps_training': 1, 'num_inner_steps_testing': 200,
                  'test_shots': [1], 'max_epochs': 500, 'min_epochs': 40, 'patience': 10}
        return params
    
    elif val_dataset in ['PC']:
        params = {'batch_size': 8, 'lr' : 0.01, 'n' : 16, 'cl' : ['Fail', 'Pass'],
                  'lr_inner':  0.1, 'lr_outer': 0.1,
                  'weight_decay': 0, 'loss_func': 'CS',
                  'num_inner_steps_training': 1, 'num_inner_steps_testing': 20,
                  'test_shots': [1,2,4,8,16,32,64], 'max_epochs': 500, 'min_epochs': 40, 
                  'patience': 10}
        return params
    
def config_model():
    """
    Retrieves the default hyperparameters for the deep learning model configuration.

    Returns:
        dict: A dictionary of model hyperparameters.
    """
    
    params_model = {'kernel_size_1': 5,
                    'kernel_size_2': 3,
                    'dilation_1': 1,
                    'dilation_2': 2,
                    'stride': 1,
                    'dropout_p' : 0.5,
                    'use_dropout': False,
                    }
    return params_model


  
    
