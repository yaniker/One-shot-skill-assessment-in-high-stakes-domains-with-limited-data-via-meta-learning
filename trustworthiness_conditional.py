"""
Author: Dr. Erim Yanik
Date: 04/05/2024
Licence: MIT Licence
"""

import os
import numpy as np

class binary_trustworthiness():
    """
    Defines a class for evaluating binary trustworthiness based on predicted softmax probabilities and actual binary outcomes.
    
    Inputs:
        y_test (list): Actual binary outcomes.
        y_pred_sm (list): Predicted probabilities from a softmax layer.
        cl (list of str): Names of the classes for reporting.
        alpha (int): Penalty term for confidence scores of correct predictions.
        beta (int): Penalty term for confidence scores of incorrect predictions.
    
    Returns:
        NTS (tuple): The trustworthiness scores.
    """

    def __init__ (self, y_test, y_pred_sm, cl, alpha = 1, beta = 1):
        self.y_test = y_test
        self.y_pred_sm = y_pred_sm
        self.cl = cl
        self.alpha = alpha
        self.beta = beta

    def main(self):
        oracle = np.array(self.y_test)
        actor_response = np.argmax(self.y_pred_sm, axis=-1)
        actor_confidence = np.array(self.y_pred_sm)

        # True if prediction matches the oracle, False otherwise
        correct_predictions = (oracle == actor_response)

        # Confidence scores adjusted by alpha for correct predictions, by beta for incorrect ones
        adjusted_confidence = np.where(correct_predictions[:, np.newaxis],
                                        actor_confidence * self.alpha,
                                        (1 - actor_confidence) * self.beta)

        # Split the adjusted confidence by class and correctness
        class_0_confidence = adjusted_confidence[:, 0]
        class_1_confidence = adjusted_confidence[:, 1]
        
        TN = class_0_confidence[~correct_predictions & (oracle == 0)]  # True Negatives
        FP = class_1_confidence[~correct_predictions & (oracle == 0)]  # False Positives
        TP = class_1_confidence[correct_predictions & (oracle == 1)]   # True Positives
        FN = class_0_confidence[~correct_predictions & (oracle == 1)]  # False Negatives

        TN_all = np.concatenate([TN, 1 - FP])  # All Class 0 predictions (True or False)
        TP_all = np.concatenate([TP, 1 - FN])  # All Class 1 predictions (True or False)

        args = [TN_all, TN, FP, TP_all, TP, FN]
        labels = ['All ' + self.cl[0], 'True Negatives', 'False Positives',
                  'All ' + self.cl[1], 'True Positives', 'False Negatives']
        
        NTS = [(1 / arg.shape[0]) * np.sum(arg) if arg.size else 'na' for arg in args]
        return NTS
                
class multi_trustworthiness():
    """
    Defines a class for evaluating multi-class trustworthiness based on predicted softmax probabilities and actual outcomes.
    
    Inputs:
        y_test (list): Actual outcomes.
        y_pred_sm (list): Predicted probabilities from a softmax layer.
        cl (list of str): Names of the classes for reporting.
        alpha (int): Penalty term for confidence scores of correct predictions.
        beta (int): Penalty term for confidence scores of incorrect predictions.
    
    Returns:
        NTS (tuple): The trustworthiness scores.
    """
    
    def __init__(self, y_test, y_pred_sm, cl=['0', '1', '2'], alpha=1, beta=1):
        self.y_test = y_test
        self.y_pred_sm = y_pred_sm
        self.cl = cl
        self.alpha = alpha
        self.beta = beta

    def main(self):
        oracle = np.array(self.y_test)
        actor_response = np.argmax(self.y_pred_sm, axis=-1)
        actor_confidence = np.array(self.y_pred_sm)
        
        # Initialize trust score collections
        TTs = {c: [] for c in self.cl}  # True Trust scores
        TFs = {f"{i}{j}": [] for i in self.cl for j in self.cl if i != j}  # False Trust scores
        TFis = {f"{i}{j}": [] for i in self.cl for j in self.cl if i != j}  # Inverse False Trust scores
        
        for i in range(len(oracle)):
            if oracle[i] == actor_response[i]:  # Correct predictions
                TTs[self.cl[oracle[i]]].append(self.alpha * np.max(actor_confidence[i]))
            else:  # Incorrect predictions
                wrong_label = f"{self.cl[oracle[i]]}{self.cl[actor_response[i]]}"
                TFs[wrong_label].append(self.beta * (1 - np.max(actor_confidence[i])))
                TFis[wrong_label].append(self.beta * np.max(actor_confidence[i]))
        
        # Calculate trust scores
        NTS = []
        for key in TTs:  # True Trust scores
            NTS.append(np.mean(TTs[key]) if TTs[key] else 'na')
        for key in TFs:  # False Trust scores and their inverse
            NTS.append(np.mean(TFs[key]) if TFs[key] else 'na')
            NTS.append(np.mean(TFis[key]) if TFis[key] else 'na')
        
        return NTS
