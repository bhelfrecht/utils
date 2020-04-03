#!/usr/bin/python

import numpy as np

def SUP(predicted, true):
    """
        Supremum absolute error

        ---Arguments---
        predicted: vector of predicted property values
        true: vector of true property values

        ---Returns---
        sup: maximum absolute error
    """
    if np.shape(predicted) != np.shape(true):
        sys.exit("Predicted and true vectors not same shape")
    else:
        abs_err = np.abs(predicted-true)
        sup = np.amax(abs_err, axis=0)
    return sup

def MAE(predicted, true):
    """
        Mean absolute error

        ---Arguments---
        predicted: vector of predicted property values
        true: vector of true property values

        ---Returns---
        mae: mean absolute error
    """
    if np.shape(predicted) != np.shape(true):
        sys.exit("Predicted and true vectors not same shape")
    else:
        abs_err = np.abs(predicted-true)
        mae = np.mean(abs_err, axis=0)
    return mae

def RMSE(predicted, true):
    """
        Root mean square error

        ---Arguments---
        predicted: vector of predicted property values
        true: vector of true property values

        ---Returns---
        rmse: root mean squared error
    """
    if np.shape(predicted) != np.shape(true):
        sys.exit("Predicted and true vectors not same shape")
    else:
        rmse = np.sqrt(np.mean(np.power(predicted-true, 2), axis=0))
    return rmse    
