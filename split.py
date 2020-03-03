#!/usr/bin/env python

import os
import sys
import numpy as np

def simple_split(X, Y, f_train):
    # TODO: add option for structure selection via structure_idxs argument
    """
        Perform train-test split of a dataset

        ---Arguments---
        X: independent variable
        Y: dependent variable

        ---Returns---
        X_train: X training set
        X_test: X test set
        Y_train: Y data corresponding to the X training set
        Y_test: Y data corresponding to the X test set
    """

    # Check for consistent shapes
    if X.shape[0] != Y.shape[0]:
        print("Error: X and Y must have same length")
        return

    # Build array of all sample indices
    n_total = X.shape[0]
    idxs = np.arange(0, n_total, dtype=np.int)
    np.random.shuffle(idxs)

    # Number of training points
    n_train = int(np.floor(f_train*n_total))

    # Build lists of indices of training and test sets
    idxs_train = idxs[0:n_train]
    idxs_test = idxs[n_train:]
    
    # Split the dataset according to training and test indices
    X_train = X[idxs_train]
    X_test = X[idxs_test]
    Y_train = Y[idxs_train]
    Y_test = Y[idxs_test]

    return X_train, X_test, Y_train, Y_test

def cv_split(X, Y, k, stratified=False):
    # TODO: add option for structure selection via structure_idxs argument
    """
        Performs a k-fold cross-validation split of a dataset

        ---Arguments---
        X: independent variable
        Y: dependent variable
        k: number of folds
        stratified: flag to perform stratified cross validation

        ---Returns---
        X_folds: list of k subarrays of X data (i.e., split into k folds)
        Y_folds: list of k subarrays of Y data (i.e., split into k folds)
    """

    # Check for consistent shapes
    if X.shape[0] != Y.shape[0]:
        print("Error: X and Y must have same length")
        return

    # Total number of samples
    n_samples = X.shape[0]

    # Check for valid splitting
    try:
        idxs_split = np.split(np.arange(0, n_samples), k)
    except ValueError:
        print("Error: number of samples must be divisible by k; "
                "choose a different k or change the sample size")

    # Stratified sampling
    if stratified:

        # Sort by property
        idxs_sort = np.argsort(Y, axis=0)
        Y = Y[idxs_sort]
        X = X[idxs_sort]

        # Split the sorted data
        Y = [Y[i] for i in idxs_split]
        X = [X[i] for i in idxs_split]

        # Shuffle the data in the splits
        for x, y in zip(X, Y):
            idxs = np.arange(0, y.shape[0])
            np.random.shuffle(idxs)
            y = y[idxs]
            x = x[idxs]

        # Concantenate the shuffled splits
        Y = np.concatenate(Y)
        X = np.concatenate(X)

        # Get new folds
        Y_folds = [Y[i::k] for i in range(0, k)]
        X_folds = [X[i::k] for i in range(0, k)]

    # Standard sampling
    else:
        X_folds = [X[i] for i in idxs_split]
        Y_folds = [Y[i] for i in idxs_split]

    return X_folds, Y_folds
