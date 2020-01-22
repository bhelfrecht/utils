#!/usr/bin/env python

import os
import sys
import numpy as np

def FPS(X, n=0):
    """
        Does Farthest Point Selection on a set of points X
        Adapted from a routine by Michele Ceriotti

        ---Arguments---
        X: data on which to perform the FPS
        n: number of points to select (<= 0 for all points)

        ---Returns---
        fps_idxs: indices of the FPS points
        d: min max distances at each iteration
    """
    N = X.shape[0]

    # If desired number of points less than or equal to zero,
    # select all points
    if n <= 0:
        n = N

    # Initialize arrays to store distances and indices
    fps_idxs = np.zeros(n, dtype=np.int)
    d = np.zeros(n)

    # Pick first point at random
    idx = np.random.randint(0, N)
    fps_idxs[0] = idx

    # Compute distance from all points to the first point
    d1 = np.linalg.norm(X-X[idx], axis=1)**2

    # Loop over the remaining points...
    for i in range(1, n):

        # Get maximum distance and corresponding point
        fps_idxs[i] = np.argmax(d1)
        d[i-1] = np.amax(d1)

        # Exit if we have exhausted the unique points
        # (in which case we select a point we have selected before)
        if fps_idxs[i] in fps_idxs[0:i]:
            fps_idxs = fps_idxs[0:i]
            d = d[0:i]
            break

        # Compute distance from all points to the selected point
        d2 = np.linalg.norm(X-X[fps_idxs[i]], axis=1)**2

        # Set distances to minimum among the last two selected points
        d1 = np.minimum(d1, d2)

    return fps_idxs, d


def simple_split(X, Y, f_train):
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
