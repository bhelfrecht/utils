#!/usr/bin/env python

import os
import sys
import numpy as np

def FPS(X, n=0):
    """
        Does Farthest Point Selection on a set of points X
        Adapted from a routine by Michele Ceriotti
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


def split(X, Y, ftrain):
    if X.shape[0] != Y.shape[0]:
        print("Error: X and Y must have same length")
        return

    idxs = np.arange(0, X.shape[0], dtype=np.int)
    n_train = int(np.floor(ftrain*X.shape[0]))
    np.random.shuffle(idxs)
    idxs_train = idxs[0:n_train]
    idxs_test = idxs[n_train:]

    return X[idxs_train], X[idxs_test], Y[idxs_train], Y[idxs_test]
