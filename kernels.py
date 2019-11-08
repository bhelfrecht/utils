#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy.spatial.distance import cdist

def linear_kernel(XA, XB, zeta=1):
    """
        Builds a dot product kernel

        ---Arguments---
        XA, XB: vectors of data with which to build the kernel,
            where each row is a sample and each column a feature

        ---Returns---
        K: dot product kernel between XA and XB
    """

    K = np.matmul(XA, XB.T)**zeta
    return K

def gaussian_kernel(XA, XB, gamma=1):
    """
        Builds a Gaussian kernel

        ---Arguments---
        XA, XB: vectors of data with which to build the kernel,
            where each row is a sample and each column a feature
        gamma: scaling parameter for the Gaussian

        ---Returns---
        K: Gaussian kernel between XA and XB
    """

    D = cdist(XA, XB, metric='sqeuclidean')
    K = np.exp(-gamma*D)
    return K

def center_kernel(K, Kref=None):
    """
        Centers a kernel matrix
        (written with assistance from Michele Ceriotti)

        ---Arguments---
        K: the kernel to center
        Kref: reference (training) kernel

        ---Returns---
        Kc: the centered kernel

        ---References---
        1. https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        2. B. Scholkopf, A. Smola, K.-R. Muller, Nonlinear Component Analysis
            as a Kernel Eigenvalue Problem, Neural Computation 10, 1299-1319 (1998).

    """

    if Kref is None:
        Kref = K

    if K.shape[1] != Kref.shape[0] or Kref.shape[0] != Kref.shape[1]:
        print("Error: kernels must have compatible shapes" \
                + "and the reference kernel must be square")
    else:
        oneN = np.ones((K.shape[0], K.shape[0]))/K.shape[1]
        oneM = np.ones((K.shape[1], K.shape[1]))/K.shape[1]

        Kc = K - np.matmul(oneN, Kref) - np.matmul(K, oneM) \
                + np.matmul(np.matmul(oneN, Kref), oneM)

        return Kc
