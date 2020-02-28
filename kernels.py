#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy.spatial.distance import cdist

def build_phi(KNM, KMM, tiny=1.0E-15):
    """
        Build the approximate feature space based on the Nystrom Approximation.
        The feature space must still be centered afterwards.

        ---Arguments---
        KNM: centered kernel matrix between the points to transform
            and the representative points
        KMM: centered kernel matrix between the representative points

        ---Returns---
        PhiNM: Approximate RKHS features
    """

    # Eigendecomposition of KMM
    U, V = np.linalg.eigh(KMM)
    U = np.flip(U, axis=0)
    V = np.flip(V, axis=1)
    V = V[:, U > tiny]
    U = U[U > tiny]

    # Build approximate feature space
    PhiNM = np.matmul(KNM, V)
    PhiNM = np.matmul(PhiNM, np.diagflat(1.0/sqrt(U)))

    return PhiNM

def build_kernel(XA, XB, XR=None, kernel='linear', gamma=1.0, zeta=1.0):
    """
        Build a kernel
        
        ---Arguments---
        XA: XA data; if XA is a list, with element i being an array of environments
            in structure i, then row i of the kernel matrix will be an
            an average over environments in structure i
        XB: XB data; if XB is a list, with element j being an array of environments
            in structure j, then column j of the kernel matrix will be an
            average over environments in structure j
        XR: XR data (Nystrom mode); if XR is provided, a Nystrom approximation
            of the kernel will be computed. If XR is a list, with element k being
            an array of environments in structure k, then the column k
            in the kernel KAR, the row k in the kernel KRB, and the columns
            and rows in kernel KRR will be an average over environments in structure k
        kernel: kernel type (linear or gaussian)
        gamma: gamma (width) parameter for gaussian kernels
        zeta: zeta (exponent) parameter for linear kernels
        
        ---Returns---
        K: kernel matrix
    """
    
    # Initialize kernel functions and special arguments
    if kernel == 'gaussian':
        kernel_func = gaussian_kernel
        kw = {'gamma': gamma}
    else:
        kernel_func = linear_kernel
        kw = {'zeta': zeta}
        
    # Initialize kernel matrices
    KRR = None
    KAR = None
    KRB = None
    K = None

    # Compute the kernels, where we sum over the axes
    # corresponding to the data that are provided in lists, 
    # where each element of a list represents a structure
    # as an array with the feature vectors of the environments
    # present in that structure as rows
    
    # Nystrom mode
    if XR is not None:
        
        # XR structures
        if isinstance(XR, list):
            KRR = np.zeros((len(XR), len(XR)))
            for rdx1, r1 in enumerate(XR):
                for rdx2, r2 in enumerate(XR):
                    KRR[rdx1, rdx2] = np.sum(kernel_func(r1, r2, **kw))/(r1.shape[0]*r2.shape[0])
        
            # XA structures, XR structures
            if isinstance(XA, list):
                KAR = np.zeros((len(XA), len(XR)))
                for adx, a in enumerate(XA):
                    for rdx, r in enumerate(XR):
                        KAR[adx, rdx] = np.sum(kernel_func(a, r, **kw))/(a.shape[0]*r.shape[0])
            
            # XA environments, XR structures
            else:
                KAR = np.zeros((XA.shape[0], len(XR)))
                for rdx, r in enumerate(XR):
                    KAR[:, rdx] = np.sum(kernel_func(XA, r, **kw), axis=1)/r.shape[0]
                
            # XR structures, XB structures
            if isinstance(XB, list):
                KRB = np.zeros((len(XR), len(XB)))
                for rdx, r in enumerate(XR):
                    for bdx, b in enumerate(XB):
                        KRB[rdx, bdx] = np.sum(kernel_func(r, b, **kw))/(r.shape[0]*b.shape[0])
            
            # XR structures, XB environments
            else:
                KRB = np.zeros((len(XR), XB.shape[0]))
                for rdx, r in enumerate(XR):
                    KRB[rdx, :] = np.sum(kernel_func(r, XB, **kw), axis=0)/r.shape[0]
        
        # XR environments
        else:
            KRR = kernel_func(XR, XR, **kw)
            
            # XA structures, XR environments
            if isinstance(XA, list):
                KAR = np.zeros((len(XA), XR.shape[0]))
                for adx, a in enumerate(XA):
                    KAR[adx, :] = np.sum(kernel_func(a, XR, **kw), axis=0)/a.shape[0]
                    
            # XA environments, XR environments
            else:
                KAR = kernel_func(XA, XR, **kw)

            # XR environments, XB structures
            if isinstance(XB, list):
                KRB = np.zeros((XR.shape[0], len(XB)))
                for bdx, b in enumerate(XB):
                    KRB[:, bdx] = np.sum(kernel_func(XR, b, **kw), axis=1)/b.shape[0]
                    
            # XR environments, XB environments
            else:
                KRB = kernel_func(XR, XB, **kw)
         
        # Build approximate kernel
        KRR_inv = np.linalg.inv(KRR)
        K = np.matmul(KAR, KRR_inv)
        K = np.matmul(K, KRB)
              
    # Normal mode
    else:
        
        # XA structures, XB structures
        if isinstance(XA, list) and isinstance(XB, list):
            K = np.zeros((len(XA), len(XB)))
            for adx, a in enumerate(XA):
                for bdx, b in enumerate(XB):
                    K[adx, bdx] = np.sum(kernel_func(a, b, **kw))/(a.shape[0]*b.shape[0])

        # XA structures, XB environments
        elif isinstance(XA, list):
            K = np.zeros((len(XA), XB.shape[0]))
            for adx, a in enumerate(XA):
                K[adx, :] = np.sum(kernel_func(a, XB, **kw), axis=0)/a.shape[0]

        # XA environments, XB structures
        elif isinstance(XB, list):
            K = np.zeros((XA.shape[0], len(XB)))
            for bdx, b in enumerate(XB):
                K[:, bdx] = np.sum(kernel_func(XA, b, **kw), axis=1)/b.shape[0]

        # XA environments, XB environments
        else:
            K = kernel_func(XA, XB, **kw)        

    return K

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

def center_kernel(K, K_ref=None):
    """
        Centers a kernel matrix
        (written with assistance from Michele Ceriotti)

        ---Arguments---
        K: the kernel to center
        K_ref: reference (training) kernel

        ---Returns---
        Kc: the centered kernel

        ---References---
        1. https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        2. B. Scholkopf, A. Smola, K.-R. Muller, Nonlinear Component Analysis
            as a Kernel Eigenvalue Problem, Neural Computation 10, 1299-1319 (1998).

    """

    if K_ref is None:
        K_ref = K

    if K.shape[1] != K_ref.shape[0] or K_ref.shape[0] != K_ref.shape[1]:
        print("Error: kernels must have compatible shapes " \
                + "and the reference kernel must be square")
    else:
        oneNM = np.ones((K.shape[0], K.shape[1]))/K.shape[1]
        oneMM = np.ones((K.shape[1], K.shape[1]))/K.shape[1]

        Kc = K - np.matmul(oneNM, K_ref) - np.matmul(K, oneMM) \
                + np.matmul(np.matmul(oneNM, K_ref), oneMM)

        return Kc
