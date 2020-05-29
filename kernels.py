#!/usr/bin/env python

import os
import sys
import functools
import numpy as np
from scipy.spatial.distance import cdist
from tools import sorted_eigh
from tqdm import tqdm

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
    U, V = sorted_eigh(KMM, tiny=tiny)

    # Build approximate feature space
    PhiNM = np.matmul(KNM, V)
    PhiNM = np.matmul(PhiNM, np.diagflat(1.0/sqrt(U)))

    return PhiNM

def kernel_decorator(kernel_func):
    """
        Decorator for kernel functions.

        ---Arguments---
        kernel_func: kernel function to wrap

        ---Returns---
        kernel_wrapper: wrapped kernel function
    """

    @functools.wraps(kernel_func)
    def kernel_wrapper(XA, XB, **kwargs):
        """
            Wrapper for kernel functions

            ---Arguments---
            XA, XB: datasets with which to build the kernel.
                If a dataset is provided as a list,
                the kernel is averaged over the corresponding
                axis, blocked according to the list elements
            kwargs: keyword arguments passed to the kernel functions

            ---Returns---
            K: the kernel matrix
        """

        # XA structures, XB structures
        if isinstance(XA, list) and isinstance(XB, list):
            K = np.zeros((len(XA), len(XB)))
            for adx, a in enumerate(tqdm(XA)):
                for bdx, b in enumerate(XB):
                    K[adx, bdx] = np.mean(kernel_func(a, b, **kwargs))

        # XA structures, XB environments
        elif isinstance(XA, list):
            K = np.zeros((len(XA), XB.shape[0]))
            for adx, a in enumerate(tqdm(XA)):
                K[adx, :] = np.mean(kernel_func(a, XB, **kwargs), axis=0)

        # XA environments, XB structures
        elif isinstance(XB, list):
            K = np.zeros((XA.shape[0], len(XB)))
            for bdx, b in enumerate(tqdm(XB)):
                K[:, bdx] = np.mean(kernel_func(XA, b, **kwargs), axis=1)

        # XA environments, XB environments
        else:
            K = kernel_func(XA, XB, **kwargs)

        return K

    return kernel_wrapper 

def subkernel_decorator(subkernel_func):
    """
        Decorator for subkernel functions.

        ---Arguments---
        subkernel_func: subkernel function to wrap

        ---Returns---
        subkernel_wrapper: wrapped subkernel function
    """

    @functools.wraps(subkernel_func)
    def subkernel_wrapper(XA, XB, **kwargs):
        """
            Wrapper for subkernel functions

            ---Arguments---
            XA, XB: datasets with which to build the subkernel.
                If a dataset is provided as a list,
                the subkernel is averaged over the corresponding
                axis, blocked according to the list elements
            kwargs: keyword arguments passed to the subkernel functions

            ---Returns---
            K: the subkernel vector 
        """

        # XA structures, XB structures
        if isinstance(XA, list) and isinstance(XB, list):

            # XA and XB should be the same length
            K = np.zeros(len(XA)) 

            # If both XA and XB are lists (structures),
            # we have to get the full distance matrix.
            # Since a, b will be simple numpy arrays,
            # we can just override the subkernel function
            # with the 'standard' kernel functions.
            # We have to access by name since the
            # function object appears to change by passing
            # through the decorator
            if subkernel_func.__name__ == 'linear_subkernel':
                subkernel_func_override = linear_kernel
            elif subkernel_func.__name__ == 'gaussian_subkernel':
                subkernel_func_override = gaussian_kernel
            else:
                print("Error: unrecognized kernel function")
                return
            for idx, (a, b) in enumerate(zip(tqdm(XA), XB)):
                K[idx] = np.mean(subkernel_func_override(a, b, **kwargs))

        # XA structures, XB environments
        elif isinstance(XA, list):
            K = np.zeros(len(XA))
            for idx, (a, b) in enumerate(zip(tqdm(XA), XB)):
                K[idx] = np.mean(subkernel_func(a, b, **kwargs), axis=0)

        # XA environments, XB structures
        elif isinstance(XB, list):
            K = np.zeros(len(XB))
            for idx, (a, b) in enumerate(tqdm(zip(XA, XB))):
                K[idx] = np.mean(subkernel_func(a, b, **kwargs), axis=1)

        # XA environments, XB environments
        else:
            K = subkernel_func(XA, XB, **kwargs)

        return K

    return subkernel_wrapper 

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
        
        # Build kernels between XA/XB/XR and XR
        KRR = kernel_func(XR, XR, **kw)
        KAR = kernel_func(XA, XR, **kw)
        KRB = kernel_func(XA, XR, **kw)
        
        # Build approximate kernel
        KRR_inv = np.linalg.inv(KRR)
        K = np.matmul(KAR, KRR_inv)
        K = np.matmul(K, KRB)
              
    # Normal mode
    else:

        # Build kernel between XA and XB
        K = kernel_func(XA, XB, **kw)

    return K

def build_subkernel(XA, XB, kernel='linear', gamma=1.0, zeta=1.0,
        section='diag', k=0):
    """
        Build a subkernel
        
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
        section: portion of the kernel to compute. Options are 
            'diag', 'upper', or 'lower' for computing the kernel diagonal,
            upper triangle, or lower triangle
        k: kth diagonal (0 for the main diagonal,
            k < 0 for below main diagonal, k > 0 for above main diagonal)
        
        ---Returns---
        K: vector of values from the kernel matrix in row major order
    """
    # TODO: Nystrom mode
    
    # Initialize kernel functions and special arguments
    if kernel == 'gaussian':
        kernel_func = gaussian_subkernel
        kw = {'gamma': gamma}
    else:
        kernel_func = linear_subkernel
        kw = {'zeta': zeta}

    if section == 'diag':
        XA_idxs, XB_idxs = diag_indices((len(XA), len(XB)), k=k)
    elif section == 'upper' or section == 'lower':
        XA_idxs, XB_idxs = tri_indices((len(XA), len(XB)), k=k,
                tri=section)
    else:
        print("Error: invalid selection. Valid options are "
                "'diag', 'upper', and 'lower'")
        return

    if isinstance(XA, list):
        XA = [XA[i] for i in XA_idxs]
    else:
        XA = XA[XA_idxs, :]

    if isinstance(XB, list):
        XB = [XB[i] for i in XB_idxs]
    else:
        XB = XB[XB_idxs, :]

    K = kernel_func(XA, XB, **kw)

    return K

def sqeuclidean_distances(XA, XB):
    """
        Evaluation of a distance matrix
        of squared euclidean distances

        ---Arugments---
        XA, XB: matrices of data with which to build the distance matrix,
            where each row is a sample and each column a feature

        ---Returns---
        D: distance matrix of shape A x B
    """

    # Reshape so arrays can be broadcast together into shape A x B
    XA2 = np.sum(XA**2, axis=1).reshape((-1, 1))
    XB2 = np.sum(XB**2, axis=1).reshape((1, -1))

    # Compute distance matrix
    D = XA2 + XB2 - 2*np.matmul(XA, XB.T)

    return D

def sqeuclidean_distances_vector(XA, XB): 
    """
        Evaluation of a vector
        of squared euclidean distances

        ---Arugments---
        XA, XB: matrices of data with which to build the distance matrix,
            where each row is a sample and each column a feature.
            The distance vector is computed between
            corresponding elements of XA and XB

        ---Returns---
        D: distance matrix of shape A x B
    """

    if XA.shape != XB.shape:
        print("Error: XA and XB must have same shape")
        return

    XA2 = np.sum(XA**2, axis=1)
    XB2 = np.sum(XB**2, axis=1)
    XAXB = np.sum(XA*XB, axis=1)

    # Compute distance matrix
    D = XA2 + XB2 - 2*XAXB

    return D

@kernel_decorator
def linear_kernel(XA, XB, zeta=1):
    """
        Builds a dot product kernel

        ---Arguments---
        XA, XB: matrices of data with which to build the kernel,
            where each row is a sample and each column a feature

        ---Returns---
        K: dot product kernel between XA and XB
    """

    K = np.matmul(XA, XB.T)**zeta
    return K

@kernel_decorator
def gaussian_kernel(XA, XB, gamma=1):
    """
        Builds a Gaussian kernel

        ---Arguments---
        XA, XB: matrices of data with which to build the kernel,
            where each row is a sample and each column a feature
        gamma: scaling parameter for the Gaussian

        ---Returns---
        K: Gaussian kernel between XA and XB
    """

    D = sqeuclidean_distances(XA, XB)
    K = np.exp(-gamma*D)
    return K

@subkernel_decorator
def gaussian_subkernel(XA, XB, gamma=1):
    """
        Computes a vector of Gaussian kernel values
        between corresponding samples

        ---Arguments---
        XA, XB: matrices of data with which to build the subkernel,
            where each row is a sample and each column a feature
        gamma: scaling parameter for the Gaussian

        ---Returns---
        K: Gaussian subkernel between corresponding samples
            in XA and XB
    """

    D = sqeuclidean_distances_vector(XA, XB)
    K = np.exp(-gamma*D)
    return K

@subkernel_decorator
def linear_subkernel(XA, XB, zeta=1):
    """
        Computes a vector of linear kernel values
        between corresponding samples

        ---Arguments---
        XA, XB: matrices of data with which to build the subkernel,
            where each row is a sample and each column a feature
        gamma: scaling parameter for the Gaussian

        ---Returns---
        K: Gaussian subkernel between corresponding samples
            in XA and XB
    """

    K = np.sum(XA*XB, axis=1)**zeta
    return K

def diag_indices(shape, k=0):
    """
        Computes the indices of the kth diagonal
        of a 2D matrix

        ---Arguments---
        shape: 2D tuple in the form (n_rows, n_columns)
        k: kth diagonal (0 for the main diagonal,
            k < 0 for below main diagonal, k > 0 for above main diagonal)

        ---Returns---
        idxs: tuple of array indices in the from (row_idxs, col_idxs)
    """

    row_start = np.abs(np.minimum(k, 0))
    row_end = np.minimum(np.abs(k - shape[1] + 1), shape[0] - 1)
    col_start = np.maximum(k, 0)
    col_end = np.minimum(k + shape[0] - 1, shape[1] - 1)

    row_idxs = np.arange(row_start, row_end + 1, dtype=int)
    col_idxs = np.arange(col_start, col_end + 1, dtype=int)
    idxs = (row_idxs, col_idxs)

    return idxs

def tri_indices(shape, k=0, tri='upper'):
    """
        Computes the indices of the upper or lower
        triangular matrix based on the diagonal

        ---Arguments---
        shape: 2D tuple in the form (n_rows, n_columns)
        k: kth diagonal (0 for the main diagonal,
            k < 0 for below main diagonal, k > 0 for above main diagonal)
        tri: 'upper' for upper triangular, 'lower' for lower triangular

        ---Returns---
        idxs: tuple of array indices in the form (row_idxs, col_idxs)
    """

    if tri == 'upper':
        start = k
        end = shape[1]

    elif tri == 'lower':
        start = -shape[0] + 1
        end = k + 1

    else:
        print("Error: 'tri' must be 'upper' or 'lower'")
        return

    row_idxs = []
    col_idxs = []
    for kk in np.arange(start, end):
        diag_idxs = diag_indices(shape, k=kk)
        row_idxs.append(diag_idxs[0])
        col_idxs.append(diag_idxs[1])

    row_idxs = np.concatenate(row_idxs)
    col_idxs = np.concatenate(col_idxs)
    row_idxs = np.sort(row_idxs)
    idxs = (row_idxs, col_idxs)

    return idxs

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

def center_kernel_fast(K, K_ref=None):
    """
        Centers a kernel matrix
        (written with assistance from Michele Ceriotti
        and Rose Cersonsky)

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
        col_mean = np.mean(K_ref, axis=0)
        row_mean = np.reshape(np.mean(K, axis=1), (-1, 1))
        k_mean = np.mean(K_ref)

        Kc = K - row_mean - col_mean + k_mean

        return Kc

def center_kernel_oos(K, K_bridge, K_ref):
    """
        Centers a kernel matrix
        with respect to a reference matrix with 
        no common elements (e.g., center a kernel matrix
        between the test set and itself relative to the
        kernel matrix between the train set and itself)

        ---Arguments---
        K: the kernel to center
        K_bridge: the kernel that "bridges" K and K_ref;
            for example, if K is the kernel between the test set
            and itself, and K_ref is the kernel between the train set
            and itself, K_bridge is the kernel between
            the test set and train set
        K_ref: reference (training) kernel

        ---Returns---
        Kc: the centered kernel
    """

    if (K.shape[0] != K.shape[1] or 
        K.shape[0] != K_bridge.shape[0] or
        K_bridge.shape[1] != K_ref.shape[0] or 
        K_ref.shape[0] != K_ref.shape[1]):
        print("Error: kernels must have compatible shapes " \
                + "and the reference kernel must be square")

    else:
        one_MN = np.ones((K.shape[0], K_ref.shape[0])) / K_ref.shape[0]
        one_NM = np.ones((K_ref.shape[0], K.shape[0])) / K_ref.shape[0]
        Kc = K - np.matmul(K_bridge, one_NM) - np.matmul(one_MN, K_bridge.T) + np.matmul(np.matmul(one_MN, K_ref), one_NM)

        return Kc

def center_kernel_oos_fast(K, K_bridge, K_ref):
    """
        Centers a kernel matrix
        with respect to a reference matrix with 
        no common elements (e.g., center a kernel matrix
        between the test set and itself relative to the
        kernel matrix between the train set and itself)

        ---Arguments---
        K: the kernel to center
        K_bridge: the kernel that "bridges" K and K_ref;
            for example, if K is the kernel between the test set
            and itself, and K_ref is the kernel between the train set
            and itself, K_bridge is the kernel between
            the test set and train set
        K_ref: reference (training) kernel

        ---Returns---
        Kc: the centered kernel
    """

    if (K.shape[0] != K.shape[1] or 
        K.shape[0] != K_bridge.shape[0] or
        K_bridge.shape[1] != K_ref.shape[0] or 
        K_ref.shape[0] != K_ref.shape[1]):
        print("Error: kernels must have compatible shapes " \
                + "and the reference kernel must be square")
    else:
    
        K_bridge_mean = np.mean(K_bridge.T, axis=0)
        K_bridge_mean_T = np.mean(K_bridge, axis=1).reshape((-1, 1))
        K_ref_mean = np.mean(K_ref)
        Kc = K - K_bridge_mean - K_bridge_mean_T + K_ref_mean

        return Kc
