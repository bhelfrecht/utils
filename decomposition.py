#!/usr/bin/env python

import os
import sys
import numpy as np
from kernels import center_kernel

class PCA(object):
    """
        Performs principal component analysis

        ---Attributes---
        n_pca: number of PCA components to retain
            (`None` retains all components)
        C: covariance matrix of the data
        U: eigenvalues of the covariance matrix
        V: eigenvectors of the covariance matrix

        ---Methods---
        fit: fit the PCA
        transform: transform data based on the PCA fit

        ---References---
        1.  https://en.wikipedia.org/wiki/Principal_component_analysis
        2.  M. E. Tipping 'Sparse Kernel Principal Component Analysis',
            Advances in Neural Information Processing Systems 13, 633-639, 2001
    """

    def __init__(self, n_pca=None):

        # Initialize attributes
        self.n_pca = n_pca
        self.C = None
        self.U = None
        self.V = None

    def fit(self, X):
        """
            Fits the PCA

            ---Arguments---
            X: centered data on which to build the PCA
        """

        # Compute covariance
        self.C = np.matmul(X.T, X)/(X.shape[0] - 1)

        # Compute eigendecomposition of covariance matrix
        self.U, self.V = np.linalg.eigh(self.C)
        self.U = np.flip(self.U, axis=0)
        self.V = np.flip(self.V, axis=1)

    def transform(self, X):
        """
            Transforms the PCA

            ---Arguments---
            X: centered data to transform based on the PCA
            
            ---Returns---
            T: transformed PCA scores
        """

        if self.V is None:
            print("Error: must fit the PCA before transforming")
        else:

            # Compute PCA scores
            T = np.matmul(X, self.V[:, 0:self.n_pca])

            return T

class KPCA(object):
    """
        Performs kernel principal component analysis on a dataset
        based on a kernel between all of the constituent data points
        
        ---Attributes---
        n_kpca: number of principal components to retain in the decomposition
        U: eigenvalues of the kernel matrix
        V: eigenvectors of the kernel matrix
        
        ---Methods---
        fit: fit the KPCA
        transform: transform the KPCA
        
        ---References---
        1.  https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        2.  M. E. Tipping, 'Sparse Kernel Principal Component Analysis',
            Advances in Neural Information Processing Systems 13, 633-639, 2001  
    """
    
    def __init__(self, n_kpca=None):
        self.n_kpca = n_kpca
        self.U = None
        self.V = None
    
    def fit(self, K):
        """
            Fits the kernel PCA

            ---Arguments---
            K: centered kernel matrix with which to build the KPCA
        """
        
        # Compute eigendecomposition of kernel
        self.U, self.V = np.linalg.eigh(K)
        self.U = np.flip(self.U, axis=0)
        self.V = np.flip(self.V, axis=1)
        self.V = self.V[:, self.U > 0]
        self.U = self.U[self.U > 0]
    
    def transform(self, K):
        """
            Transforms the KPCA
            
            ---Arguments---
            K: centered kernel matrix to transform
            
            ---Returns---
            T: centered KPCA scores
        """

        if self.V is None or self.U is None:
            print("Error: must fit the KPCA before transforming")
        else:
            
            # Compute KPCA transformation
            T = np.matmul(K, self.V[:, 0:self.n_kpca])
            T = np.matmul(T, np.diagflat(1.0/np.sqrt(self.U[0:self.n_kpca])))

            return T

class SparseKPCA(object):
    """
        Performs sparsified principal component analysis
        
        ---Attributes---
        n_kpca: number of principal components to retain
        T_mean: the column means of the approximate feature space
        Um: eigenvectors of KMM
        Vm: eigenvalues of KMM
        Uc: eigenvalues of the covariance of T
        Vc: eigenvectors of the covariance of T
        V: projection matrix

        ---Methods---
        fit: fit the sparse KPCA
        transform: transform the sparse KPCA
        
        ---References---
        1.  https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        2.  M. E. Tipping 'Sparse Kernel Principal Component Analysis',
            Advances in Neural Information Processing Systems 13, 633-639, 2001
        3.  C. Williams, M. Seeger, 'Using the Nystrom Method to Speed Up Kernel Machines',
            Avnaces in Neural Information Processing Systems 13, 682-688, 2001
        4.  K. Zhang, I. W. Tsang, J. T. Kwok, 'Improved Nystrom Low-Rank Approximation
            and Error Analysis', Proceedings of the 25th International Conference
            on Machine Learning, 1232-1239, 2008
    """
    
    def __init__(self, n_kpca=None):
        self.n_kpca = n_kpca
        self.T_mean = None
        self.Um = None
        self.Vm = None
        self.Uc = None
        self.Vc = None
        self.V = None
        
    def fit(self, KNM, KMM):
        """
            Fits the sparse KPCA

            ---Arguments---
            KNM: centered kernel between all training points
                and the representative points
            KMM: centered kernel between the representative points
        """

        # Limit for zero eigenvalues to throw away
        tiny = 0.0 #1.0E-15

        # Compute eigendecomposition on KMM
        self.Um, self.Vm = np.linalg.eigh(KMM)
        self.Um = np.flip(self.Um, axis=0)
        self.Vm = np.flip(self.Vm, axis=1)
        self.Vm = self.Vm[:, self.Um > tiny]
        self.Um = self.Um[self.Um > tiny]
            
        # Compute a KPCA based on the eigendecomposition of KMM
        T = np.matmul(KNM, self.Vm)
        T = np.matmul(T, np.diagflat(1.0/np.sqrt(self.Um)))

        # Center the feature space
        self.T_mean = np.mean(T, axis=0)
        T -= self.T_mean

        # Compute covariance of projections, since the eigenvectors
        # of KMM are not necessarily uncorrelated for the whole
        # training set KNM
        C = np.matmul(T.T, T)

        # Eigendecomposition on the covariance
        self.Uc, self.Vc = np.linalg.eigh(C)
        self.Uc = np.flip(self.Uc, axis=0)
        self.Vc = np.flip(self.Vc, axis=1)

        # Compute projection matrix
        self.V = np.matmul(self.Vm, np.diagflat(1.0/np.sqrt(self.Um)))
        self.V = np.matmul(self.V, self.Vc)

    def transform(self, KNM):
        """
            Transforms the sparse KPCA

            ---Arguments---
            KNM: centered kernel between the training/testing
                points and the representatitve points

            ---Returns---
            T: transformed KPCA scores
        """

        if self.V is None:
            print("Error: must fit the KPCA before transforming")
        else:
            #T = np.matmul(KNM, self.V[:, 0:self.n_kpca])
            T = np.matmul(KNM, self.V)
            T -= np.matmul(self.T_mean, self.Vc)
            return T[:, 0:self.n_kpca]
