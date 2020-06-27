#!/usr/bin/env python

import os
import sys
import numpy as np
from kernels import center_kernel
from regression import KRR, SparseKRR, IterativeSparseKRR
from tools import sorted_eigh

# TODO: make regression parameters attributes of the object
# TODO: check proper centering in classes and center if not done
# TODO: eliminate auxiliary centering of T/phi in sparse methods
# TODO: make abstract base class with fit, transform

class PCA(object):
    """
        Performs principal component analysis

        ---Attributes---
        n_components: number of PCA components to retain
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

    def __init__(self, n_components=None, tiny=1.0E-15):

        # Initialize attributes
        self.n_components = n_components
        self.tiny = tiny 
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
        self.U, self.V = sorted_eigh(self.C, tiny=self.tiny)

        # Truncate the projections
        self.U = self.U[0:self.n_components]
        self.V = self.V[:, 0:self.n_components]

    def transform(self, X):
        """
            Transforms the PCA

            ---Arguments---
            X: centered data to transform based on the PCA
            
            ---Returns---
            T: centered transformed PCA scores
        """

        if self.V is None:
            print("Error: must fit the PCA before transforming")
        else:

            # Compute PCA scores
            T = np.matmul(X, self.V)

            return T

    def inverse_transform(self, X):
        """
            Reconstructs the original input data

            ---Arguments---
            X: centered data to be reconstructed

            ---Returns---
            Xr: centered reconstructed X data
        """

        if self.V is None:
            print("Error: must fit the PCA before transforming")
        else:

            # Compute reconstruction
            T = self.transform(X)
            Xr = np.matmul(T, self.V.T)

            return Xr

class KPCA(object):
    """
        Performs kernel principal component analysis on a dataset
        based on a kernel between all of the constituent data points
        
        ---Attributes---
        n_components: number of principal components to retain in the decomposition
        tiny: threshold for discarding small eigenvalues
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
    
    def __init__(self, n_components=None, tiny=1.0E-15):
        self.n_components = n_components
        self.tiny = tiny
        self.U = None
        self.V = None
    
    def fit(self, K):
        """
            Fits the kernel PCA

            ---Arguments---
            K: centered kernel matrix with which to build the KPCA
        """
        
        # Compute eigendecomposition of kernel
        self.U, self.V = sorted_eigh(K, tiny=self.tiny)

        # Truncate the projections
        self.U = self.U[0:self.n_components]
        self.V = self.V[:, 0:self.n_components]
    
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
            T = np.matmul(K, self.V)
            T = np.matmul(T, np.diagflat(1.0/np.sqrt(self.U)))

            return T

    def inverse_transform(self, KTT, KXT, X, reg=1.0E-12, rcond=None):
        """
            Computes the reconstruction of X

            ---Arguments---
            KTT: centered kernel between the KPCA transformed training data
            KXT: centered kernel between the transformed data and the 
                transformed training data
            X: the centered original input data
            reg: regularization for the KRR scheme to find the pre-image
            rcond: cutoff ratio for small singular values in the least squares
                solution to determine the inverse transform

            ---Returns---
            Xr: centered reconstructed input data

            ---References---
            1.  J. Weston, O. Chapelle, V. Vapnik, A. Elisseeff, B. Scholkopf,
                'Kernel Dependency Estimation', Advances in Neural Information
                Processing Systems 15, 897-904, 2003.
            2.  J. Weston, B. Scholkopf, G. Bakir, 'Learning to Find Pre-Images',
                Advances in Neural Information Processing Systems 16, 449-456, 2004.
        """

        # Build the KRR model and get the weights
        # (Can also use LR solution)
        krr = KRR(reg=reg, rcond=rcond)
        krr.fit(KTT, X)
        W = krr.W

        # Compute the reconstruction
        Xr = np.matmul(KXT, W)

        return Xr

class SparseKPCA(object):
    """
        Performs sparsified principal component analysis
        
        ---Attributes---
        n_components: number of principal components to retain
        T_mean: the column means of the approximate feature space
        tiny: threshold for discarding small eigenvalues
        T_mean: auxiliary centering of the kernel matrix
            because the centering must be based on the
            feature space, which is approximated
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
    
    def __init__(self, n_components=None, tiny=1.0E-15):
        self.n_components = n_components
        self.tiny = tiny
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

        # Compute eigendecomposition on KMM
        self.Um, self.Vm = sorted_eigh(KMM, tiny=self.tiny)

        # Compute a KPCA based on the eigendecomposition of KMM
        T = np.matmul(KNM, self.Vm)
        T = np.matmul(T, np.diagflat(1.0/np.sqrt(self.Um)))

        # Auxiliary centering of T
        # since we are working with an approximate feature space
        self.T_mean = np.mean(T, axis=0)
        T -= self.T_mean

        # Compute covariance of projections, since the eigenvectors
        # of KMM are not necessarily uncorrelated for the whole
        # training set KNM
        C = np.matmul(T.T, T)

        # Eigendecomposition on the covariance
        self.Uc, self.Vc = sorted_eigh(C, tiny=None)

        self.T_mean = np.matmul(self.T_mean, self.Vc)

        # Compute projection matrix
        self.V = np.matmul(self.Vm, np.diagflat(1.0/np.sqrt(self.Um)))
        self.V = np.matmul(self.V, self.Vc)

        # Truncate the projections
        # TODO: how to compute and truncate the eigenvalues?
        self.V = self.V[:, 0:self.n_kpca]
        self.T_mean = self.T_mean[0:self.n_kpca]

    def transform(self, KNM):
        """
            Transforms the sparse KPCA

            ---Arguments---
            KNM: centered kernel between the training/testing
                points and the representatitve points

            ---Returns---
            T: centered transformed KPCA scores
        """

        if self.V is None:
            print("Error: must fit the KPCA before transforming")
        else:
            T = np.matmul(KNM, self.V) - self.T_mean
            return T


    def inverse_transform(self, KTM, KMM, KXM, X, sigma=1.0, reg=1.0E-12, rcond=None):
        """
            Computes the reconstruction of X

            ---Arguments---
            KTM: centered kernel between the KPCA transformed training data
                and the transformed representative points
            KMM: centered kernel between the transformed representative points
            KXM: centered kernel between the transformed data and the 
                representative transformed data
            X: the centered original input data
            sigma: regulariztion parameter 
            reg: additional regularization for the Sparse KRR solution
                for the inverse transform
            rcond: cutoff ratio for small singular values in the least squares
                solution to determine the inverse transform

            ---Returns---
            Xr: reconstructed centered input data
        """

        # Build the KRR model and get the weights
        # (can also do LR here)
        skrr = SparseKRR(sigma=sigma, reg=reg, rcond=rcond)
        skrr.fit(KTM, KMM, X)
        W = skrr.W

        # Compute the reconstruction
        Xr = np.matmul(KXM, W)

        return Xr

class IterativeSparseKPCA(object):
    """
        Performs sparsified principal component analysis
        using batches. Example usage:

        KMM = build_kernel(Xm, Xm)
        iskpca = IterativeSparseKPCA()
        iskpca.initialize_fit(KMM)
        for i in batches:
            KNMi = build_kernel(Xi, Xm)
            iskpca.fit_batch(KNMi)
        iskpca.finalize_fit()
        for i in batches:
            KNMi = build_kernel(Xi, Xm)
            iskpca.transform(KNMi)
        
        ---Attributes---
        n_components: number of principal components to retain
        T_mean: the column means of the approximate feature space
        n_samples: number of training points
        tiny: threshold for discarding small eigenvalues
        T_mean: auxiliary centering of the kernel matrix
            because the centering must be based on the
            feature space, which is approximated
        Um: eigenvectors of KMM
        Vm: eigenvalues of KMM
        Uc: eigenvalues of the covariance of T
        Vc: eigenvectors of the covariance of T
        V: projection matrix
        iskrr: SparseKRR object used to construct the inverse transform

        ---Methods---
        initialize_fit: initialize the sparse KPCA fit
            (i.e., compute eigendecomposition of KMM)
        fit_batch: fit a batch of training data
        finalize_fit: finalize the sparse KPCA fitting procedure
            (i.e., compute the KPCA projection vectors)
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
        5.  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    
    def __init__(self, n_components=None, tiny=1.0E-15):
        self.n_components = n_components
        self.tiny = tiny
        self.C = None
        self.T_mean = None
        self.n_samples = None
        self.Um = None
        self.Vm = None
        self.Uc = None
        self.Vc = None
        self.V = None
        self.iskrr = None

    def initialize_fit(self, KMM):
        """
            Computes the eigendecomposition of the
            kernel matrix between the representative points

            ---Arguments---
            KMM: centered kernel between the representative points
        """

        # Compute eigendecomposition on KMM
        self.Um, self.Vm = sorted_eigh(KMM, tiny=self.tiny)

        # Set shape of T_mean and C according to the
        # number of nonzero eigenvalues
        self.C = np.zeros((self.Um.size, self.Um.size))
        self.T_mean = np.zeros(self.Um.size)
        self.n_samples = 0
        
    def fit_batch(self, KNM):
        """
            Fits a batch for the sparse KPCA

            ---Arguments---
            KNM: centered kernel between all training points
                and the representative points
        """

        if self.Um is None or self.Vm is None:
            print("Error: must initialize the fit with a KMM"
                    "before fitting batches")
            return

        # Reshape 1D arrays
        if KNM.ndim < 2:
            KNM = np.reshape(KNM, (1, -1))

        # Don't need to do auxiliary centering of T or KNM
        # since the covariance matrix will be centered once
        # we are finished building it

        # Compute a KPCA based on the eigendecomposition of KMM
        # TODO: also scale T?
        T = np.matmul(KNM, self.Vm)
        T = np.matmul(T, np.diagflat(1.0/np.sqrt(self.Um)))

        # Increment T_mean and number of samples
        old_mean = self.T_mean
        self.n_samples += KNM.shape[0]
        self.T_mean = old_mean + np.sum(T-old_mean, axis=0)/self.n_samples

        # Compute covariance of projections, since the eigenvectors
        # of KMM are not necessarily uncorrelated for the whole
        # training set KNM
        self.C += np.matmul((T-self.T_mean).T, T-old_mean)

    def finalize_fit(self):
        """
            Finalize the sparse KPCA fitting procedure
        """

        if self.n_samples < 1:
            print("Error: must fit at least one batch"
                    "before finalizing the fit")
            return

        # Eigendecomposition on the covariance
        self.Uc, self.Vc = sorted_eigh(self.C, tiny=None)

        # Compute T_mean
        self.T_mean = np.matmul(self.T_mean, self.Vc)

        # Compute projection matrix
        self.V = np.matmul(self.Vm, np.diagflat(1.0/np.sqrt(self.Um)))
        self.V = np.matmul(self.V, self.Vc)

        # Truncate the projections
        # TODO: how to compute and truncate the eigenvalues?
        self.V = self.V[:, 0:self.n_components]
        self.T_mean = self.T_mean[0:self.n_components]

    def transform(self, KNM):
        """
            Transforms the sparse KPCA

            ---Arguments---
            KNM: centered kernel between the training/testing
                points and the representatitve points

            ---Returns---
            T: centered transformed KPCA scores
        """

        if self.V is None:
            print("Error: must fit the KPCA before transforming")
        else:
            T = np.matmul(KNM, self.V) - self.T_mean
            return T

    def initialize_inverse_transform(self, KMM, x_dim=1, sigma=1.0, 
            reg=1.0E-12, reg_type='scalar', rcond=None):
        """
            Initialize the sparse KPCA inverse transform

            ---Arguments---
            KMM: centered kernel between the transformed representative points
            x_dim: dimension of X data
            sigma: regulariztion parameter 
            reg: additional regularization for the Sparse KRR solution
                for the inverse transform
            rcond: cutoff ratio for small singular values in the least squares
                solution to determine the inverse transform
        """

        # (can also do LR here)
        self.iskrr = IterativeSparseKRR(sigma=sigma, reg=reg, 
                reg_type=reg_type, rcond=rcond)
        self.iskrr.initialize_fit(KMM, y_dim=x_dim)

    def fit_inverse_transform_batch(self, KTM, X):
        """
            Fit a batch for the inverse KPCA transform

            ---Arguments---
            KTM: centered kernel between the KPCA transformed training data
                and the transformed representative points
            X: the centered original input data
        """

        self.iskrr.fit_batch(KTM, X)

    def finalize_inverse_transform(self):
        """
            Finalize the fitting of the inverse KPCA transform
        """

        self.iskrr.finalize_fit()

    def inverse_transform(self, KXM):
        """
            Computes the reconstruction of X

            ---Arguments---
            KXM: centered kernel between the transformed data and the 
                representative transformed data

            ---Returns---
            Xr: reconstructed centered input data
        """

        # Compute the reconstruction
        W = self.iskrr.W
        Xr = np.matmul(KXM, W)

        return Xr
