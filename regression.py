#!/usr/bin/env python

import os
import sys
import numpy as np

class LR(object):
    """
        Performs linear regression

        ---Attributes---
        w: regression weights

        ---Methods---
        fit: fit the linear regression model by computing regression weights
        transform: compute predicted Y values

        ---References---
        1.  https://en.wikipedia.org/wiki/Linear_regression
    """

    def __init__(self):
        self.w = None

    def fit(self, X, Y):
        """
            Fits the linear regression model

            ---Arguments---
            X: centered, independent (predictor) variable
            Y: centered, dependent (response) variable
        """

        # Compute inverse of covariance
        XTX = np.linalg.pinv(np.matmul(X.T, X))

        # Compute LR solution
        self.w = np.matmul(XTX, X.T)
        self.w = np.matmul(self.w, Y)

    def transform(self, X):
        """
            Computes predicted Y values

            ---Arguments---
            X: centered, independent (predictor) variable
        """

        # Compute predicted Y
        Yp = np.matmul(X, self.w)

        return Yp

class KRR(object):
    """
        Performs kernel ridge regression
        
        ---Attributes---
        jitter: jitter/regularization parameter
        w: regression weights
        
        ---Methods---
        fit: fit the KRR model by computing regression weights
        transform: compute predicted Y values

        ---References---
        1.  M. Ceriotti, M. J. Willatt, G. Csanyi,
            'Machine Learning of Atomic-Scale Properties
            Based on Physical Principles', Handbook of Materials Modeling,
            Springer, 2018
    """
    
    def __init__(self, jitter=1.0E-16):
        self.jitter = jitter
        self.w = None
        
    def fit(self, K, Y):
        """
            Fits the KRR model by computing the regression weights

            ---Arguments---
            K: kernel between training data
            Y: property values
        """

        # Compute maximum eigenvalue of kernel matrix
        maxeig = np.amax(np.linalg.eigvalsh(K))

        # Regularize the model
        Kreg = K + np.eye(K.shape[0])*maxeig*self.jitter

        # Solve the model
        self.w = np.linalg.solve(Kreg, Y)
        
    def transform(self, K):
        """
            Computes predicted Y values

            ---Arguments---
            K: kernel matrix between training and testing data

            ---Returns---
            w: regression weights

        """

        if self.w is None:
            print("Error: must fit the KRR model before transforming")
        else:
        
            # Compute predicted Y values
            Yp = np.matmul(K, self.w)
        
            return Yp


class SparseKRR(object):
    """
        Performs sparsified kernel ridge regression
        
        ---Attributes---
        sigma: regularization parameter
        jitter: additional regularization scale based on the maximum eigenvalue
            of sigma*KMM + KNM.T * KNM
        
        ---Methods---
        fit: fit the sparse KRR model by computing regression weights
        transform: compute predicted Y values
        
        ---References---
        1.  M. Ceriotti, M. J. Willatt, G. Csanyi,
            'Machine Learning of Atomic-Scale Properties
            Based on Physical Principles', Handbook of Materials Modeling,
            Springer, 2018
        2.  A. J. Smola, B. Scholkopf, 'Sparse Greedy Matrix Approximation 
            for Machine Learning', Proceedings of the 17th International
            Conference on Machine Learning, 911-918, 2000
    """
    
    def __init__(self, sigma=1, jitter=1.0E-16):
        self.sigma = sigma
        self.jitter = jitter
        self.w = None
        
    def fit(self, KNM, KMM, Y):
        """
            Fits the KRR model by computing the regression weights

            ---Arguments---
            KNM: kernel between the whole dataset and the representative points
            KMM: kernel between the representative points
            Y: property values
        """
    
        # Compute max eigenvalue of regularized model
        Kreg = self.sigma*KMM + np.matmul(KNM.T, KNM)
        maxeig = np.amax(np.linalg.eigvalsh(K))

        # Use max eigenvalue as additional regularization
        Kreg += np.eye(KMM.shape[0])*maxeig*self.jitter

        YY = np.matmul(KNM.T, Y)

        # Solve KRR model
        self.w = np.linalg.solve(K, YY)
        
    def transform(self, KNM):
        """
            Computes predicted Y values

            ---Arguments---
            K: kernel matrix between training and testing data

            ---Returns---
            w: regression weights

        """

        if w is None:
            print("Error: must fit the KRR model before transforming")
        else:
            Yp = np.matmul(KNM, w)
            
            return Yp

class PCovR(object):
    """
        Performs principal covariates regression

        ---Attributes---
        alpha: tuning parameter between PCA and LR
        n_pca: number of PCA components to retain
        U: eigenvalues of G
        V: eigenvectors of G
        W: component weights
        B: regression weights

        ---Methods---
        _Y: computes the LR predicted Y
        fit_structure_space: fits the PCovR model for features > samples
        fit_feature_space: fits the PCovR model for samples > features
        transform_X: computes the reconstructed and projected X
        transform_Y: computes the projected Y
        loss: computes the components of the loss functions

        ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015

    """

    def __init__(self, alpha=0.0, n_pca=None):
        self.alpha = alpha
        self.n_pca = n_pca
        self.U = None
        self.V = None
        self.W = None
        self.B = None

    def _Y(self, X, Y):
        """
            Compute the linear regression prediction of Y

            ---Arguments---
            X: independent (predictor) variable data
            Y: dependent (response) variable data

            ---Returns---
            Yhat: linear regression prediction of Y
        """

        # Compute predicted Y
        Yhat = np.linalg.pinv(np.matmul(X.T, X))
        Yhat = np.matmul(X, Yhat)
        Yhat = np.matmul(Yhat, X)
        Yhat = np.matmul(Yhat, Y)

        return Yhat

    def fit_structure_space(self, X, Y, tiny=0.0):
        """
            Fit the PCovR model for features > samples

            ---Arguments---
            X: independent (predictor) variable data
            Y: dependent (response) variable data
            tiny: threshold for discarding small eigenvalues
        """

        # Compute LR approximation of Y
        Yhat = self._Y(X, Y)

        # Compute G matrix
        G_pca = np.matmul(X, X.T)/np.linalg.norm(X)**2
        G_lr = np.matmul(Yhat, Yhat.T)/np.linalg.norm(Y)**2
        G = self.alpha*G_pca + (1.0 - self.alpha)*G_lr

        # Compute eigendecomposition of G
        self.U, self.V = np.linalg.eigh(G)
        self.U = np.flip(self.U, axis=0)
        self.V = np.flip(self.V, axis=1)
        self.V = self.V[:, self.U > tiny]
        self.U = self.U[self.U > tiny]

        # Compute projections (component scores)
        T = np.matmul(self.V, np.diagflat(np.sqrt(self.U)))

        # Compute component weights
        W_pca = X.T/np.linalg.norm(X)**2 
        W_lr = np.linalg.pinv(np.matmul(X.T, X))
        W_lr = np.matmul(W_lr, X.T)
        W_lr = np.matmul(W_lr, Y)
        W_lr = np.matmul(W_lr, Yhat)
        self.W = self.alpha*W_pca + (1.0-self.alpha)*W_lr
        self.W = np.matmul(self.W, self.V)
        self.W = np.matmul(self.W, np.diagflat(1.0/np.sqrt(self.U)))

        # Compute regression parameters
        Py = np.matmul(np.diagflat(1.0/self.U), T.T)
        Py = np.matmul(Py, Y)

        # Compute regression weights
        self.B = np.matmul(self.W, Py)

    def fit_feature_space(self, X, Y, tiny=0.0):
        """
            Fit the PCovR model for samples > features 

            ---Arguments---
            X: independent (predictor) variable data
            Y: dependent (response) variable data
            tiny: threshold for discarding small eigenvalues
        """

        # Compute LR approximation of Y
        Yhat = self._Y(X, Y)

        # Compute covariance matrix
        C = np.matmul(X.T, X)

        # Compute eigendecomposition of the covariance
        Uc, Vc = np.linalg.eigh(C)
        Uc = np.flip(Uc, axis=0)
        Vc = np.flip(Vc, axis=1)
        Vc = Vc[:, Uc > tiny]
        Uc = Uc[Uc > tiny]

        # Compute inverse square root of the covariance
        C_inv_sqrt = np.matmul(Vc, np.diagflat(1.0/np.sqrt(Uc)))
        C_inv_sqrt = np.matmul(C_inv_sqrt, Vc.T)

        # Compute the S matrix
        S_pca = C/np.trace(C)
        S_lr = np.matmul(C_inv_sqrt, X.T)
        S_lr = np.matmul(S_lr, Yhat)
        S_lr = np.matmul(S_lr, Yhat.T)
        S_lr = np.matmul(S_lr, X)
        S_lr = np.matmul(S_lr, C_inv_sqrt)/np.linalg.norm(Y)**2

        S = self.alpha*S_pca + (1.0-self.alpha)*S_lr

        # Compute the eigendecomposition of the S matrix
        self.U, self.V = np.linalg.eigh(S)
        self.U = np.flip(self.U, axis=0)
        self.V = np.flip(self.V, axis=1)
        self.V = self.V[:, self.U > tiny]
        self.U = self.U[self.U > tiny]

        # Compute component weights
        self.W = np.matmul(C_inv_sqrt, self.V)
        self.W = np.matmul(self.W, np.diagflat(np.sqrt(self.U)))

        # Compute component scores
        T = np.matmul(X, self.W)

        # Compute regression parameters
        Py = np.matmul(np.diagflat(1.0/self.U), T.T)
        Py = np.matmul(Py, Y)

        # Compute regression weights
        self.B = np.matmul(self.W, Py)

    def transform_X(self, X):
        """
            Compute the projection and reconstruction of X

            ---Arguments---
            X: data to project and reconstruct

            ---Returns---
            T: projection of X
            Xr: reconstruction of X
        """

        if self.W is None:
            print("Error: must fit the PCovR model before transforming")
        else:
            T = np.matmul(X, self.W)*np.linalg.norm(X)
            Px = np.matmul(T.T, X)
            Xr = np.matmul(T, Px)
            
            return T, Xr

    def transform_Y(self, X):
        """
            Compute the projection (prediction) of Y

            ---Arguments---
            X: predictor data for Y

            ---Returns---
            Yp: predicted Y values
        """

        if self.B is None:
            print("Error: must fit the PCovR model before transforming")
        else:

            # Compute predicted Y
            Yp = np.matmul(X, self.B)

            return Yp

    def loss(self, X, Y):
        """
            Compute the PCA and LR loss functions

            ---Arguments---
            X: independent (predictor) data
            Y: dependent (response) data

            ---Returns---
            L_pca: PCA loss
            L_lr: LR loss
        """

        # Compute reconstructed X and predicted Y
        _, Xr = self.transform_X(X)
        Yp = self.transform_Y(X)

        # Compute separate loss terms
        L_pca = np.linalg.norm(X - Xr)**2/np.linalg.norm(X)**2
        L_lr = np.linalg.norm(Y - Yp)**2/np.linalg.norm(Y)**2

        return L_pca, L_lr

class KPCovR(object):
    """
        Performs kernel principal covariates regression

        ---Attributes---
        U: eigenvalues of G
        V: eigenvectors of G
        W: component weights
        B: regression weights

        ---Methods---
        _Y: computes the KRR prediction of Y
        fit: fits the KPCovR model
        transform_K: transforms the kernel data into KPCA space
        transform_Y: yields predicted Y values based on KRR

    """

    def __init__(self, alpha=0.0, n_kpca=None, jitter=1E-12):
        self.alpha = alpha
        self.n_kpca = n_kpca
        self.U = None
        self.V = None
        self.W = None
        self.B = None

    def _Y(self, K, Y):
        """
            Computes the KRR prediction of Y

            ---Arguments---
            K: kernel matrix
            Y: dependent (response) data

            ---Returns---
            Yhat: KRR prediction of Y
        """

        # Compute predicted Y # TODO: replace inverse with np.linalg.solve
        Yhat = np.linalg.pinv(K + np.eye(K.shape[0])*jitter)
        Yhat = np.matmul(K, Yhat)
        Yhat = np.matmul(Yhat, Y)

        return Yhat

    def fit(self, K, Y, tiny=0.0):
        """
            Fits the KPCovR model

            ---Arguments---
            K: kernel matrix
            Y: dependent (response) data
            tiny: threshold for discarding small eigenvalues
        """

        # Compute predicted Y
        Yhat = _Y(K, Y)

        # Compute G
        G_kpca = K/np.trace(K)
        G_krr = np.matmul(Yhat, Yhat.T)/np.linalg.norm(Y)**2
        G = self.alpha*G_kpca + (1.0 - self.alpha)*G_krr

        # Compute eigendecomposition of G
        self.U, self.V = np.linalg.eigh(G)
        self.U = np.flip(self.U, axis=0)
        self.V = np.flip(self.V, axis=1)
        self.V = self.V[:, self.U > tiny]
        self.U = self.U[self.U > tiny]

        # Compute component scores
        T = np.matmul(self.V, np.diagflat(np.sqrt(self.U)))

        # Compute component weights
        W_kpca = np.eye(K.shape[0])/np.trace(K)
        W_krr = np.linalg.pinv(K + np.eye(K.shape[0])*jitter)
        W_krr = np.matmul(W_krr, Yhat)

        self.W = self.alpha*W_kpca + (1.0 - self.alpha)*W_krr
        self.W = np.matmul(self.W, self.V)
        self.W = np.matmul(self.W, np.diagflat(1.0/np.sqrt(self.U)))

        # Compute regression parameters
        Py = np.matmul(np.diagflat(1.0/self.U), T.T)
        Py = np.matmul(P, Y)

        # Compute regression weights
        B = np.matmul(self.W, Py)

    def transform_K(self, K):
        """
            Transform the data into KPCA space

            ---Arguments---
            K: kernel matrix

            ---Returns---
            T: the KPCA-like projection
        """

        if self.W is None:
            print("Error: must fit the PCovR model before transforming")
        else:
            T = np.matmul(K, self.W)*np.trace(K)

            # TODO: reconstruct X
            
            return T

    def transform_Y(self, K):
        """
            Compute the predicted Y values

            ---Arguments---
            K: kernel matrix

            ---Returns---
            Yp: predicted Y values
        """

        if self.B is None:
            print("Error: must fit the PCovR model before transforming")
        else:

            # Compute predicted Y
            Yp = np.matmul(K, self.B)

            return Yp

    # TODO: loss functions

class SparseKPCovR(object):
    """
        Performs sparsified kernel principal covariates regression

        ---Attributes---
        alpha: tuning parameter
        n_kpca: number of kernel principal components to retain
        U: eigenvalues of KMM
        V: eigenvectors of KMM
        W: component weights
        B: regression weights
        phi: independent (predictor) data in feature space

        ---Methods---
        fit: fit the sparse KPCovR model
        transform_K: transform the data into KPCA space
        transform_Y: compute predicted Y values
    """

    def __init__(self, alpha=0.0, n_kpca=None, jitter=1E-12):
        self.alpha = alpha
        self.n_kpca = n_kpca
        self.U = None
        self.V = None
        self.W = None
        self.B = None
        self.phi = None

    def fit(self, KNM, KMM, Y, tiny=0.0):
        """
            Fit the sparse KPCovR model

            ---Arguments---
            KNM: kernel between all points and the subsampled points
            KMM: kernel between the subsampled points
            Y: dependent (response) variable
            tiny: threshold for discarding small eigenvalues
        """

        # Compute eigendecomposition of KMM
        self.U, self.V = np.linalg.eigh(KMM)
        self.U = np.flip(self.U, axis=0)
        self.V = np.flip(self.V, axis=1)
        self.V = self.V[:, self.U > tiny]
        self.U = self.U[self.U > tiny]

        # Compute the feature space data
        self.phi = np.matmul(KNM, self.V)
        self.phi = np.matmul(self.phi, np.diagflat(1.0/np.sqrt(self.U)))

        # Compute covariance of the feature space data
        C = np.matmul(self.phi.T, self.phi)
        Uc, Vc = np.linalg.eigh(C)
        Uc = np.flip(Uc, axis=0)
        Vc = np.flip(Vc, axis=1)
        Vc = Vc[:, Uc > tiny]
        Uc = Uc[Uc > tiny]

        # Compute inverse square root of the covariance
        C_inv_sqrt = np.matmul(Vc, np.diagflat(1.0/np.sqrt(Uc)))
        C_inv_sqrt = np.matmul(C_inv_sqrt, Vc.T)

        # Compute the LR prediction of Y, since we already have
        # the feature space representation
        Yhat = np.matmul(phi, w)

        # Compute the S matrix
        S_kpca = C/np.trace(C)
        w = np.linalg.solve(C, np.matmul(phi.T, Y))

        S_lr = np.matmul(C_inv_sqrt, phi.T)
        S_lr = np.matmul(S_lr, Yhat)
        S_lr = np.matmul(S_lr, S_lr.T)/np.linalg.norm(Y)**2

        S = self.alpha*S_kpca + (1.0 - alpha)*S_lr

        # Compute eigendecomposition of S
        Us, Vs = np.linalg.eigh(S)
        Us = np.flip(Us, axis=0)
        Vs = np.flip(Vs, axis=1)
        Vs = Vs[:, Us > tiny]
        Us = Us[Us > tiny]

        # Compute component weights
        self.W = np.matmul(C_inv_sqrt, Us)
        self.W = np.matmul(self.W, np.diagflat(np.sqrt(Us)))

        # Compute component scores
        T = np.matmul(phi, self.W)

        # Compute regression parameters
        Py = np.matmul(np.diagflat(1.0/Us))
        Py = np.matmul(Py, T.T)
        Py = np.matmul(Py, Y)

        # Compute regression weights
        self.B = np.matmul(self.W, P)

    def transform_K(self, KNM):
        """
            Transform the data into KPCA space

            ---Arguments---
            KNM: kernel between all points and the representative points
        """

        if self.W is None:
            print("Error: must fit the KPCovR model before transforming")
        else:
            # TODO: compute reconstructed X

            # Compute the KPCA-like projections
            Xp = np.matmul(KNM, self.V)
            Xp = np.matmul(Xp, np.diagflat(1.0/np.sqrt(self.U)))
            Xp = np.matmul(Xp, self.W)*np.linalg.norm(np.matmul(KNM, self.phi))

            return Xp

    def transform_Y(self, KNM):
        """
            Compute the predictions of Y

            ---Arguments---
            KNM: kernel between all points and the representative points

            ---Returns---
            Yp: predicted Y values
        """

        if self.B is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute predicted Y values
            Yp = np.matmul(KNM, self.V)
            Yp = np.matmul(Yp, np.diagflat(1.0/np.sqrt(self.U)))
            Yp = np.matmul(Yp, self.B)

            return Yp

    # TODO: loss functions
