#!/usr/bin/env python

import os
import sys
import numpy as np

class LR(object):
    """
        Performs linear regression

        ---Attributes---
        W: regression weights
        reg: regularization parameter

        ---Methods---
        fit: fit the linear regression model by computing regression weights
        transform: compute predicted Y values

        ---References---
        1.  https://en.wikipedia.org/wiki/Linear_regression
    """

    def __init__(self, reg=1.0E-15):
        self.W = None
        self.reg = reg

    def fit(self, X, Y):
        """
            Fits the linear regression model

            ---Arguments---
            X: centered, independent (predictor) variable
            Y: centered, dependent (response) variable
        """

        # Compute inverse of covariance
        XTX = np.matmul(X.T, X)

        # Compute max eigenvalue of covariance
        maxeig = np.amax(np.linalg.eigvalsh(XTX))

        # Add regularization
        XTX += np.eye(X.shape[1])*maxeig*self.reg

        XY = np.matmul(X.T, Y)

        # Compute LR solution
        self.W = np.linalg.solve(XTX, XY)

    def transform(self, X):
        """
            Computes predicted Y values

            ---Arguments---
            X: centered, independent (predictor) variable
            
            ---Returns---
            Yp: predicted Y values
        """

        if self.W is None:
            print("Must fit the LR model before transforming")
        else:

            # Compute predicted Y
            Yp = np.matmul(X, self.W)

            return Yp

class KRR(object):
    """
        Performs kernel ridge regression
        
        ---Attributes---
        reg: regularization parameter
        W: regression weights
        
        ---Methods---
        fit: fit the KRR model by computing regression weights
        transform: compute predicted Y values

        ---References---
        1.  M. Ceriotti, M. J. Willatt, G. Csanyi,
            'Machine Learning of Atomic-Scale Properties
            Based on Physical Principles', Handbook of Materials Modeling,
            Springer, 2018
    """
    
    def __init__(self, reg=1.0E-15):
        self.reg = reg
        self.W = None
        
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
        Kr = K + np.eye(K.shape[0])*maxeig*self.reg

        # Solve the model
        self.W = np.linalg.solve(Kr, Y)
        
    def transform(self, K):
        """
            Computes predicted Y values

            ---Arguments---
            K: kernel matrix between training and testing data

            ---Returns---
            Yp: predicted Y values

        """

        if self.W is None:
            print("Error: must fit the KRR model before transforming")
        else:
        
            # Compute predicted Y values
            Yp = np.matmul(K, self.W)
        
            return Yp


class SparseKRR(object):
    """
        Performs sparsified kernel ridge regression
        
        ---Attributes---
        sigma: regularization parameter
        reg: additional regularization scale based on the maximum eigenvalue
            of sigma*KMM + KNM.T * KNM
        W: regression weights
        
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
    
    def __init__(self, sigma=1, reg=1.0E-15):
        self.sigma = sigma
        self.reg = reg
        self.W = None
        
    def fit(self, KNM, KMM, Y):
        """
            Fits the KRR model by computing the regression weights

            ---Arguments---
            KNM: kernel between the whole dataset and the representative points
            KMM: kernel between the representative points
            Y: property values
        """
    
        # Compute max eigenvalue of regularized model
        K = self.sigma*KMM + np.matmul(KNM.T, KNM)
        maxeig = np.amax(np.linalg.eigvalsh(K))

        # Use max eigenvalue as additional regularization
        K += np.eye(KMM.shape[0])*maxeig*self.reg

        KY = np.matmul(KNM.T, Y)

        # Solve KRR model
        self.W = np.linalg.solve(K, KY)
        
    def transform(self, KNM):
        """
            Computes predicted Y values

            ---Arguments---
            K: kernel matrix between training and testing data

            ---Returns---
            Yp: predicted Y values

        """

        if self.W is None:
            print("Error: must fit the KRR model before transforming")
        else:
            Yp = np.matmul(KNM, self.W)
            
            return Yp

class PCovR(object):
    """
        Performs principal covariates regression

        ---Attributes---
        alpha: tuning parameter between PCA and LR
        n_pca: number of PCA components to retain
        reg: regularization parameter
        tiny: cutoff for throwing away small eigenvalues
        U: eigenvalues of G
        V: eigenvectors of G
        Pxt: projection matrix from input space (X) to latent space (T)
        Ptx: projection matrix from latent space (T) to input space (X)
        Pty: projection matrix from latent space (T) to properties (Y)

        ---Methods---
        _YW: computes the LR predicted Y and weights
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

    def __init__(self, alpha=0.0, n_pca=None, reg=1.0E-15, tiny=1.0E-15):
        self.alpha = alpha
        self.n_pca = n_pca
        self.reg = reg
        self.tiny = tiny
        self.U = None
        self.V = None
        self.Pxt = None
        self.Ptx = None
        self.Pty = None

    def _YW(self, X, Y):
        """
            Compute the linear regression prediction of Y

            ---Arguments---
            X: independent (predictor) variable data
            Y: dependent (response) variable data

            ---Returns---
            Yhat: linear regression prediction of Y
            W: linear regression weights
        """

        # Compute predicted Y with LR
        lr = LR(reg=self.reg)
        lr.fit(X, Y)
        Yhat = lr.transform(X)
        W = lr.W

        return Yhat, W

    def fit_structure_space(self, X, Y):
        """
            Fit the PCovR model for features > samples

            ---Arguments---
            X: independent (predictor) variable data
            Y: dependent (response) variable data
        """

        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))

        # Compute LR approximation of Y
        Yhat, W = self._YW(X, Y)

        # Compute G matrix
        G_pca = np.matmul(X, X.T)/np.linalg.norm(X)**2
        G_lr = np.matmul(Yhat, Yhat.T)/np.linalg.norm(Y)**2
        G = self.alpha*G_pca + (1.0 - self.alpha)*G_lr

        # Compute eigendecomposition of G
        self.U, self.V = np.linalg.eigh(G)
        self.U = np.flip(self.U, axis=0)
        self.V = np.flip(self.V, axis=1)
        self.V = self.V[:, self.U > self.tiny]
        self.U = self.U[self.U > self.tiny]

        self.V = self.V[:, 0:self.n_pca]
        self.U = self.U[0:self.n_pca]

        # Compute projection matrix Pxt
        Pxt_pca = X.T/np.linalg.norm(X)**2 
        Pxt_lr = np.matmul(W, Yhat.T)/np.linalg.norm(Y)**2
        P_scale = np.linalg.norm(X)

        self.Pxt = self.alpha*Pxt_pca + (1.0 - self.alpha)*Pxt_lr
        self.Pxt = np.matmul(self.Pxt, self.V)
        self.Pxt = np.matmul(self.Pxt, np.diagflat(1.0/np.sqrt(self.U)))
        self.Pxt *= P_scale

        P = np.matmul(np.diagflat(1.0/np.sqrt(self.U)), self.V.T)

        # Compute projection matrix Pty
        self.Pty = np.matmul(P, Y)
        self.Pty /= P_scale

        # Compute projection matrix Ptx
        self.Ptx = np.matmul(P, X)
        self.Ptx /= P_scale

    def fit_feature_space(self, X, Y):
        """
            Fit the PCovR model for samples > features 

            ---Arguments---
            X: independent (predictor) variable data
            Y: dependent (response) variable data
        """

        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))

        # Compute LR approximation of Y
        Yhat, W = self._YW(X, Y)

        # Compute covariance matrix
        C = np.matmul(X.T, X)

        # Compute eigendecomposition of the covariance
        Uc, Vc = np.linalg.eigh(C)
        Uc = np.flip(Uc, axis=0)
        Vc = np.flip(Vc, axis=1)
        Vc = Vc[:, Uc > self.tiny]
        Uc = Uc[Uc > self.tiny]

        # Compute inverse square root of the covariance
        C_inv_sqrt = np.matmul(Vc, np.diagflat(1.0/np.sqrt(Uc)))
        C_inv_sqrt = np.matmul(C_inv_sqrt, Vc.T)

        # Compute square root of the covariance
        C_sqrt = np.matmul(Vc, np.diagflat(np.sqrt(Uc)))
        C_sqrt = np.matmul(C_sqrt, Vc.T)

        # Compute the S matrix
        S_pca = C/np.trace(C)
        S_lr = np.matmul(C_sqrt, W)
        S_lr = np.matmul(S_lr, S_lr.T)/np.linalg.norm(Y)**2

        S = self.alpha*S_pca + (1.0-self.alpha)*S_lr

        # Compute the eigendecomposition of the S matrix
        self.U, self.V = np.linalg.eigh(S)
        self.U = np.flip(self.U, axis=0)
        self.V = np.flip(self.V, axis=1)
        self.V = self.V[:, self.U > self.tiny]
        self.U = self.U[self.U > self.tiny]

        self.V = self.V[:, 0:self.n_pca]
        self.U = self.U[0:self.n_pca]

        P_scale = np.linalg.norm(X)

        # Compute projection matrix Pxt
        self.Pxt = np.matmul(C_inv_sqrt, self.V)
        self.Pxt = np.matmul(self.Pxt, np.diagflat(np.sqrt(self.U)))
        self.Pxt *= P_scale

        P = np.matmul(np.diagflat(1.0/np.sqrt(self.U)), self.V.T)

        # Compute projection matrix Pty
        self.Pty = np.matmul(P, C_inv_sqrt)
        self.Pty = np.matmul(self.Pty, X.T)
        self.Pty = np.matmul(self.Pty, Y)
        self.Pty /= P_scale

        # Compute projection matrix Ptx
        self.Ptx = np.matmul(P, C_sqrt)
        self.Ptx /= P_scale

    def transform_X(self, X):
        """
            Compute the projection and reconstruction of X

            ---Arguments---
            X: data to project and reconstruct

            ---Returns---
            T: projection of X
            Xr: reconstruction of X
        """

        if self.Pxt is None:
            print("Error: must fit the PCovR model before transforming")
        else:
            T = np.matmul(X, self.Pxt)
            Xr = np.matmul(T, self.Ptx)
            
            return T, Xr

    def transform_Y(self, X):
        """
            Compute the projection (prediction) of Y

            ---Arguments---
            X: predictor data for Y

            ---Returns---
            Yp: predicted Y values
        """

        if self.Pty is None:
            print("Error: must fit the PCovR model before transforming")
        else:

            # Compute predicted Y
            Yp = np.matmul(X, self.Pxt)
            Yp = np.matmul(Yp, self.Pty)

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
        alpha: tuning parameter between KPCA and KRR
        n_kpca: number of KPCA components to retain in the latent
            space projection
        reg: regularization parameter
        tiny: threshold for discarding small eigenvalues
        U: eigenvalues of G
        V: eigenvectors of G
        Pkt: projection matrix from the kernel matrix (K) to
            the latent space (T)
        Pty: projection matrix from the latent space (T) to
            the properties (Y)
        Ptk: projection matrix from the latent space (T) to
            the kernel matrix (K)

        ---Methods---
        _YW: computes the KRR prediction of Y and weights
        fit: fits the KPCovR model
        transform_K: transforms the kernel data into KPCA space
        transform_Y: yields predicted Y values based on KRR

    """

    def __init__(self, alpha=0.0, n_kpca=None, reg=1E-15, tiny=1.0E-15):
        self.alpha = alpha
        self.n_kpca = n_kpca
        self.reg = reg
        self.tiny = tiny
        self.U = None
        self.V = None
        self.Pkt = None
        self.Pty = None
        self.Ptk = None

    def _YW(self, K, Y):
        """
            Computes the KRR prediction of Y

            ---Arguments---
            K: kernel matrix
            Y: dependent (response) data

            ---Returns---
            Yhat: KRR prediction of Y
            W: regression weights
        """

        # Compute predicted Y with KRR
        krr = KRR(reg=self.reg)
        krr.fit(K, Y)
        Yhat = krr.transform(K)
        W = krr.W

        return Yhat, W

    def fit(self, K, Y):
        """
            Fits the KPCovR model

            ---Arguments---
            K: kernel matrix
            Y: dependent (response) data
        """

        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))

        # Compute predicted Y with KRR
        Yhat, W = self._YW(K, Y)

        # Compute G
        G_kpca = K/np.trace(K)
        G_krr = np.matmul(Yhat, Yhat.T)/np.linalg.norm(Y)**2
        G = self.alpha*G_kpca + (1.0 - self.alpha)*G_krr

        # Compute eigendecomposition of G
        self.U, self.V = np.linalg.eigh(G)
        self.U = np.flip(self.U, axis=0)
        self.V = np.flip(self.V, axis=1)
        self.V = self.V[:, self.U > self.tiny]
        self.U = self.U[self.U > self.tiny]

        self.V = self.V[:, 0:self.n_kpca]
        self.U = self.U[0:self.n_kpca]

        # Compute projection matrix Pkt
        Pkt_kpca = np.eye(K.shape[0])/np.trace(K)
        Pkt_krr = np.matmul(W, Yhat.T)/np.linalg.norm(Y)**2
        P_scale = np.sqrt(np.trace(K))

        self.Pkt = self.alpha*Pkt_kpca + (1.0 - self.alpha)*Pkt_krr
        self.Pkt = np.matmul(self.Pkt, self.V)
        self.Pkt = np.matmul(self.Pkt, np.diagflat(1.0/np.sqrt(self.U)))
        self.Pkt *= P_scale

        P = np.matmul(np.diagflat(1.0/np.sqrt(self.U)), self.V.T)

        # Compute projection matrix Pty
        self.Pty = np.matmul(P, Y)
        self.Pty /= P_scale

        # Compute projection matrix Ptk
        self.Ptk = np.matmul(P, K)
        self.Ptk /= P_scale

        # TODO: fit inverse transform: take X as argument, compute self.Ptx
        # as self.Ptx = np.matmul(P, X) and then use for X reconstruction

    def transform_K(self, K):
        """
            Transform the data into KPCA space

            ---Arguments---
            K: kernel matrix

            ---Returns---
            T: the KPCA-like projection
        """

        if self.Pkt is None:
            print("Error: must fit the PCovR model before transforming")
        else:
            T = np.matmul(K, self.Pkt)
            Kr = np.matmul(T, self.Ptk)

            return T, Kr

    def transform_Y(self, K):
        """
            Compute the predicted Y values

            ---Arguments---
            K: kernel matrix

            ---Returns---
            Yp: predicted Y values
        """

        if self.Pty is None:
            print("Error: must fit the PCovR model before transforming")
        else:

            # Compute predicted Y
            Yp = np.matmul(K, self.Pkt)
            Yp = np.matmul(Yp, self.Pty)

            return Yp

    # TODO: loss functions

class SparseKPCovR(object):
    """
        Performs sparsified kernel principal covariates regression

        ---Attributes---
        alpha: tuning parameter
        n_kpca: number of kernel principal components to retain
        reg: regularization parameter
        tiny: threshold for discarding small eigenvalues
        KNM_mean: auxiliary centering for the kernel matrix
            because the centering must be done based on the
            feature space, which is approximated
        U: eigenvalues of KMM
        V: eigenvectors of KMM
        Pkt: projection matrix from the kernel matrix (K) to
            the latent space (T)
        Pty: projection matrix from the latent space (T) to
            the properties (Y)
        Ptk: projection matrix from the latent space (T) to
            the kernel matrix (K)
        phi: independent (predictor) data in feature space

        ---Methods---
        fit: fit the sparse KPCovR model
        transform_K: transform the data into KPCA space
        transform_Y: compute predicted Y values
    """

    def __init__(self, alpha=0.0, n_kpca=None, reg=1E-15, sigma=1.0, tiny=1.0E-15):
        self.alpha = alpha
        self.n_kpca = n_kpca
        self.reg = reg
        self.sigma = sigma
        self.tiny = tiny
        self.KNM_mean = None
        self.U = None
        self.V = None
        self.Pkt = None
        self.Pty = None
        self.Ptk = None

    def _YW(self, KNM, KMM, Y):
        """
            Computes the KRR prediction of Y

            ---Arguments---
            KMM: kernel matrix between representative points
            KNM: kernel matrix between input data and representative points
            Y: dependent (response) data

            ---Returns---
            Yhat: KRR prediction of Y
            W: regression weights
        """

        # Compute the predicted Y with sparse KRR
        skrr = SparseKRR(reg=self.reg, sigma=self.sigma)
        skrr.fit(KNM, KMM, Y)
        Yhat = skrr.transform(KNM)
        W = skrr.W

        return Yhat, W

    def fit(self, KNM, KMM, Y):
        """
            Fit the sparse KPCovR model

            ---Arguments---
            KNM: kernel between all points and the subsampled points
            KMM: kernel between the subsampled points
            Y: dependent (response) variable
        """

        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))

        # Compute predicted Y
        Yhat, W = self._YW(KNM, KMM, Y)

        # Compute eigendecomposition of KMM
        self.U, self.V = np.linalg.eigh(KMM)
        self.U = np.flip(self.U, axis=0)
        self.V = np.flip(self.V, axis=1)
        self.V = self.V[:, self.U > self.tiny]
        self.U = self.U[self.U > self.tiny]

        self.KNM_mean = np.mean(KNM, axis=0)

        # Change from kernel-based W to phi-based W
        W = np.matmul(self.V.T, W)
        W = np.matmul(np.diagflat(np.sqrt(self.U)), W)

        # TODO: don't truncate yet?
        self.V = self.V[:, 0:self.n_kpca]
        self.U = self.U[0:self.n_kpca]

        # Compute the feature space data
        phi = np.matmul(KNM-self.KNM_mean, self.V)
        phi = np.matmul(phi, np.diagflat(1.0/np.sqrt(self.U)))

        # Compute covariance of the feature space data
        C = np.matmul(phi.T, phi)
        Uc, Vc = np.linalg.eigh(C)
        Uc = np.flip(Uc, axis=0)
        Vc = np.flip(Vc, axis=1)
        Vc = Vc[:, Uc > self.tiny]
        Uc = Uc[Uc > self.tiny]

        # Compute inverse square root of the covariance
        C_inv_sqrt = np.matmul(Vc, np.diagflat(1.0/np.sqrt(Uc)))
        C_inv_sqrt = np.matmul(C_inv_sqrt, Vc.T)

        # Compute square root of the covariance
        C_sqrt = np.matmul(Vc, np.diagflat(np.sqrt(Uc)))
        C_sqrt = np.matmul(C_sqrt, Vc.T)

        # Compute the S matrix
        S_kpca = C/np.trace(C)

        S_lr = np.matmul(C_sqrt, W)
        S_lr = np.matmul(S_lr, S_lr.T)/np.linalg.norm(Y)**2

        S = self.alpha*S_kpca + (1.0 - self.alpha)*S_lr

        # Compute eigendecomposition of S
        Us, Vs = np.linalg.eigh(S)
        Us = np.flip(Us, axis=0)
        Vs = np.flip(Vs, axis=1)
        Vs = Vs[:, Us > self.tiny]
        Us = Us[Us > self.tiny]

        P_scale = np.sqrt(np.trace(C))

        # Compute projection matrix Pkt
        self.Pkt = np.matmul(self.V, np.diagflat(1.0/np.sqrt(self.U)))
        self.Pkt = np.matmul(self.Pkt, C_inv_sqrt)
        self.Pkt = np.matmul(self.Pkt, Vs)
        self.Pkt = np.matmul(self.Pkt, np.diagflat(np.sqrt(Us)))
        self.Pkt *= P_scale

        P = np.matmul(np.diagflat(1.0/np.sqrt(Us)), Vs.T)

        # Compute projection matrix Pty
        self.Pty = np.matmul(P, C_inv_sqrt)
        self.Pty = np.matmul(self.Pty, phi.T)
        self.Pty = np.matmul(self.Pty, Y)
        self.Pty /= P_scale

        # Compute projection matrix Ptk
        self.Ptk = np.matmul(P, C_sqrt)
        self.Ptk = np.matmul(self.Ptk, np.diagflat(np.sqrt(self.U)))
        self.Ptk = np.matmul(self.Ptk, self.V.T)
        self.Ptk /= P_scale

        # TODO: fit inverse transform: take X as argument, compute self.Ptx
        # as self.Ptx = np.matmul(P, X) and then use for X reconstruction

    def transform_K(self, KNM):
        """
            Transform the data into KPCA space

            ---Arguments---
            KNM: kernel between all points and the representative points
        """

        if self.Pkt is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute the KPCA-like projections
            T = np.matmul(KNM-self.KNM_mean, self.Pkt)
            Kr = np.matmul(T, self.Ptk)

            return T, Kr

    def transform_Y(self, KNM):
        """
            Compute the predictions of Y

            ---Arguments---
            KNM: kernel between all points and the representative points

            ---Returns---
            Yp: predicted Y values
        """

        if self.Pty is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute predicted Y values
            Yp = np.matmul(KNM-self.KNM_mean, self.Pkt)
            Yp = np.matmul(Yp, self.Pty)

            return Yp

    # TODO: loss functions
