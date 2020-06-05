#!/usr/bin/env python

import os
import sys
import numpy as np
from tools import sorted_eigh

# TODO: Remove auxiliary phi centering in sparse methods
# TODO: CovR scaling consistent with paper? Should probably include auto-scaling and auto-centering, so make different scale types an option, and make sure sparse is done correctly. Also try per-property Y (and X) scaling, which will propagate to the loss functions 
# TODO: check loss functions
# TODO: make abstract base class with fit, transform, losses

class LR(object):
    """
        Performs linear regression

        ---Attributes---
        W: regression weights
        reg: regularization parameter
        reg_type: type of regularization.
            Choices are 'scalar' (constant regularization),
            or 'max_eig' (regularization based on maximum eigenvalue)

        ---Methods---
        fit: fit the linear regression model by computing regression weights
        transform: compute predicted Y values

        ---References---
        1.  https://en.wikipedia.org/wiki/Linear_regression
    """

    def __init__(self, reg=1.0E-12, reg_type='scalar', rcond=None):
        self.reg = reg
        self.reg_type = reg_type
        self.rcond = rcond
        self.W = None

    def fit(self, X, Y):
        """
            Fits the linear regression model

            ---Arguments---
            X: centered independent (predictor) variable
            Y: centered dependent (response) variable
        """

        XTX = np.matmul(X.T, X)

        # Regularize the model
        if self.reg_type == 'max_eig':
            scale = np.amax(np.linalg.eigvalsh(XTX))
        elif self.reg_type == 'scalar':
            scale = 1.0
        else:
            print("Error: invalid reg_type. Use 'scalar' or 'max_eig'")
            return

        XTX += np.eye(XTX.shape[0])*scale*self.reg
        XY = np.matmul(X.T, Y)

        # Compute LR solution
        self.W = np.linalg.lstsq(XTX, XY, rcond=self.rcond)[0]

        # The below is another valid formulation for numpy lstsq;
        # we use the above so we can add the regularization
        # and for consistency with the kernel methods
        #self.W = np.linalg.lstsq(X, Y, rcond=self.rcond)[0]

    def transform(self, X):
        """
            Computes predicted Y values

            ---Arguments---
            X: centered independent (predictor) variable
            
            ---Returns---
            Yp: centered predicted Y values
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
        reg_type: type of regularization.
            Choices are 'scalar' (constant regularization),
            or 'max_eig' (regularization based on maximum eigenvalue)
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
    
    def __init__(self, reg=1.0E-12, reg_type='scalar', rcond=None):
        self.reg = reg
        self.reg_type = reg_type
        self.rcond = rcond
        self.W = None
        
    def fit(self, K, Y):
        """
            Fits the KRR model by computing the regression weights

            ---Arguments---
            K: centered kernel between training data
            Y: centered property values
        """

        # Regularize the model
        if self.reg_type == 'max_eig':
            scale = np.amax(np.linalg.eigvalsh(K))
        elif self.reg_type == 'scalar':
            scale = 1.0
        else:
            print("Error: invalid reg_type. Use 'scalar' or 'max_eig'")
            return

        KX = K + np.eye(K.shape[0])*scale*self.reg

        # Solve the model
        self.W = np.linalg.lstsq(KX, Y, rcond=self.rcond)[0]
        
    def transform(self, K):
        """
            Computes predicted Y values

            ---Arguments---
            K: centered kernel matrix between training and testing data

            ---Returns---
            Yp: centered predicted Y values

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
        reg: additional regularization
        reg_type: type of regularization.
            Choices are 'scalar' (constant regularization),
            or 'max_eig' (regularization based on maximum eigenvalue)
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
    
    def __init__(self, sigma=1.0, reg=1.0E-12, reg_type='scalar', rcond=None):
        self.sigma = sigma
        self.reg = reg
        self.reg_type = reg_type
        self.rcond = rcond
        self.W = None
        
    def fit(self, KNM, KMM, Y):
        """
            Fits the KRR model by computing the regression weights

            ---Arguments---
            KNM: centered kernel between the whole dataset 
                and the representative points
            KMM: centered kernel between the representative points
            Y: centered property values
        """

        KX = self.sigma*KMM + np.matmul(KNM.T, KNM)
    
        # Regularize the sparse kernel model
        if self.reg_type == 'max_eig':
            scale = np.amax(np.linalg.eigvalsh(KX))
        elif self.reg_type == 'scalar':
            scale = 1.0
        else:
            print("Error: invalid reg_type. Use 'scalar' or 'max_eig'")
            return

        KX += np.eye(KMM.shape[0])*scale*self.reg
        KY = np.matmul(KNM.T, Y)

        # Solve KRR model
        self.W = np.linalg.lstsq(KX, KY, rcond=self.rcond)[0]
        
    def transform(self, KNM):
        """
            Computes predicted Y values

            ---Arguments---
            K: centered kernel matrix between training and testing data

            ---Returns---
            Yp: centered predicted Y values

        """

        if self.W is None:
            print("Error: must fit the KRR model before transforming")
        else:
            Yp = np.matmul(KNM, self.W)
            
            return Yp

class IterativeSparseKRR(object):
    """
        Performs sparsified kernel ridge regression. Example usage:

        KMM = build_kernel(Xm, Xm)
        iskrr = IterativeSparseKRR()
        iskrr.initialize_fit(KMM)
        for i in batches:
            KNMi = build_kernel(Xi, Xm)
            iskrr.fit_batch(KNMi, Yi)
        iskrr.finalize_fit()
        for i in batches:
            KNMi = build_kernel(Xi, Xm)
            iskrr.transform(KNMi)

        ---Attributes---
        sigma: regularization parameter
        reg: additional regularization
        reg_type: type of regularization.
            Choices are 'scalar' (constant regularization),
            or 'max_eig' (regularization based on maximum eigenvalue)
        W: regression weights
        KMM: centered kernel between representative points
        KY: product of the KNM kernel and the properties Y
        
        ---Methods---
        initialize_fit: initialize the fit of the sparse KRR
            (i.e., store KMM)
        fit_batch: fit a batch of training data
        finalize_fit: finalize the KRR fitting
            (i.e., compute regression weights)
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
    
    def __init__(self, sigma=1, reg=1.0E-12, reg_type='scalar', rcond=None):
        self.sigma = sigma
        self.reg = reg
        self.reg_type = reg_type
        self.rcond = rcond
        self.W = None
        self.KX = None
        self.KY = None
        self.y_dim = None

    def initialize_fit(self, KMM, y_dim=1):
        """
            Initialize the KRR fitting by computing the
            eigendecomposition of KMM

            ---Arguments---
            KMM: centered kernel between the representative points
            y_dim: number of properties
        """

        # Check for valid Y dimension
        if y_dim < 1:
            print("Y dimension must be at least 1")
            return

        # Initialize arrays
        self.y_dim = y_dim
        self.KX = self.sigma*KMM
        self.KY = np.zeros((KMM.shape[0], self.y_dim)) 
        
    def fit_batch(self, KNM, Y):
        """
            Fits the KRR model by computing the regression weights

            ---Arguments---
            KNM: centered kernel between the whole dataset "
                "and the representative points
            Y: centered property values
        """

        if self.KX is None:
            print("Error: must initialize the fit before fitting the batch")
            return

        # Turn scalar into 2D array
        if not isinstance(Y, np.ndarray):
            Y = np.array([[Y]])

        # Reshape 1D kernel
        if KNM.ndim < 2:
            KNM = np.reshape(KNM, (1, -1))

        # Reshape 1D properties
        if Y.ndim < 2:
            Y = np.reshape(Y, (-1, self.y_dim))

        # Increment KX and KY
        self.KX += np.matmul(KNM.T, KNM)
        self.KY += np.matmul(KNM.T, Y)
    
    def finalize_fit(self):
        """
            Finalize the iterative fitting of the sparse KRR model
            by computing regression weights
        """

        # Regularize the model
        if self.reg_type == 'max_eig':
            scale = np.amax(np.linalg.eigvalsh(KX))
        elif self.reg_type == 'scalar':
            scale = 1.0
        else:
            print("Error: invalid reg_type. Use 'scalar' or 'max_eig'")
            return

        self.KX += np.eye(self.KX.shape[0])*scale*self.reg

        # Solve KRR model
        self.W = np.linalg.lstsq(self.KX, self.KY, rcond=self.rcond)[0]
        
    def transform(self, KNM):
        """
            Computes predicted Y values

            ---Arguments---
            KNM: centered kernel matrix between training and testing data

            ---Returns---
            Yp: centered predicted Y values

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
        reg_type: type of regularization.
            Choices are 'scalar' (constant regularization),
            or 'max_eig' (regularization based on maximum eigenvalue)
        tiny: cutoff for throwing away small eigenvalues
        U: eigenvalues of G
        V: eigenvectors of G
        Pxt: projection matrix from input space (X) to latent space (T)
        Ptx: projection matrix from latent space (T) to input space (X)
        Pty: projection matrix from latent space (T) to properties (Y)
        P_scale: scaling for projection matrices
        Y_norm: scaling for the LR term of G

        ---Methods---
        _YW: computes the LR predicted Y and weights
        fit_structure_space: fits the PCovR model for features > samples
        fit_feature_space: fits the PCovR model for samples > features
        transform_X: computes the projected X
        inverse_transform_X: computes the reconstructed X
        transform_Y: computes the projected Y
        regression_loss: computes the linear regression loss
        projection_loss: computes the projection loss
        gram_loss: computes the Gram loss

        ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015

    """

    def __init__(self, alpha=0.0, n_pca=None, reg=1.0E-12, 
            reg_type='scalar', tiny=1.0E-15, rcond=None):
        self.alpha = alpha
        self.n_pca = n_pca
        self.reg = reg
        self.reg_type = reg_type
        self.tiny = tiny
        self.rcond = rcond
        self.U = None
        self.V = None
        self.Pxt = None
        self.Ptx = None
        self.Pty = None
        self.P_scale = None
        self.Y_norm = None

    def _YW(self, X, Y):
        """
            Compute the linear regression prediction of Y

            ---Arguments---
            X: centered independent (predictor) variable data
            Y: centered dependent (response) variable data

            ---Returns---
            Y_hat: centered linear regression prediction of Y
            W: linear regression weights
        """

        # Compute predicted Y with LR
        lr = LR(reg=self.reg, reg_type=self.reg_type, rcond=self.rcond)
        lr.fit(X, Y)
        Y_hat = lr.transform(X)
        W = lr.W

        return Y_hat, W

    def fit_structure_space(self, X, Y):
        """
            Fit the PCovR model for features > samples

            ---Arguments---
            X: centered independent (predictor) variable data
            Y: centered dependent (response) variable data
        """

        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))

        # Compute LR approximation of Y
        Y_hat, W = self._YW(X, Y)

        # Set scaling and norms
        self.Y_norm = np.linalg.norm(Y)
        self.P_scale = np.linalg.norm(X)

        # Compute G matrix
        G_pca = np.matmul(X, X.T)/self.P_scale**2
        G_lr = np.matmul(Y_hat, Y_hat.T)/self.Y_norm**2
        G = self.alpha*G_pca + (1.0 - self.alpha)*G_lr

        # Compute eigendecomposition of G
        self.U, self.V = sorted_eigh(G, tiny=self.tiny)

        # Truncate the projections
        self.V = self.V[:, 0:self.n_pca]
        self.U = self.U[0:self.n_pca]

        # Compute projection matrix Pxt
        Pxt_pca = X.T/self.P_scale**2 
        Pxt_lr = np.matmul(W, Y_hat.T)/self.Y_norm**2

        self.Pxt = self.alpha*Pxt_pca + (1.0 - self.alpha)*Pxt_lr
        self.Pxt = np.matmul(self.Pxt, self.V)
        self.Pxt = np.matmul(self.Pxt, np.diagflat(1.0/np.sqrt(self.U)))
        self.Pxt *= self.P_scale

        P = np.matmul(np.diagflat(1.0/np.sqrt(self.U)), self.V.T)

        # Compute projection matrix Pty
        self.Pty = np.matmul(P, Y)
        self.Pty /= self.P_scale

        # Compute projection matrix Ptx
        self.Ptx = np.matmul(P, X)
        self.Ptx /= self.P_scale

    def fit_feature_space(self, X, Y):
        """
            Fit the PCovR model for samples > features 

            ---Arguments---
            X: centered independent (predictor) variable data
            Y: centered dependent (response) variable data
        """

        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))

        # Compute LR approximation of Y
        Y_hat, W = self._YW(X, Y)

        # Compute covariance matrix
        C = np.matmul(X.T, X)

        # Compute eigendecomposition of the covariance
        Uc, Vc = sorted_eigh(C, tiny=self.tiny)

        # Set scaling and norms
        self.P_scale = np.linalg.norm(X)
        self.Y_norm = np.linalg.norm(Y)

        # Compute inverse square root of the covariance
        C_inv_sqrt = np.matmul(Vc, np.diagflat(1.0/np.sqrt(Uc)))
        C_inv_sqrt = np.matmul(C_inv_sqrt, Vc.T)

        # Compute square root of the covariance
        C_sqrt = np.matmul(Vc, np.diagflat(np.sqrt(Uc)))
        C_sqrt = np.matmul(C_sqrt, Vc.T)

        # Compute the S matrix
        S_pca = C/self.P_scale**2
        S_lr = np.matmul(C_sqrt, W)
        S_lr = np.matmul(S_lr, S_lr.T)/self.Y_norm**2

        S = self.alpha*S_pca + (1.0-self.alpha)*S_lr

        # Compute the eigendecomposition of the S matrix
        self.U, self.V = sorted_eigh(S, tiny=self.tiny)

        # Truncate the projections
        self.V = self.V[:, 0:self.n_pca]
        self.U = self.U[0:self.n_pca]

        # Compute projection matrix Pxt
        self.Pxt = np.matmul(C_inv_sqrt, self.V)
        self.Pxt = np.matmul(self.Pxt, np.diagflat(np.sqrt(self.U)))
        self.Pxt *= self.P_scale

        P = np.matmul(np.diagflat(1.0/np.sqrt(self.U)), self.V.T)

        # Compute projection matrix Pty
        self.Pty = np.matmul(P, C_inv_sqrt)
        self.Pty = np.matmul(self.Pty, X.T)
        self.Pty = np.matmul(self.Pty, Y)
        self.Pty /= self.P_scale

        # Compute projection matrix Ptx
        self.Ptx = np.matmul(P, C_sqrt)
        self.Ptx /= self.P_scale

    def transform_X(self, X):
        """
            Compute the projection of X

            ---Arguments---
            X: centered data to project

            ---Returns---
            T: centered projection of X
        """

        if self.Pxt is None:
            print("Error: must fit the PCovR model before transforming")
        else:
            T = np.matmul(X, self.Pxt)
            
            return T

    def inverse_transform_X(self, X):
        """
            Compute the reconstruction of X

            ---Arguments---
            X: centered data to reconstruct

            ---Returns---
            Xr: centered reconstruction of X
        """

        if self.Ptx is None:
            print("Error: must fit the PCovR model before transforming")
        else:
            T = self.transform_X(X)
            Xr = np.matmul(T, self.Ptx)

            return Xr

    def transform_Y(self, X):
        """
            Compute the projection (prediction) of Y

            ---Arguments---
            X: centered predictor data for Y

            ---Returns---
            Yp: centered predicted Y values
        """

        if self.Pty is None:
            print("Error: must fit the PCovR model before transforming")
        else:

            # Compute predicted Y
            T = self.transform_X(X)
            Yp = np.matmul(T, self.Pty)

            return Yp

    def regression_loss(self, X, Y):
        """
            Compute the (linear) regression loss

            ---Arguments---
            X: centered independent (predictor) data
            Y: centered dependent (response) data

            ---Returns---
            regression_loss: regression loss
        """

        # Compute loss
        Yp = self.transform_Y(X)
        regression_loss = np.linalg.norm(Y - Yp)**2/self.Y_norm**2

        return regression_loss

    def projection_loss(self, X):
        """
            Compute the projection loss

            ---Arguments---
            X: centered independent (predictor) data

            ---Returns---
            projection_loss: projection loss
        """

        # Compute loss
        Xr = self.inverse_transform_X(X)
        projection_loss = np.linalg.norm(X - Xr)**2/self.P_scale**2

        return projection_loss

    def gram_loss(self, X):
        """
            Compute the Gram loss

            ---Arguments---
            X: centered independent (predictor) data

            ---Returns---
            gram_loss: gram loss
        """

        # Compute loss
        T = self.transform_X(X)
        gram_loss = np.linalg.norm(np.matmul(X, X.T) \
                - np.matmul(T, T.T))**2 / self.P_scale**4

        return gram_loss

class KPCovR(object):
    """
        Performs kernel principal covariates regression

        ---Attributes---
        alpha: tuning parameter between KPCA and KRR
        n_kpca: number of KPCA components to retain in the latent
            space projection
        reg: regularization parameter
        reg_type: type of regularization.
            Choices are 'scalar' (constant regularization),
            or 'max_eig' (regularization based on maximum eigenvalue)
        tiny: threshold for discarding small eigenvalues
        U: eigenvalues of G
        V: eigenvectors of G
        Pkt: projection matrix from the kernel matrix (K) to
            the latent space (T)
        Pty: projection matrix from the latent space (T) to
            the properties (Y)
        Ptk: projection matrix from the latent space (T) to
            the kernel matrix (K)
        P_scale: scaling for the projection matrices
        Y_norm: scaling for the LR term of G

        ---Methods---
        _YW: computes the KRR prediction of Y and weights
        fit: fits the KPCovR model
        transform_K: transforms the kernel data into the latent space
        inverse_transform_K: computes the reconstructed kernel
        inverse_transform_X: computes the reconstructed original data
            (if provided during the fit)
        transform_Y: yields predicted Y values based on KRR
        regression_loss: compute the kernel ridge regression loss
        projection_loss: compute the projection loss
        gram_loss: compute the Gram loss

    """

    def __init__(self, alpha=0.0, n_kpca=None, reg=1E-12, 
            reg_type='scalar', tiny=1.0E-15, rcond=None):
        self.alpha = alpha
        self.n_kpca = n_kpca
        self.reg = reg
        self.reg_type = reg_type
        self.tiny = tiny
        self.rcond = rcond
        self.U = None
        self.V = None
        self.Pkt = None
        self.Pty = None
        self.Ptk = None
        self.Ptx = None
        self.P_scale = None
        self.Y_norm = None

    def _YW(self, K, Y):
        """
            Computes the KRR prediction of Y

            ---Arguments---
            K: centered kernel matrix
            Y: centered dependent (response) data

            ---Returns---
            Y_hat: centered KRR prediction of Y
            W: regression weights
        """

        # Compute predicted Y with KRR
        krr = KRR(reg=self.reg, reg_type=self.reg_type, rcond=self.rcond)
        krr.fit(K, Y)
        Y_hat = krr.transform(K)
        W = krr.W

        return Y_hat, W

    def fit(self, K, Y, X=None):
        """
            Fits the KPCovR model

            ---Arguments---
            K: centered kernel matrix
            Y: centered dependent (response) data
            X: centered original independent (predictor) data
        """

        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))

        # Compute predicted Y with KRR
        Y_hat, W = self._YW(K, Y)

        # Set scaling and norms
        self.P_scale = np.sqrt(np.trace(K))
        self.Y_norm = np.linalg.norm(Y)

        # Compute G
        G_kpca = K/self.P_scale**2
        G_krr = np.matmul(Y_hat, Y_hat.T)/self.Y_norm**2
        G = self.alpha*G_kpca + (1.0 - self.alpha)*G_krr

        # Compute eigendecomposition of G
        self.U, self.V = sorted_eigh(G, tiny=self.tiny)

        # Truncate the projections
        self.V = self.V[:, 0:self.n_kpca]
        self.U = self.U[0:self.n_kpca]

        # Compute projection matrix Pkt
        Pkt_kpca = np.eye(K.shape[0])/self.P_scale**2
        Pkt_krr = np.matmul(W, Y_hat.T)/self.Y_norm**2

        self.Pkt = self.alpha*Pkt_kpca + (1.0 - self.alpha)*Pkt_krr
        self.Pkt = np.matmul(self.Pkt, self.V)
        self.Pkt = np.matmul(self.Pkt, np.diagflat(1.0/np.sqrt(self.U)))
        self.Pkt *= self.P_scale

        P = np.matmul(np.diagflat(1.0/np.sqrt(self.U)), self.V.T)

        # Compute projection matrix Pty
        self.Pty = np.matmul(P, Y)
        self.Pty /= self.P_scale

        # Compute projection matrix Ptk
        self.Ptk = np.matmul(P, K)
        self.Ptk /= self.P_scale

        # Compute the projection matrix Ptx
        if X is not None:
            self.Ptx = np.matmul(P, X)
            self.Ptx /= self.P_scale

    def transform_K(self, K):
        """
            Transform the data into KPCA space

            ---Arguments---
            K: centered kernel matrix

            ---Returns---
            T: centered KPCA-like projection
        """

        if self.Pkt is None:
            print("Error: must fit the PCovR model before transforming")
        else:
            T = np.matmul(K, self.Pkt)

            return T

    def inverse_transform_K(self, K):
        """
            Compute the reconstruction of the kernel

            ---Arguments---
            K: centered kernel matrix

            ---Returns---
            Kr: the centered reconstructed kernel
        """

        if self.Ptk is None:
            print("Error: must fit the PCovR model before transforming")
        else:
            T = self.transform_K(K) 
            Kr = np.matmul(T, self.Ptk)

            return Kr

    def inverse_transform_X(self, K):
        """
            Compute the reconstruction of the original X data

            ---Arguments---
            K: centered kernel matrix

            ---Returns---
            Xr: centered reconstructed X data
        """

        if self.Ptx is None:
            print("Error: must provide X data during the PCovR fit "
                    "before transforming")
        else:
            T = self.transform_K(K)
            Xr = np.matmul(T, self.Ptx)

            return Xr

    def transform_Y(self, K):
        """
            Compute the predicted Y values

            ---Arguments---
            K: centered kernel matrix

            ---Returns---
            Yp: centered predicted Y values
        """

        if self.Pty is None:
            print("Error: must fit the PCovR model before transforming")
        else:

            # Compute predicted Y
            T = self.transform_K(K)
            Yp = np.matmul(T, self.Pty)

            return Yp

    def regression_loss(self, K, Y):
        """
            Compute the (kernel) regression loss

            ---Arguments---
            K: centered kernel
            Y: centered dependent (response) data

            ---Returns---
            regression_loss: regression loss
        """

        # Compute loss
        Yp = self.transform_Y(K)
        regression_loss = np.linalg.norm(Y - Yp)**2/self.Y_norm**2

        return regression_loss

    def projection_loss(self, K, K_bridge=None, K_ref=None):
        """
            Compute the projection loss

            ---Arguments---
            X: centered independent (predictor) data

            ---Returns---
            projection_loss: projection loss
        """

        # TODO: special formulation
        # Compute loss
        if K_bridge is not None and K_ref is not None:
            pass
        else:
            pass

        projection_loss = None

        return projection_loss

    def gram_loss(self, K):
        """
            Compute the Gram loss

            ---Arguments---
            K: centered kernel

            ---Returns---
            gram_loss: Gram loss
        """

        # Compute loss
        T = self.transform_K(K)
        gram_loss = np.linalg.norm(K - np.matmul(T, T.T))**2 / self.P_scale**4

        return gram_loss

class SparseKPCovR(object):
    """
        Performs sparsified kernel principal covariates regression

        ---Attributes---
        alpha: tuning parameter
        n_kpca: number of kernel principal components to retain
        reg: regularization parameter
        reg_type: type of regularization.
            Choices are 'scalar' (constant regularization),
            or 'max_eig' (regularization based on maximum eigenvalue)
        tiny: threshold for discarding small eigenvalues
        T_mean: auxiliary centering for the kernel matrix
            because the centering must be done based on the
            feature space, which is approximated
        Um: eigenvalues of KMM
        Vm: eigenvectors of KMM
        Uc: eigenvalues of Phi.T x Phi
        Vc: eigenvectors of Phi.T x Phi
        U: eigenvalues of S
        V: eigenvectors of S
        Pkt: projection matrix from the kernel matrix (K) to
            the latent space (T)
        Pty: projection matrix from the latent space (T) to
            the properties (Y)
        Ptk: projection matrix from the latent space (T) to
            the kernel matrix (K)
        P_scale: scaling for projection matrices
        Y_norm: scaling for the LR term of G

        ---Methods---
        fit: fit the sparse KPCovR model
        transform_K: transform the data into KPCA space
        inverse_transform_K: computes the reconstructedkernel matrix
        inverse_transform_X: computes the reconstructed original input data
            (if provided during the fit)
        transform_Y: computes predicted Y values
        regression_loss: computes the sparse kernel ridge regression loss
        projection_loss: computes the projection loss
        gram_loss: computes the Gram loss
    """

    def __init__(self, alpha=0.0, n_kpca=None, sigma=1.0, 
            reg=1.0E-12, reg_type='scalar', tiny=1.0E-15, rcond=None):
        self.alpha = alpha
        self.n_kpca = n_kpca
        self.reg = reg
        self.reg_type = reg_type
        self.sigma = sigma
        self.tiny = tiny
        self.rcond = rcond
        self.T_mean = None
        self.Um = None
        self.Vm = None
        self.Uc = None
        self.Vc = None
        self.U = None
        self.V = None
        self.Pkt = None
        self.Pty = None
        self.Ptk = None
        self.Ptx = None
        self.P_scale = None
        self.Y_norm = None

    def _YW(self, KNM, KMM, Y):
        """
            Computes the KRR prediction of Y

            ---Arguments---
            KMM: centered kernel matrix between representative points
            KNM: centered kernel matrix between input data and representative points
            Y: centered dependent (response) data

            ---Returns---
            Y_hat: centered KRR prediction of Y
            W: regression weights
        """

        # Compute the predicted Y with sparse KRR
        # NOTE: If centered kernels not used, may need to 
        # do instead LR in the centered feature space (i.e., with phi)
        # and KPCA part is based on phi centering as well anyway
        skrr = SparseKRR(sigma=self.sigma, reg=self.reg, 
                reg_type=self.reg_type, rcond=self.rcond)
        skrr.fit(KNM, KMM, Y)
        Y_hat = skrr.transform(KNM)
        W = skrr.W

        return Y_hat, W

    def fit(self, KNM, KMM, Y, X=None):
        """
            Fit the sparse KPCovR model

            ---Arguments---
            KNM: centered kernel between all points and the subsampled points
            KMM: centered kernel between the subsampled points
            Y: centered dependent (response) variable
        """

        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))

        # Compute predicted Y
        Y_hat, W = self._YW(KNM, KMM, Y)

        # Compute eigendecomposition of KMM
        self.Um, self.Vm = sorted_eigh(KMM, tiny=self.tiny)

        # Change from kernel-based W to phi-based W
        W = np.matmul(self.Vm.T, W)
        W = np.matmul(np.diagflat(np.sqrt(self.Um)), W)

        # Compute the feature space data
        phi = np.matmul(KNM, self.Vm)
        phi = np.matmul(phi, np.diagflat(1.0/np.sqrt(self.Um)))

        # Auxiliary centering of phi
        # since we are working with an approximate feature space
        phi_mean = np.mean(phi, axis=0)
        phi -= phi_mean

        # Compute covariance of the feature space data
        C = np.matmul(phi.T, phi)
        self.Uc, self.Vc = sorted_eigh(C, tiny=self.tiny)

        # Set scaling and norms
        self.Y_norm = np.linalg.norm(Y)
        self.P_scale = np.sqrt(np.trace(C))

        # Compute inverse square root of the covariance
        C_inv_sqrt = np.matmul(self.Vc, np.diagflat(1.0/np.sqrt(self.Uc)))
        C_inv_sqrt = np.matmul(C_inv_sqrt, self.Vc.T)

        # Compute square root of the covariance
        C_sqrt = np.matmul(self.Vc, np.diagflat(np.sqrt(self.Uc)))
        C_sqrt = np.matmul(C_sqrt, self.Vc.T)

        # Compute the S matrix
        S_kpca = C/self.P_scale**2
        S_lr = np.matmul(C_sqrt, W)
        S_lr = np.matmul(S_lr, S_lr.T)/self.Y_norm**2

        S = self.alpha*S_kpca + (1.0 - self.alpha)*S_lr

        # Compute eigendecomposition of S
        self.U, self.V = sorted_eigh(S, tiny=self.tiny)

        # Truncate the projections
        self.U = self.U[0:self.n_kpca]
        self.V = self.V[:, 0:self.n_kpca]

        # Define some matrices that will be re-used
        P = np.matmul(np.diagflat(1.0/np.sqrt(self.U)), self.V.T)
        PP = np.matmul(C_inv_sqrt, self.V)
        PP = np.matmul(PP, np.diagflat(np.sqrt(self.U)))

        # Compute and store mean of the projections
        self.T_mean = np.matmul(phi_mean, PP)
        self.T_mean *= self.P_scale 

        # Compute projection matrix Pkt
        self.Pkt = np.matmul(self.Vm, np.diagflat(1.0/np.sqrt(self.Um)))
        self.Pkt = np.matmul(self.Pkt, PP)
        self.Pkt *= self.P_scale

        # Compute projection matrix Pty
        self.Pty = np.matmul(P, C_inv_sqrt)
        self.Pty = np.matmul(self.Pty, phi.T)
        self.Pty = np.matmul(self.Pty, Y)
        self.Pty /= self.P_scale

        # Compute projection matrix Ptk
        self.Ptk = np.matmul(P, C_sqrt)
        self.Ptk = np.matmul(self.Ptk, np.diagflat(np.sqrt(self.Um)))
        self.Ptk = np.matmul(self.Ptk, self.Vm.T)
        self.Ptk /= self.P_scale

        # Compute projection matrix Ptx
        if X is not None:
            self.Ptx = np.matmul(P, C_inv_sqrt)
            self.Ptx = np.matmul(self.Ptx, phi.T)
            self.Ptx = np.matmul(self.Ptx, X)
            self.Ptx /= self.P_scale

    def transform_K(self, KNM):
        """
            Transform the data into KPCA space

            ---Arguments---
            KNM: centered kernel between all points and the representative points

            ---Returns---
            T: centered transformed data
        """

        if self.Pkt is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute the KPCA-like projections
            T = np.matmul(KNM, self.Pkt) - self.T_mean

            return T

    def inverse_transform_K(self, KNM):
        """
            Compute the reconstruction of the kernel matrix

            ---Arguments---
            KNM: centered kernel between all points and the representative points

            ---Returns---
            Kr: centered reconstructed kernel matrix
        """

        if self.Ptk is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute the KPCA-like projections
            T = self.transform_K(KNM)
            Kr = np.matmul(T + self.T_mean, self.Ptk)

            return Kr

    def inverse_transform_X(self, KNM):
        """
            Compute the reconstruction of the X data

            ---Arguments---
            KNM: centered kernel between all points and the representative points

            ---Returns---
            Xr: centered reconstructed X data
        """

        if self.Ptx is None:
            print("Error: must provide X data during the KPCovR fit "
                    "before transforming")
        else:

            # Compute the KPCA-like projections
            T = self.transform_K(KNM)

            # NOTE: why does adding T_mean not work?
            # perhaps because we need X centered in feature space,
            # not the RKHS feature space
            Xr = np.matmul(T, self.Ptx)

            return Xr

    def transform_Y(self, KNM):
        """
            Compute the predictions of Y

            ---Arguments---
            KNM: centered kernel between all points and the representative points

            ---Returns---
            Yp: centered predicted Y values
        """

        if self.Pty is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute predicted Y values
            T = self.transform_K(KNM)
            Yp = np.matmul(T + self.T_mean, self.Pty)

            return Yp

    def regression_loss(self, KNM, Y):
        """
            Compute the (kernel) regression loss

            ---Arguments---
            KNM: centered kernel
            Y: centered dependent (response) data

            ---Returns---
            regression_loss: regression loss
        """

        # Compute loss
        Yp = self.transform_Y(KNM)
        regression_loss = np.linalg.norm(Y - Yp)**2/self.Y_norm**2

        return regression_loss

    def projection_loss(self, KNM, KMM):
        """
            Compute the projection loss

            ---Arguments---
            K: centered kernel

            ---Returns---
            projection_loss: projection loss
        """

        # TODO: special formulation
        # Compute loss
        projection_loss = None

        return projection_loss

    def gram_loss(self, KNM, KMM, KNM_ref=None):
        """
            Compute the Gram loss

            ---Arguments---
            KNM: centered kernel between data and representative points
            KMM: centered kernel between representative points
            KNM_ref: centered reference kernel between the (training) points
                and the reference points

            ---Returns---
            gram_loss: Gram loss
        """

        T = self.transform_K(KNM)
        K = np.matmul(KNM, np.linalg.inv(KMM))

        # Compute Nystrom approximation to the kernel
        if KNM_ref is not None:
            K = np.matmul(K, KNM_ref.T)
        else:
            K = np.matmul(K, KNM.T)

        # Compute loss
        gram_loss = np.linalg.norm(K - np.matmul(T, T.T))**2 / self.P_scale**4

        return gram_loss

class IterativeSparseKPCovR(object):
    """
        Performs sparsified kernel principal covariates regression
        with iterative fitting

        ---Attributes---
        alpha: tuning parameter
        n_kpca: number of kernel principal components to retain
        reg: regularization parameter
        reg_type: type of regularization.
            Choices are 'scalar' (constant regularization),
            or 'max_eig' (regularization based on maximum eigenvalue)
        tiny: threshold for discarding small eigenvalues
        T_mean: auxiliary centering for the kernel matrix
        phi_mean: mean of RKHS features
        C: covariance (Phi.T x Phi)
            because the centering must be done based on the
            feature space, which is approximated
        Um: eigenvalues of KMM
        Vm: eigenvectors of KMM
        Uc: eigenvalues of Phi.T x Phi
        Vc: eigenvectors of Phi.T x Phi
        U: eigenvalues of S
        V: eigenvectors of S
        Pkt: projection matrix from the kernel matrix (K) to
            the latent space (T)
        Pty: projection matrix from the latent space (T) to
            the properties (Y)
        Ptk: projection matrix from the latent space (T) to
            the kernel matrix (K)
        P_scale: scaling for projection matrices
        Y_norm: scaling for the LR term of G
        iskrr: incremental sparse KRR object

        ---Methods---
        initialize_fit: initialize the fitting for the sparse KPCovR model
        fit_batch: fit a batch of data
        finalize_fit: finalize the fitting procedure
        transform_K: transform the data into KPCA space
        inverse_transform_K: computes the reconstructed kernel matrix
        inverse_transform_X: computes the reconstructed original input data
            (if provided during the fit)
        transform_Y: computes predicted Y values
        regression_loss: computes the sparse kernel ridge regression loss
        projection_loss: computes the projection loss
        gram_loss: computes the Gram loss
    """

    def __init__(self, alpha=0.0, n_kpca=None, sigma=1.0, reg=1.0E-12, 
            reg_type='scalar', tiny=1.0E-15, rcond=None):
        self.alpha = alpha
        self.n_kpca = n_kpca
        self.reg = reg
        self.reg_type = reg_type
        self.sigma = sigma
        self.tiny = tiny
        self.rcond = rcond
        self.T_mean = None
        self.C = None
        self.Um = None
        self.Vm = None
        self.Uc = None
        self.Vc = None
        self.U = None
        self.V = None
        self.Pkt = None
        self.Pty = None
        self.Ptk = None
        self.Ptx = None
        self.P_scale = None
        self.Y_norm = None
        self.iskrr = None

    def initialize_fit(self, KMM, y_dim=1):
        """
            Computes the eigendecomposition of the
            kernel matrix between the representative points

            ---Arguments---
            KMM: centered kernel between the representative points
            y_dim: number of properties
        """

        # Compute eigendecomposition of KMM
        self.Um, self.Vm = sorted_eigh(KMM, tiny=self.tiny)

        # Set shape of T_mean and C according ot the
        # number of nonzero eigenvalues
        self.C = np.zeros((self.Um.size, self.Um.size))
        self.T_mean = np.zeros(self.Um.size)
        self.n_samples = 0
        self.Y_norm = 0

        # Initialize the iterative sparse KRR
        self.iskrr = IterativeSparseKRR(sigma=self.sigma, reg=self.reg, 
                reg_type=self.reg_type, rcond=self.rcond)

        self.iskrr.initialize_fit(KMM, y_dim=y_dim)

    def fit_batch(self, KNM, Y):
        """
            Fits a batch of the sparse KPCovR model

            ---Arguments---
            KNM: centered kernel between all points and the subsampled points
            KMM: centered kernel between the subsampled points
            Y: centered dependent (response) variable
        """

        # Iterative fit of the ISKRR
        self.iskrr.fit_batch(KNM, Y)

        # Iterative norm computation
        self.Y_norm += np.sum(Y**2)

        # Compute the feature space data
        phi = np.matmul(KNM, self.Vm)
        phi = np.matmul(phi, np.diagflat(1.0/np.sqrt(self.Um)))

        # Auxiliary centering of phi
        # since we are working with an approximate feature space
        old_mean = self.T_mean
        self.n_samples += phi.shape[0]
        self.T_mean = old_mean + np.sum(phi-old_mean, axis=0)/self.n_samples

        # Compute the covariance of the approximate RKHS
        self.C += np.matmul((phi-self.T_mean).T, phi-old_mean)

        # TODO: incremental fit on X (if provided) for inverse transformation
        # Do with separate functions like IterativeSparseKPCA

    def finalize_fit(self, X=None):

        # Set scaling and norms
        self.P_scale = np.sqrt(np.trace(self.C))
        self.Y_norm = np.sqrt(self.Y_norm)

        # Extract weights from the ISKRR
        self.iskrr.finalize_fit()
        W = self.iskrr.W

        # Change from kernel-based W to phi-based W
        W = np.matmul(self.Vm.T, W)
        W = np.matmul(np.diagflat(np.sqrt(self.Um)), W)

        # Eigiendecomposition of the RKHS covariance
        self.Uc, self.Vc = sorted_eigh(self.C, tiny=self.tiny)

        # Compute inverse square root of the covariance
        C_inv_sqrt = np.matmul(self.Vc, np.diagflat(1.0/np.sqrt(self.Uc)))
        C_inv_sqrt = np.matmul(C_inv_sqrt, self.Vc.T)

        # Compute square root of the covariance
        C_sqrt = np.matmul(self.Vc, np.diagflat(np.sqrt(self.Uc)))
        C_sqrt = np.matmul(C_sqrt, self.Vc.T)

        # Compute the S matrix
        S_kpca = C/self.P_scale**2
        S_lr = np.matmul(C_sqrt, W)
        S_lr = np.matmul(S_lr, S_lr.T)/self.Y_norm**2

        S = self.alpha*S_kpca + (1.0 - self.alpha)*S_lr

        # Compute eigendecomposition of S
        self.U, self.V = sorted_eigh(S, tiny=self.tiny)

        # Truncate the projections
        self.U = self.U[0:self.n_kpca]
        self.V = self.V[:, 0:self.n_kpca]

        # Define some matrices that will be re-used
        P = np.matmul(np.diagflat(1.0/np.sqrt(self.U)), self.V.T)
        PP = np.matmul(C_inv_sqrt, self.V)
        PP = np.matmul(PP, np.diagflat(np.sqrt(self.U)))

        # Compute and store mean of the projections
        self.T_mean = np.matmul(self.T_mean, PP)
        self.T_mean *= self.P_scale 

        # Compute projection matrix Pkt
        self.Pkt = np.matmul(self.Vm, np.diagflat(1.0/np.sqrt(self.Um)))
        self.Pkt = np.matmul(self.Pkt, PP)
        self.Pkt *= self.P_scale

        # Compute projection matrix Pty
        self.Pty = np.matmul(P, C_inv_sqrt)
        self.Pty = np.matmul(self.Pty, phi.T)
        self.Pty = np.matmul(self.Pty, Y)
        self.Pty /= self.P_scale

        # Compute projection matrix Ptk
        self.Ptk = np.matmul(P, C_sqrt)
        self.Ptk = np.matmul(self.Ptk, np.diagflat(np.sqrt(self.Um)))
        self.Ptk = np.matmul(self.Ptk, self.Vm.T)
        self.Ptk /= self.P_scale

        # Compute projection matrix Ptx
        if X is not None:
            self.Ptx = np.matmul(P, C_inv_sqrt)
            self.Ptx = np.matmul(self.Ptx, phi.T)
            self.Ptx = np.matmul(self.Ptx, X)
            self.Ptx /= self.P_scale

    def transform_K(self, KNM):
        """
            Transform the data into KPCA space

            ---Arguments---
            KNM: centered kernel between all points and the representative points

            ---Returns---
            T: centered transformed data
        """

        if self.Pkt is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute the KPCA-like projections
            T = np.matmul(KNM, self.Pkt) - self.T_mean

            return T

    def inverse_transform_K(self, KNM):
        """
            Compute the reconstruction of the kernel matrix

            ---Arguments---
            KNM: centered kernel between all points and the representative points

            ---Returns---
            Kr: centered reconstructed kernel matrix
        """

        if self.Ptk is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute the KPCA-like projections
            T = self.transform_K(KNM)
            Kr = np.matmul(T + self.T_mean, self.Ptk)

            return Kr

    def inverse_transform_X(self, KNM):
        """
            Compute the reconstruction of the X data

            ---Arguments---
            KNM: centered kernel between all points and the representative points

            ---Returns---
            Xr: centered reconstructed X data
        """

        if self.Ptx is None:
            print("Error: must provide X data during the KPCovR fit before transforming")
        else:

            # Compute the KPCA-like projections
            T = self.transform_K(KNM)

            # NOTE: why does adding T_mean not work?
            # perhaps because we need X centered in feature space,
            # not the RKHS feature space
            Xr = np.matmul(T, self.Ptx)

            return Xr

    def transform_Y(self, KNM):
        """
            Compute the predictions of Y

            ---Arguments---
            KNM: centered kernel between all points and the representative points

            ---Returns---
            Yp: centered predicted Y values
        """

        if self.Pty is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute predicted Y values
            T = self.transform_K(KNM)
            Yp = np.matmul(T + self.T_mean, self.Pty)

            return Yp

    def regression_loss(self, K, Y):
        """
            Compute the (kernel) regression loss

            ---Arguments---
            K: centered kernel
            Y: centered dependent (response) data

            ---Returns---
            regression_loss: regression loss
        """

        # Compute loss
        Yp = self.transform_Y(K)
        regression_loss = np.linalg.norm(Y - Yp)**2/self.Y_norm**2

        return regression_loss

    def projection_loss(self, K):
        """
            Compute the projection loss

            ---Arguments---
            K: centered kernel

            ---Returns---
            projection_loss: projection loss
        """

        # TODO: special formulation
        # Compute loss
        projection_loss = None

        return projection_loss

    def gram_loss(self, KNM, KMM, KNM_ref=None):
        """
            Compute the Gram loss

            ---Arguments---
            KNM: centered kernel between data and representative points
            KMM: centered kernel between representative points
            KNM_ref: centered reference kernel between the (training) points
                and the reference points

            ---Returns---
            gram_loss: Gram loss
        """

        T = self.transform_K(KNM)
        K = np.matmul(KNM, np.linalg.inv(KMM))

        # Compute Nystrom approximation to the kernel
        if KNM_ref is not None:
            K = np.matmul(K, KNM_ref.T)
        else:
            K = np.matmul(K, KNM.T)

        # Compute loss
        gram_loss = np.linalg.norm(K - np.matmul(T, T.T))**2 / self.P_scale**4

        return gram_loss
