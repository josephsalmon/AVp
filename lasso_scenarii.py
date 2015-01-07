# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:23:08 2014

@author: joseph salmon
"""

import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg import toeplitz


def ScenarioMEG(n_samples=10, n_features=500, sig_noise=0.1, s=5,
                    normalize=True, noise_type = 'Normal'):

    """
    REQUIRE THE MATRIX X_meg.mat

    Compute an n-sample y=X b+ e where the covariates have a fixed
    correlation level and the noise added is Gaussian with std sig_noise

    Parameters
    ----------

    sig_noise: std deviation of the additive White Gaussian noise

    rho: correlation between the covariates (S_ii=1 and S_ij=rho) where
    S is the covariance matrix of the p covariates.

    s: sparsity index of the underlying true coefficient vector

    normalize : boolean, optional, default True
        If ``True``, the regressors X will be normalized before regression.

    Returns
    -------
    y : ndarray, shape = (n_samples,); Target values of the scenario

    X :  shape (n_samples, n_features); Design matrix aka covariates elements

    beta : ndarray, shape = (,n_features);


    """
    import scipy.io as sio
    X_meg=sio.loadmat('X_meg.mat')
    X=X_meg['X_fixed']
    X=X[:n_samples,:n_features]

    if normalize is True:
        X /= np.sqrt(np.sum(X ** 2, axis=0)/n_samples)
    else:
        X
    beta = np.zeros((n_features,))
    beta[0:s] = 1
    beta=np.random.permutation(beta)
    y= AddNoise(np.dot(X, beta),sig_noise,noise_type)
    return y, beta, X

def ScenarioEquiCor(n_samples=10, n_features=50, sig_noise=0.1, rho=0.5, s=5,
                    normalize=True, noise_type = 'Normal'):
    """Compute an n-sample y=X b+ e where the covariates have a fixed
    correlation level and the noise added is Gaussian with std sig_noise

    Parameters
    ----------
    n_samples: number of independant sample to be generated (usually "n")

    n_features: number of features to be generated (usually "p")

    sig_noise: std deviation of the additive White Gaussian noise

    rho: correlation between the covariates (S_ii=1 and S_ij=rho) where
    S is the covariance matrix of the p covariates.

    s: sparsity index of the underlying true coefficient vector

    normalize : boolean, optional, default True
        If ``True``, the regressors X will be normalized before regression.

    Returns
    -------
    y : ndarray, shape = (n_samples,); Target values of the scenario

    X :  shape (n_samples, n_features); Design matrix aka covariates elements

    beta : ndarray, shape = (,n_features);

    """
    beta = np.zeros((n_features,))
    beta[0:s] = 1
    covar= (1-rho)*np.eye(n_features)+rho*np.ones([n_features,n_features])

    X=multivariate_normal(np.zeros(n_features,),covar,[n_samples])
    if normalize is True:
        X /= np.sqrt(np.sum(X ** 2, axis=0)/n_samples)
    else:
        X

    y= AddNoise(np.dot(X, beta),sig_noise,noise_type)
    return y, beta, X




def ScenarioPowDecayCor(n_samples=10, n_features=50, sig_noise=0.1, rho=0.5,
                        s=5, normalize=True, noise_type = 'Normal'):
    """Compute an n-sample y=X b+ e where the covariates have a power decay
    correlation level of the form rho^|i-j| and the noise added is Gaussian
    with std sig_noise

    Parameters
    ----------
    n_samples: number of independant sample to be generated (usually "n")

    n_features: number of features to be generated (usually "p")

    sig_noise: std deviation of the additive White Gaussian noise

    rho: correlation between the covariates (S_ii=1 and S_ij=rho) where
    S is the covariance matrix of the p covariates.

    s: sparsity index of the underlying true coefficient vector

    normalize : boolean, optional, default True
        If ``True``, the regressors X will be normalized before regression.

    Returns
    -------
    y : ndarray, shape = (n_samples,); Target values of the scenario

    X :  shape (n_samples, n_features); Design matrix aka covariates elements

    beta : ndarray, shape = (,n_features);

    """
    beta = np.zeros((n_features,))
    beta[0:s] = 1
    vect=np.zeros(n_features,)
    for k in xrange(n_features):
       vect[k]=rho**k

    covar=toeplitz(vect, vect)
    X=multivariate_normal(np.zeros(n_features,),covar,[n_samples])
    if normalize is True:
        X /= np.sqrt(np.sum(X ** 2, axis=0)/n_samples)
    else:
        X

    y= AddNoise(np.dot(X, beta),sig_noise,noise_type)
    return y, beta, X



def ScenarioConfoundingVar(n_samples=10, n_features=50, sig_noise=0.1,
                               rho=0.5, n_confounding_var = 3, s=5,
                               normalize=True, noise_type='Normal'):
    """ Scenerio from http://arxiv.org/pdf/1305.0355.pdf,  page 28

    Parameters
    ----------
    n_samples: number of independant sample to be generated (usually "n")

    n_features: number of features to be generated (usually "p")

    sig_noise: std deviation of the additive White Gaussian noise

    rho: correlation between the covariates (S_ii=1 and S_ij=rho) where
    S is the covariance matrix of the p covariates.

    s: sparsity index of the underlying true coefficient vector

    normalize : boolean, optional, default True
        If ``True``, the regressors X will be normalized before regression.

    Returns
    -------
    y : ndarray, shape = (n_samples,); Target values of the scenario

    X :  shape (n_samples, n_features); Design matrix aka covariates elements

    beta : ndarray, shape = (,n_features);

    """
    beta = np.zeros((n_features,))
    beta[0:s] = 1
    covar= np.eye(n_features)
    covar[n_features-n_confounding_var: n_features,0:s-1]=rho
    covar[0:s-1,n_features-n_confounding_var: n_features]=rho
    X=multivariate_normal(np.zeros(n_features,),covar,[n_samples])
    if normalize is True:
        X /= np.sqrt(np.sum(X ** 2, axis=0)/n_samples)
    else:
        X

    y= AddNoise(np.dot(X, beta),sig_noise,noise_type)
    return y, beta, X




def ScenarioShaoDengI(n_samples=10, n_features=50, sig_noise=10,
                               rho=0.5, s=5, normalize=True, noise_type='Normal'):
    """ Scenerio from J. Shao and X. Deng ,  page 28

    Parameters
    ----------
    n_samples: number of independant sample to be generated (usually "n")

    n_features: number of features to be generated (usually "p")

    sig_noise: std deviation of the additive White Gaussian noise

    rho: correlation between the covariates (S_ii=1 and S_ij=rho) where
    S is the covariance matrix of the p covariates.

    s: sparsity index of the underlying true coefficient vector

    normalize : boolean, optional, default True
        If ``True``, the regressors X will be normalized before regression.

    Returns
    -------
    y : ndarray, shape = (n_samples,); Target values of the scenario

    X :  shape (n_samples, n_features); Design matrix aka covariates elements

    beta : ndarray, shape = (,n_features);

    """
    beta = np.zeros((n_features,))

    beta[0:s] = np.linspace(1.1,1+0.1*s,s,endpoint=True)
    covar= (1-rho)*np.eye(n_features)+rho*np.ones([n_features,n_features])

    X=multivariate_normal(np.zeros(n_features,),covar,[n_samples])
    if normalize is True:
        X /= np.sqrt(np.sum(X ** 2, axis=0)/n_samples)
    else:
        X

    y= AddNoise(np.dot(X, beta),sig_noise,noise_type)
    return y, beta, X



def AddNoise(vect, sig_noise, noise_type = 'Normal'):
    # sig_noise represents the standard deviation!
    n_samples = vect.shape[0]
    if noise_type == 'Normal':
        epsilon =  sig_noise*np.random.randn(n_samples,)
    elif noise_type == 'Laplace':
        epsilon = np.random.laplace(0,sig_noise/np.sqrt(2),n_samples)
    else:
        epsilon = sig_noise*np.random.randn(n_samples,)
    y=vect+epsilon
    return y