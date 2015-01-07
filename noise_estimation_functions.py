# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 18:03:34 2014

@author: joseph salmon
"""

import numpy as np
from general_tools import softthresh
from sklearn.linear_model import Lasso
import scipy as sp


def SqrtLasso(X, y, alpha, max_iter, tol, coef_ini):
    K=2
    NorColX=np.sqrt(np.sum(X**2,0))
    X=X/NorColX# *np.sqrt(y.shape[0])
    X=X/K
    y=y/K

    #SHE?????
    #    k0 = 1*norm(X, 2)/sqrt(2); # norme spectrale de X
    #    X = X / k0;
    #    y = y / k0;


    #normalize X first such that ||X_{:,j}||=1

   #print np.sqrt(np.sum(X**2,0))
    j=0
    coef = np.zeros(coef_ini.shape)
    coef_old = coef_ini
    print coef_old
    while j<max_iter:
        j=j+1
        resid = y - np.dot(X,coef_old)
        coef_to_update = np.dot(np.transpose(X), resid) + coef_old
        sig_hat_unnormalized = np.sqrt(np.sum(resid**2))
        thresh = alpha * sig_hat_unnormalized
        print thresh
        ttInds = np.where(np.abs(coef_to_update) > thresh)[0];
        print ttInds
        print coef_to_update
        coef = (softthresh(coef_to_update, thresh));
        print coef
        if (np.max(np.abs(coef - coef_old)) < tol):
            print coef
            print 'weird'
            print coef_old
            break
        coef_old = (coef)
        print '000000'
        print coef_old
        print coef
        print '----'
    coef = coef/NorColX # *np.sqrt(y.shape[0])
    sig_hat = sig_hat_unnormalized/np.sqrt(y.shape[0])
    print j
    return coef, sig_hat


def SqrtLasso_ista(X, y, alpha, max_iter, tol, coef_ini):
    step_size=1./y.shape[0]
    #normalize X first such that ||X_{:,j}||=1
    NorColX=np.sqrt(np.sum(X**2,0))
    print NorColX
    X=X/NorColX# *np.sqrt(y.shape[0])
    #print np.sqrt(np.sum(X**2,0))
    j=0
    coef = coef_ini
    coef_old = coef_ini
    while j<max_iter:
        resid = y - np.dot(X, coef_old)
        sig_hat_unnormalized = np.sqrt(np.sum(resid**2))
        coef_to_update = step_size*np.dot(np.transpose(X), resid)+coef_old
        #print coef_to_update
        thresh = alpha * sig_hat_unnormalized*step_size
        coef = softthresh(coef_to_update,thresh)
        #print thresh
        #print 'above threh'
        #print coef
        #print sig_hat_unnormalized
        j=j+1
        if (np.max(np.abs(coef - coef_old)) < tol):
            break
        coef_old = coef
    coef = coef/NorColX # *np.sqrt(y.shape[0])
    sig_hat = sig_hat_unnormalized/np.sqrt(y.shape[0])
    print j
    return coef, sig_hat


def SqrtLasso_SZ(X, y, alpha, max_iter, tol, Lasso_tol, coef_ini):
    j=0
    print max_iter
    coef = coef_ini
    coef_old = coef_ini
    resid = y - np.dot(X, coef_old)
    sig_hat_old = np.sqrt(np.sum(resid**2)/y.shape[0])
    clf = Lasso(alpha=alpha*sig_hat_old, warm_start = True, tol = Lasso_tol, fit_intercept=False)
    clf.fit(X, y)
    coef_old=clf.coef_

    while j<max_iter:
        resid = y - np.dot(X, coef_old)
        sig_hat = np.sqrt(np.sum(resid**2)/y.shape[0])
        print sig_hat
        print '--------'
        clf.alpha=alpha*sig_hat
        clf.fit(X, y)
        coef=clf.coef_
        j+=1
        if ((np.abs(sig_hat_old - sig_hat)) < tol):
        #if (np.max(np.abs(clf.coef_ - coef_old)) < tol):
            break
        sig_hat_old = sig_hat
        coef_old=coef
    print j
    return coef, sig_hat


def Lambda_SZ(n_features,n_samples):
    L=0.1
    Lold=0.
    while (abs(L-Lold)>0.001):
        k=(L**4+2*L**2)
        Lold=L
        L=-sp.stats.norm.ppf(min(k/n_features,0.99))
        L=(L+Lold)/2
        if n_features==1:
            L=0.5
    lam0 = np.sqrt(2/n_samples)*L
    return lam0