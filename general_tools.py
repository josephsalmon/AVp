# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 10:38:13 2014

@author: jo
"""

import numpy as np
from sklearn.linear_model import LinearRegression


softthresh = lambda x, a: np.sign(x) * np.maximum(np.abs(x) - a, 0)
hardthresh = lambda x, a: x * (np.abs(x) > a)



def My_nonzeros(vector, eps_machine=1e-12):
    indexes_to_keep = np.where(np.abs(vector) > eps_machine)[0]
#    indexes_to_keep = np.nonzero(vector)
    return indexes_to_keep


def Support(coefs_path, eps_machine=1e-12):
    """Compute for a list of n_kinks coefficients, their support and
    their size

    Parameters
    ----------
    coefs_path: ndarray, shape (n_features, n_kinks); Matrix of n_kinks
    candidate beta. (eg. obtained with the lars algorithm).
    Implicitely assume the first XXX

    Returns
    -------

    index_list: list, shape(n_kinks) . list of the indexes chosen at each
    kink. Note that the first element of the list is the empty list
    (no variable used first)

    index_size: float vector with ith coordinate

    """

    if np.ndim(coefs_path) == 1:

        index_list = My_nonzeros(coefs_path)
        index_size = len(index_list)
    else:
        _, n_kinks = coefs_path.shape
        index_list = []
        index_size = np.zeros(n_kinks, int)
        for k in xrange(n_kinks):
            indexes_to_keep = np.reshape(My_nonzeros(coefs_path[:, k]), -1)
            index_list.append(indexes_to_keep)
            index_size[k] = len(indexes_to_keep)
    return index_list, index_size


def Refitting(coefs, X, y, eps_machine=1e-12):
    """Compute the  Two Stage Lasso (debiased Lasso): do a LeastSquare refitting
    over the support of the estimated coeffieicents.

    Parameters
    ----------
    coefs: ndarray, shape (n_features, n_kinks); Matrix of n_kinks candidate
    beta. (eg. obtained with the lars algorithm). Implicitely assume the
    first

    X: ndarray,shape (n_samples, n_features); Design matrix aka covariates
    elements

    y : ndarray, shape = (n_samples,); noisy vector of observation

    Returns
    -------
    all_solutions: ndarray, shape (n_kinks, n_features); refitted solution
    based on the original coefs vectors

    index_list: list, shape(n_kinks) . list of the indexes chosen at each
    kink. Note that the first element of the list is the empty list
    (no variable used first)

    index_size:

    """
    regr = LinearRegression(fit_intercept=False)
    if np.ndim(coefs) == 1:
        n_features = coefs.size
        all_solutions = np.zeros(n_features)
        indexes = My_nonzeros(coefs, eps_machine=eps_machine)
        if len(indexes) == 0:
            indexes_to_keep = []
            index_size = 0
            all_solutions = all_solutions
        else:
            indexes_to_keep = np.reshape(indexes, -1)
            regr.fit(X[:, indexes_to_keep], y)
            all_solutions[indexes_to_keep] = regr.coef_
            index_list = indexes_to_keep
            index_size = len(indexes_to_keep)
    else:
        n_features, n_kinks = coefs.shape
        all_solutions = np.zeros((n_features, n_kinks))
        index_list = []
        index_list.append([])
        index_size = np.zeros(n_kinks, int)
        for k in xrange(n_kinks - 1):
            indexes = np.nonzero(coefs[:, k + 1])[0]
            indexes_to_keep = np.reshape(indexes, -1)

            if len(indexes) == 0:
                index_list.append([])
                index_size[k + 1] = 0
                all_solutions[..., k + 1] = np.zeros(n_features)
            else:
                regr.fit(X[:, indexes_to_keep], y)
                all_solutions[indexes_to_keep, k + 1] = regr.coef_
                index_list.append(indexes_to_keep)
                index_size[k + 1] = len(indexes_to_keep)

    return all_solutions, index_list, index_size


def PredictionError(X, coefs_path, beta):
    """Compute all the  ||X beta-X coefs_path[i]||^2/n_samples for true beta
    and observed coefficients coef_path.

    Parameters
    ----------
    X: shape (n_samples, n_features); Design matrix aka covariates elements

    coefs_path: shape (n_features, n_kinks); Matrix of n_kinks beta's estimation

    beta: shape (n_features, ); original coefficients

    Returns
    -------
    Err: shape (n_kinks, ) float vector with ith coordinate
    ||X beta-X coefs_path[i]||^2/n_samples

    """
    n_samples, n_features = X.shape
    if np.ndim(coefs_path) == 1:
        Err = np.sum((np.dot(X, beta) - np.dot(X, coefs_path)) ** 2)
    else:
        n_features, n_kinks = coefs_path.shape
        Err = np.sum((np.tile(np.dot(X, beta), (n_kinks, 1)).T -
                      np.dot(X, coefs_path)) ** 2, 0)

    return Err / n_samples


def PredictionErrorEstimated(X, coefs_path, y):
    """Compute all the  ||Y-X coefs_path[i]||^2 for true beta and observed
    coefficients coef_path.

    Parameters
    ----------
    X: shape (n_samples, n_features); Design matrix aka covariates elements

    coefs_path: shape (n_features, n_kinks); Matrix of n_kinks beta's estimation

    y: shape (n_samples, ); noisy signal

    Returns
    -------
    Err: shape (n_kinks, ) float vector with ith coordinate
    ||y-X coefs_path[i]||^2

    """
    n_samples, n_features = X.shape
    if np.ndim(coefs_path) == 1:
        Err = np.sum((y - np.dot(X, coefs_path))**2)
    else:
        n_features, n_kinks = coefs_path.shape
        Err = np.sum((np.tile(y, (n_kinks, 1)).T - np.dot(X, coefs_path))**2,
                     0)

    return Err / n_samples


def EstimationError(coefs_path, beta):
    """Compute all the  ||beta-coefs_path[i]||_infty for true beta and observed
    coefficients coef_path.

    Parameters
    ----------
    coefs_path: shape (n_features, n_kinks); Matrix of n_kinks beta's estimation

    beta: shape (n_features, ); original coefficients

    Returns
    -------
    Err: shape (n_kinks, ) float vector with ith coordinate
    ||beta-coefs_path[i]||_\infty

    """
    if np.ndim(coefs_path) == 1:
        Err = np.max(np.abs(beta - coefs_path))
    else:
        n_features, n_kinks = coefs_path.shape
        Err = np.max(np.abs(np.tile(beta, (n_kinks, 1)).T - coefs_path), 0)

    return Err


def EstimationError_2(coefs_path,beta):
    """Compute all the  ||beta-coefs_path[i]||_infty for true beta and observed
    coefficients coef_path.

    Parameters
    ----------
    coefs_path: shape (n_features, n_kinks); Matrix of n_kinks beta's estimation

    beta: shape (n_features, ); original coefficients

    Returns
    -------
    Err: shape (n_kinks, ) float vector with ith coordinate
    ||beta-coefs_path[i]||_\infty

    """
    if np.ndim(coefs_path) == 1:
        Err = np.sum((beta - coefs_path)**2)
    else:
        n_features, n_kinks = coefs_path.shape
        Err = np.sum((np.tile(beta, (n_kinks, 1)).T - coefs_path)**2, 0)

    return np.sqrt(Err)


def FalsePositive(coefs_path, beta):
    """Compute all the  False Positive variables for coefs_path, where beta
    is the true estimation coefficient

    Parameters
    ----------
    coefs_path: shape (n_features, n_kinks); Matrix of n_kinks beta's estimation

    beta: shape (n_features, ); original coefficients

    Returns
    -------
    nb_fp: shape (n_kinks, ) : number of variables active in coefs_path  not in beta
    (ie. False Positive))
    """

    index_list_true, _ = Support(beta)

    if np.ndim(coefs_path) == 1:
        index_list = My_nonzeros(coefs_path)
        nb_fp = len(np.setdiff1d(index_list, index_list_true))
    else:
        _, n_kinks = coefs_path.shape
        nb_fp = np.zeros([n_kinks, ])
        for k in xrange(n_kinks):
            indexes_to_keep = np.reshape(My_nonzeros(coefs_path[:, k]), -1)
            nb_fp[k] = len(np.setdiff1d(indexes_to_keep, index_list_true))
    return nb_fp


def FalseNegative(coefs_path, beta):
    """Compute all the  False Positive variables for coefs_path, where beta
    is the true estimation coefficient

    Parameters
    ----------
    coefs_path: shape (n_features, n_kinks); Matrix of n_kinks beta's estimation

    beta: shape (n_features, ); original coefficients

    Returns
    -------
    nb_fn: shape (n_kinks, ) : number of variables non active in coefs_path
    but that are active in beta
    (ie. False Positive))
    """

    index_list_true, _ = Support(beta)
    if np.ndim(coefs_path) == 1:
        index_list = My_nonzeros(coefs_path)
        nb_fn = len(np.setdiff1d(index_list_true, index_list))
    else:
        _, n_kinks = coefs_path.shape
        nb_fn = np.zeros([n_kinks, ])
        for k in xrange(n_kinks):
            indexes_to_keep = np.reshape(My_nonzeros(coefs_path[:, k]), -1)
            nb_fn[k] = len(np.setdiff1d(index_list_true, indexes_to_keep))
    return nb_fn


def AV_p_indexes_to_consider(index_list_ordered, j):
    length = len(index_list_ordered)
    to_test = index_list_ordered[j]
    lst_indexes_to_consider = [i for i, x in enumerate(index_list_ordered) if x==to_test]
    lst_indexes_to_consider = list(set(lst_indexes_to_consider + range(j, length)))
    lst_indexes_to_consider.remove(j)
    lst_indexes_to_consider.sort(reverse=False)  # put true to change the order of visiting the j for testing S_ij
    return lst_indexes_to_consider

def ThRR_grid(beta_Ridge, n_alphas, max_nb_variables):

    beta_abs = np.abs(beta_Ridge)
    sort_beta_abs = np.argsort(beta_abs)[::-1]
    alpha_th = beta_abs[sort_beta_abs[(np.linspace(0,
                                                   max_nb_variables,
                                                   n_alphas, endpoint=False)).astype(int)]]
    return alpha_th
