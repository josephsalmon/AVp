# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 15:44:16 2014

@author: jo
"""

import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model.base import LinearModel

from sklearn.base import RegressorMixin
from sklearn.linear_model import lasso_path
from general_tools import Support, My_nonzeros, hardthresh, ThRR_grid, AV_p_indexes_to_consider

def Support_old(coefs_path):
    """Compute for a list of n_kinks coefficients, their support and
    their size

    Parameters
    ----------
    coefs_path: ndarray, shape (n_features, n_kinks); Matrix of n_kinks candidate
    beta. (eg. obtained with the lars algorithm). Implicitely assume the
    first

    Returns
    -------

    index_list: list, shape(n_kinks) . list of the indexes chosen at each
    kink. Note that the first element of the list is the empty list (no variable used first)

    index_size: float vector with ith coordinate

    """

    if np.ndim(coefs_path) == 1:
        index_list = np.nonzero(coefs_path)
        index_size = len(index_list)
    else:
        _, n_kinks=coefs_path.shape
        index_list=[]
        index_size=np.zeros(n_kinks,int)
        for k in xrange(n_kinks):
            indexes_to_keep = np.reshape(np.nonzero(coefs_path[:,k]),-1)
            index_list.append(indexes_to_keep)
            index_size[k] = len(indexes_to_keep)
    return  index_list, index_size


def ListUnion(index_list,i,j):
    """Compute the  list  (index_list[i]) U (index_list[j])

    Parameters
    ----------
    index_list: list, shape(n_kinks)

    i: index of the first A
    j|

    Returns
    -------
    union= list, shape unknown

    """
    union=np.union1d(index_list[i],index_list[j])
    return union


def ListForOrdering_old(index_list,index_size):
    """Compute the  list  such that index sets are of decreasing size.
    and REMOVE EMPTY sets, but BEWARE it rather be the first one... Possible
    issues when it is present, rather avoid.

    Parameters
    ----------
    index_list: list, shape(n_kinks,). Elements are np.arrays of varying size

    index_size: narray, shape(n_kinks,).  Elements are int given the size
    of index_list[i]

    Returns
    -------
    list_ordered = list, shape unknown (depends on if there is a null set)

    index_size_order = narray shape unknown

    index_new_order =
    """
    find_nonzeros=np.where(index_size != 0)[0]
    index_size_int=index_size[find_nonzeros]
    index_list_ordered=index_list
    index_new_order=np.argsort(index_size_int)[::-1]
    index_size_order=index_size_int[index_new_order]
    index_list_ordered = [index_list_ordered[i] for i in index_new_order]
    return index_list_ordered,index_size_order, index_new_order


def ListForOrdering(index_list,index_size):
    """Compute the  list  such that index sets are of decreasing size.
    and REMOVE EMPTY sets, but BEWARE it rather be the first one... Possible
    issues when it is present, rather avoid.

    Parameters
    ----------
    index_list: list, shape(n_kinks,). Elements are np.arrays of varying size

    index_size: narray, shape(n_kinks,).  Elements are int given the size
    of index_list[i]

    Returns
    -------
    list_ordered = list, shape unknown (depends on if there is a null set)

    index_size_order = narray shape unknown

    index_new_order =
    """
    if len(index_size)==1:
        index_list_ordered=index_list
        index_size_order=index_size
        index_new_order=index_size
    else:
        index_list_ordered=index_list
        index_new_order=np.argsort(index_size)[::-1]
        index_size_order=[index_size[i] for i in index_new_order]
        index_list_ordered = [index_list_ordered[i] for i in index_new_order]
    return index_list_ordered,index_size_order, index_new_order


def ListForOrdering_fast(index_list,index_size):
    """Compute the  list  such that index sets are of decreasing size.
    and REMOVE EMPTY sets, but BEWARE it rather be the first one... Possible
    issues when it is present, rather avoid.

    Parameters
    ----------
    index_list: list, shape(n_kinks,). Elements are np.arrays of varying size

    index_size: narray, shape(n_kinks,).  Elements are int given the size
    of index_list[i]

    Returns
    -------
    list_ordered = list, shape unknown (depends on if there is a null set)

    index_size_order = narray shape unknown

    index_new_order =
    """
    if len(index_size)==1:
        index_list_ordered=index_list
        index_size_order=index_size
        index_new_order=[0]
    else:
        index_list_ordered=index_list
        index_new_order=np.argsort(index_size)[::1]
        index_size_order=[index_size[i] for i in index_new_order]
        index_list_ordered = [index_list_ordered[i] for i in index_new_order]
    return index_list_ordered,index_size_order, index_new_order



def ForEstimator_fast_v2(X, y,  index_list_ordered, index_size_order, a_param):
    """Compute the  ForEstimator

    Parameters
    ----------
    X: ndarray,shape (n_samples, n_features); Design matrix aka covariates
    elements

    y : ndarray, shape = (n_samples,); noisy vector of observation

    index_list_ordered: list, shape(n_kinks). Note that the ordering is with
    decreasing size of list (support getting smaller)

    index_size_order: ndarray, shape (n_kinks,).

    a_param: float : value of the parameters a in the paper

    Returns
    -------
    coefs_for = ndarray, shape (n_features,) : coefficient estimated

    y_for = ndarray, shape (n_samples,) : prediction vector

    index_for: int ; index such that the coefficients estimated are for the
    the indexes corresponding to list index_list_ordered[inde_for]

    support_for: ndarray, shape (n_kinks,) : encode the support selected

    """
    n_samples, n_features = X.shape
    coefs_tot_FOR = np.zeros([n_features,])
    nb_support = len(index_size_order)
    if nb_support == 0:
        print "stupid there's no candidate in this list !!!"
        coefs_for = []
        y_for = []
        index_for = []
        support_for = []
    elif nb_support == 1:

        index_for = 0
        support_for = index_list_ordered[0] # compute the A_j
        length_i = index_size_order[0]
        X_support = X[:, support_for]
        coefs_for, y_for = Prediction_Step(support_for,
                                                   length_i, X_support, y)
        coefs_tot_FOR[support_for]=coefs_for
    else:
        i = 0
        while i < nb_support:
            support_for_i = index_list_ordered[i] # compute the A_j
            length_i = index_size_order[i]
            X_support_i = X[:, support_for_i]
            coefs_for_i, y_for_i = Prediction_Step(support_for_i,
                                                   length_i, X_support_i, y)
            j = i+1
            test_result = False
            while (j < nb_support) & (test_result == False):
                support_for_ij = np.union1d(support_for_i,index_list_ordered[j])
                length_ij=len(support_for_ij)
                X_support_ij = X[:, support_for_ij]
                coefs_for_ij, y_for_ij = Prediction_Step(support_for_ij,
                                                       length_ij, X_support_ij, y)
                if np.sum((y_for_ij -y_for_i)**2)<= a_param*(length_ij+length_i):
                    j+=1
                else:
                    test_result=True
            if test_result == True:
                i+=1
            else:
                break
        index_for = i
        support_for = support_for_i
        coefs_for = coefs_for_i
        y_for = y_for_i
        coefs_tot_FOR[support_for]=coefs_for
    return coefs_tot_FOR, y_for, index_for, support_for


def AVp(X, y,  index_list_ordered, index_size_order, a_param):
    """Compute the  ForEstimator

    Parameters
    ----------
    X: ndarray,shape (n_samples, n_features); Design matrix aka covariates
    elements

    y : ndarray, shape = (n_samples,); noisy vector of observation

    index_list_ordered: list, shape(n_kinks).

    index_size_order: ndarray, shape (n_kinks,).

    a_param: float : value of the parameters a in the paper

    Returns
    -------
    coefs_for = ndarray, shape (n_features,) : coefficient estimated

    y_for = ndarray, shape (n_samples,) : prediction vector

    index_for: int ; index such that the coefficients estimated are for the
    the indexes corresponding to list index_list_ordered[inde_for]

    support_for: ndarray, shape (n_kinks,) : encode the support selected

    """
    n_samples, n_features = X.shape
    coefs_tot_AVp = np.zeros([n_features,])
    nb_support = len(index_size_order)
    if nb_support == 0:
        print "stupid there's no candidate in this list !!!"
        coefs_AVp = []
        y_AVp = []
        index_AVp = []
        support_AVp = []
    elif nb_support == 1:

        index_AVp = 0
        support_AVp = index_list_ordered[0] # compute the A_j
        length_i = index_size_order[0]
        X_support = X[:, support_AVp]
        coefs_AVp, y_AVp = Prediction_Step(support_AVp,
                                                   length_i, X_support, y)
        coefs_tot_AVp[support_AVp]=coefs_AVp
    else:
        i = 0
        while i < nb_support-1:
            support_AVp_i = index_list_ordered[i] # compute the A_j
            length_i = index_size_order[i]
            X_support_i = X[:, support_AVp_i]
            coefs_AVp_i, y_AVp_i = Prediction_Step(support_AVp_i,
                                                   length_i, X_support_i, y)

            idx_to_test = AV_p_indexes_to_consider(index_size_order,i) #get list of indexes to test
            max_numb_test = len(idx_to_test)
            test_result = False
            k=0
            while (k < max_numb_test) & (test_result == False):
                j=idx_to_test[k]
                support_AVp_ij = np.union1d(support_AVp_i,index_list_ordered[j])
                length_ij=len(support_AVp_ij)
                X_support_ij = X[:, support_AVp_ij]
                coefs_AVp_ij, y_AVp_ij = Prediction_Step(support_AVp_ij,
                                                       length_ij, X_support_ij, y)
                if np.sum((y_AVp_ij -y_AVp_i)**2)<= a_param*(length_ij+length_i):
                    k+=1
                else:
                    test_result=True
            if test_result == True:
                i+=1
            else:
                break
        index_AVp = i

        if i == (nb_support-1):
            length_i = index_size_order[i]
            support_AVp = index_list_ordered[i]
            X_support_i = X[:, support_AVp]
            coefs_AVp, y_AVp  = Prediction_Step(support_AVp,
                                                   length_i, X_support_i, y)

        else:
            support_AVp = support_AVp_i
            coefs_AVp = coefs_AVp_i
            y_AVp = y_AVp_i

        coefs_tot_AVp[support_AVp]=coefs_AVp
    return coefs_tot_AVp, y_AVp, index_AVp, support_AVp


def ForEstimator_for_display(X, y,  index_list_ordered, index_size_order,
                             a_param, idx_re_order):
    """Compute the  ForEstimator

    Parameters
    ----------
    X: ndarray,shape (n_samples, n_features); Design matrix aka covariates
    elements

    y : ndarray, shape = (n_samples,); noisy vector of observation

    index_list_ordered: list, shape(n_kinks). Note that the ordering is with
    decreasing size of list (support getting smaller)

    index_size_order: ndarray, shape (n_kinks,).

    a_param: float : value of the parameters a in the paper

    Returns
    -------
    coefs_for = ndarray, shape (n_features,) : coefficient estimated

    y_for = ndarray, shape (n_samples,) : prediction vector

    index_for: int ; index such that the coefficients estimated are for the
    the indexes corresponding to list index_list_ordered[inde_for]

    support_for: ndarray, shape (n_kinks,) : encode the support selected

    """
    n_samples, n_features = X.shape
    coefs_tot_FOR = np.zeros([n_features,])
    nb_support = len(index_size_order)
    matrix_of_test = np.zeros((nb_support, nb_support))
    print nb_support
    if nb_support == 0:
        print "stupid there's no candidate in this list !!!"
        coefs_for = []
        y_for = []
        index_for = []
        support_for = []
    elif nb_support == 1:

        index_for = 0
        support_for = index_list_ordered[0] # compute the A_j
        length_i = index_size_order[0]
        X_support = X[:, support_for]
        coefs_for, y_for = Prediction_Step(support_for,
                                                   length_i, X_support, y)
        coefs_tot_FOR[support_for]=coefs_for
    else:
        i = 0
        while i < nb_support:
            support_for_i = index_list_ordered[i] # compute the A_j
            length_i = index_size_order[i]
            X_support_i = X[:, support_for_i]
            coefs_for_i, y_for_i = Prediction_Step(support_for_i,
                                                   length_i, X_support_i, y)
            j = i+1
            while (j < nb_support):
                support_for_ij = np.union1d(support_for_i,index_list_ordered[j])
                length_ij=len(support_for_ij)
                X_support_ij = X[:, support_for_ij]
                coefs_for_ij, y_for_ij = Prediction_Step(support_for_ij,
                                                       length_ij, X_support_ij, y)
                matrix_of_test[idx_re_order[i],idx_re_order[j]] = np.sum((y_for_ij -y_for_i)**2)/(length_ij+length_i)
                j+=1
            i+=1

        index_for = i
        support_for = support_for_i
        coefs_for = coefs_for_i
        y_for = y_for_i
        coefs_tot_FOR[support_for]=coefs_for
    return coefs_tot_FOR, y_for, index_for, support_for, matrix_of_test


def Av_p_for_display(X, y, index_list_ordered, index_size_order,
                             a_param, idx_re_order):
    """Compute the  Av_p estimator for display. Not that it keeps the original
    ordering, so weird ordering can happen: when a lot support are not in a
    deacreasing order the output my give the impression that the AV_p index
    is not satisfying the rule of selection; It is just that the display
    is usually with the original indexing and not the one with increasing size
    of support.

    Parameters
    ----------
    X: ndarray,shape (n_samples, n_features); Design matrix aka covariates
    elements

    y : ndarray, shape = (n_samples,); noisy vector of observation

    index_list_ordered: list, shape(n_kinks). Note that the ordering is with
    decreasing size of list (support getting smaller)

    index_size_order: ndarray, shape (n_kinks,).

    a_param: float : value of the parameters a in the paper

    Returns
    -------
    coefs_for = ndarray, shape (n_features,) : coefficient estimated

    y_for = ndarray, shape (n_samples,) : prediction vector

    index_for: int ; index such that the coefficients estimated are for the
    the indexes corresponding to list index_list_ordered[inde_for]

    support_for: ndarray, shape (n_kinks,) : encode the support selected

    """
    n_samples, n_features = X.shape
    coefs_tot_FOR = np.zeros([n_features,])
    nb_support = len(index_size_order)
    matrix_of_test = np.zeros((nb_support, nb_support))
    if nb_support == 0:
        print "stupid there's no candidate in this list !!!"
        coefs_for = []
        y_for = []
        index_for = []
        support_for = []
    elif nb_support == 1:

        index_for = 0
        support_for = index_list_ordered[0] # compute the A_j
        length_i = index_size_order[0]
        X_support = X[:, support_for]
        coefs_for, y_for = Prediction_Step(support_for,
                                                   length_i, X_support, y)
        coefs_tot_FOR[support_for]=coefs_for
    else:
        i = 0
        while i < nb_support-1:
            support_for_i = index_list_ordered[i] # compute the A_j
            length_i = index_size_order[i]
            X_support_i = X[:, support_for_i]
            coefs_for_i, y_for_i = Prediction_Step(support_for_i,
                                                   length_i, X_support_i, y)

            idx_to_test = AV_p_indexes_to_consider(index_size_order,i) #get list of indexes to test
            max_numb_test = len(idx_to_test)
            k=0
            while (k < max_numb_test):
                j=idx_to_test[k]
                support_for_ij = np.union1d(support_for_i,index_list_ordered[j])
                length_ij=len(support_for_ij)
                X_support_ij = X[:, support_for_ij]
                coefs_for_ij, y_for_ij = Prediction_Step(support_for_ij,
                                                       length_ij, X_support_ij, y)
                matrix_of_test[idx_re_order[i],idx_re_order[j]] = np.sum((y_for_ij -y_for_i)**2)/(length_ij+length_i)
                k+=1
            i+=1

        index_for = i
        support_for = support_for_i
        coefs_for = coefs_for_i
        y_for = y_for_i
        coefs_tot_FOR[support_for]=coefs_for
    return coefs_tot_FOR, y_for, index_for, support_for, matrix_of_test


def Prediction_Step(support_for, length_for, X_support, y):
    OLS=LinearRegression(fit_intercept = False)
    if length_for == 0:
        coefs_for = np.zeros(length_for, )
        y_for = np.zeros(y.shape)
    else:
        OLS.fit(X_support, y)
        coefs_for = OLS.coef_
        y_for = OLS.predict(X_support)
    return coefs_for, y_for


class ThRR(LinearModel, RegressorMixin):
    """docstring for ThRR"""
    def __init__(self, alpha = 1.0, thresh = 1.0 , tol = 1e-7, fit_intercept =False):
        self.alpha = alpha
        self.thresh = thresh
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        coef=np.squeeze(RidgePath(X, y, self.alpha, np.array([self.thresh]), tol=self.tol, fit_intercept=self.fit_intercept))
        self.coef_ = coef
        self.intercept_ = 0.
        return self


class LSThRR(LinearModel, RegressorMixin):
    """docstring for ThRR"""
    def __init__(self, alpha = 1.0, thresh = 1.0, eps_machine = 1e-12,
                 tol = 1e-7, fit_intercept = False):
        self.alpha = alpha
        self.thresh = thresh
        self.eps_machine = eps_machine
        self.tol = tol
        self.fit_intercept =fit_intercept

    def fit(self, X, y):
        #ridge = Ridge(self.alpha, self.tol, self.fit_intercept).fit(X, y)
        coef_1stp = np.squeeze(RidgePath(X, y, self.alpha, np.array([self.thresh]), tol=self.tol, fit_intercept=self.fit_intercept))
        n_features = X.shape[1]
        indexes_to_keep = My_nonzeros(coef_1stp, eps_machine = self.eps_machine)
        #print indexes_to_keep
        #print np.nonzero(coef_1stp)[0]
        regr = LinearRegression(fit_intercept =self.fit_intercept)
        if len(indexes_to_keep)==0:
            coef=np.zeros(n_features,)
        else:
            coef=np.zeros(n_features,)
            regr.fit(X[:, indexes_to_keep], y)
            coef[indexes_to_keep]=regr.coef_
        self.coef_ = coef
        self.intercept_ = 0.
        return self


class LSLasso(LinearModel, RegressorMixin):
    """docstring for LSLasso"""
    def __init__(self, alpha = 1.0, eps_machine = 1e-12,
                 max_iter= 10000, tol = 1e-7, fit_intercept=False):
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps_machine = eps_machine
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        regr=LinearRegression(fit_intercept = self.fit_intercept)
        _,lasso_coef,_= lasso_path(X, y, alphas = [self.alpha],
                                   fit_intercept = False, return_models = False,
                                   max_iter = self.max_iter, tol = self.tol)
        n_features = lasso_coef.shape
        indexes_to_keep = My_nonzeros(lasso_coef, eps_machine = self.eps_machine)
        if len(indexes_to_keep)==0:
            coef=np.zeros(n_features[0],)
        else:
            coef=np.zeros(n_features[0],)
            regr.fit(X[:, indexes_to_keep], y)
            coef[indexes_to_keep]=regr.coef_
        self.coef_ = coef
        self.intercept_ = 0.
        return self


def LassoAVp(X, y, alpha_grid, max_iter, tol, a_param):

    _, coefs_Lasso, _ = lasso_path(X, y, alphas = alpha_grid,
                                   fit_intercept = False,
                                   return_models = False,
                                   max_iter = max_iter, tol = tol)

    idx_list, idx_size = Support(coefs_Lasso)
    idx_list_ordered, idx_size_order, idx_re_order  = ListForOrdering_fast(
                                            idx_list, idx_size)
    coef_LassoFOR, y_tot_for, idx_tot_for, support_tot_for = AVp(X,
                                                y, idx_list_ordered,
                                                idx_size_order, a_param)
    idx_LassoFOR = idx_re_order[idx_tot_for]
    return coef_LassoFOR, idx_LassoFOR


def LassoAVp_for_display(X, y, alpha_grid, max_iter, tol, a_param):

    _, coefs_Lasso, _ = lasso_path(X, y, alphas = alpha_grid,
                                   fit_intercept = False,
                                   return_models = False,
                                   max_iter = max_iter, tol = tol)

    idx_list, idx_size = Support(coefs_Lasso)
    idx_list_ordered, idx_size_order, idx_re_order  = ListForOrdering_fast(
                                            idx_list, idx_size)
    coef_LassoFOR,_, idx_tot_for, support_tot_for, matrix_test = Av_p_for_display(X,
                                                y, idx_list_ordered,
                                                idx_size_order, a_param, idx_re_order)

    return matrix_test


def ComputeRidgeCoef(X, y, alpha_ridge, fit_intercept = False, tol =  1e-7):
    clf = Ridge(alpha = alpha_ridge, fit_intercept = fit_intercept, tol =  tol)
    clf.fit(X, y)
    beta_Ridge = clf.coef_
    return beta_Ridge


def ComputeRidgeCoef2D(X, y, alpha_ridge2D, fit_intercept = False, tol =  1e-7):
    #    n_alphas=alpha_ridge_tab.shape[0]
    clf = Ridge(fit_intercept=fit_intercept)
    beta_Ridge_list = []
    for a in alpha_ridge2D:
        print a
        clf.set_params(alpha=a)
        clf.fit(X, y)
        beta_Ridge_list.append(clf.coef_)
    return beta_Ridge_list


def RidgePath(X, y, alpha_ridge, alpha_th, fit_intercept = False, tol =  1e-7):
    n_features = X.shape[1]
    n_alphas = len(alpha_th)

    beta_Ridge = ComputeRidgeCoef(X, y, alpha_ridge,
              fit_intercept = fit_intercept, tol =  tol)
    coefs_ThRR = hardthresh(np.tile(np.reshape(beta_Ridge, [n_features, 1]),
                                       [1, n_alphas]),
                               np.tile(np.reshape(alpha_th,[1,n_alphas]),
                                       [n_features,1]))
    return coefs_ThRR


def RidgePath2D(X, y, alpha_ridge2D, n_alphas_th, max_nb_variables, fit_intercept = False, tol =  1e-7):
    n_features = X.shape[1]
    #    n_alphas_th = len(alpha_th2D)
    n_alphas_ridge = len(alpha_ridge2D)
    coefs_ThRR=np.zeros([n_features, n_alphas_ridge*n_alphas_th])

    beta_Ridge = ComputeRidgeCoef2D(X, y, alpha_ridge2D,
              fit_intercept = fit_intercept, tol =  tol)

    for i in range(n_alphas_ridge):

        alpha_th2D = ThRR_grid(beta_Ridge[i], n_alphas_th, max_nb_variables)
        coefs_ThRR[:,(i)*n_alphas_th:(i+1)*n_alphas_th] = hardthresh(np.tile(np.reshape(beta_Ridge[i], [n_features, 1]),
                                       [1, n_alphas_th]),
                               np.tile(np.reshape(alpha_th2D,[1,n_alphas_th]),
                                       [n_features,1]))

    return coefs_ThRR


def ThRRAVp(X, y, alpha_ridge,a_param, n_alphas,
                    max_nb_variables, tol = 1e-7, fit_intercept = False):

    beta_Ridge = ComputeRidgeCoef(X, y, alpha_ridge,
                                  fit_intercept = fit_intercept, tol =  tol)
    alpha_th = ThRR_grid(beta_Ridge, n_alphas, max_nb_variables)


    coefs_ThRR = RidgePath(X, y, alpha_ridge, alpha_th, fit_intercept = fit_intercept , tol =  tol)
    idx_list, idx_size = Support(coefs_ThRR)
    coef_ThRRAVp, _, idx_ThRRAVp , support_th_for = AVp(X, y,
                         idx_list, idx_size, a_param)
    return coef_ThRRAVp, idx_ThRRAVp


def ThRRAVp_for_display(X, y, alpha_ridge,a_param, n_alphas,
                    max_nb_variables, tol = 1e-7, fit_intercept = False):

    beta_Ridge = ComputeRidgeCoef(X, y, alpha_ridge,
                                  fit_intercept = fit_intercept, tol =  tol)
    alpha_th = ThRR_grid(beta_Ridge, n_alphas, max_nb_variables)
    coefs_ThRR = RidgePath(X, y, alpha_ridge, alpha_th, fit_intercept = fit_intercept , tol =  tol)
    idx_list, idx_size = Support(coefs_ThRR)
    idx_list_ordered, idx_size_order, idx_re_order  = ListForOrdering_fast(
                                            idx_list, idx_size)
    coef_ThRRAVp,_, idx_tot_for, support_th_for, matrix_test = Av_p_for_display(X,
                                                y, idx_list_ordered,
                                                idx_size_order, a_param, idx_re_order)
    return  matrix_test


def ThRRCV(X, y, alpha_ridge, n_alphas, max_nb_variables, n_jobs=1, cv = 10,
              tol = 1e-7, fit_intercept = False):
    beta_Ridge = ComputeRidgeCoef(X, y, alpha_ridge,
                                  fit_intercept = fit_intercept, tol =  tol)
    alpha_th = ThRR_grid(beta_Ridge, n_alphas, max_nb_variables)
    param_grid = dict(alpha = [alpha_ridge], thresh = alpha_th)
    sr_test = ThRR(tol=tol,fit_intercept = fit_intercept)
    gs = GridSearchCV(sr_test, cv = cv, param_grid = param_grid, n_jobs = n_jobs)
    gs.fit(X, y)
    index_ThRRCV = np.where(alpha_th == gs.best_params_['thresh'])[0]
    coef_ThRRCV = gs.best_estimator_.coef_
    return coef_ThRRCV, index_ThRRCV


def ThRR2DCV(X, y, alpha_ridge2D, n_alphas, max_nb_variables, n_jobs=1, cv = 10,
              tol = 1e-7, fit_intercept = False):

    param_grid = []
    n_alphas_ridge = len(alpha_ridge2D)
    for i in range(n_alphas_ridge):
        beta_Ridge = ComputeRidgeCoef(X, y, alpha_ridge2D[i],
                                      fit_intercept = fit_intercept, tol =  tol)
        alpha_th = ThRR_grid(beta_Ridge, n_alphas, max_nb_variables)
        param_grid.append(dict(alpha = [alpha_ridge2D[i]],
                                            thresh = alpha_th))

    sr_test = ThRR(tol=tol,fit_intercept = fit_intercept)
    gs = GridSearchCV(sr_test, cv = cv, param_grid = param_grid, n_jobs = n_jobs)
    gs.fit(X, y)
    #    index_ThRRCV = np.where(alpha_th == gs.best_params_['thresh'])[0]
    index_ThRRCV = gs.best_params_
    coef_ThRRCV = gs.best_estimator_.coef_
    return coef_ThRRCV, index_ThRRCV


def LSThRRCV(X, y, alpha_ridge, n_alphas, max_nb_variables, n_jobs=1, cv = 10,
                tol = 1e-7, fit_intercept = False, eps_machine=1e-12):
    beta_Ridge = ComputeRidgeCoef(X, y, alpha_ridge,
                                  fit_intercept = fit_intercept, tol =  tol)
    alpha_th = ThRR_grid(beta_Ridge, n_alphas, max_nb_variables)
    param_grid = dict(alpha = [alpha_ridge], thresh = alpha_th)
    sr_test = LSThRR(tol=tol,fit_intercept = fit_intercept,eps_machine = eps_machine)
    gs = GridSearchCV(sr_test, cv = cv, param_grid = param_grid, n_jobs = n_jobs)
    gs.fit(X, y)
    index_LSThRRCV= np.where(alpha_th == gs.best_params_['thresh'])[0]
    coef_LSThRRCV = gs.best_estimator_.coef_
    return coef_LSThRRCV, index_LSThRRCV


def LSThRR2DCV(X, y, alpha_ridge2D, n_alphas, max_nb_variables, n_jobs=1, cv = 10,
              tol = 1e-7, fit_intercept = False):

    param_grid = []
    n_alphas_ridge = len(alpha_ridge2D)
    for i in range(n_alphas_ridge):
        beta_Ridge = ComputeRidgeCoef(X, y, alpha_ridge2D[i],
                                      fit_intercept = fit_intercept, tol =  tol)
        alpha_th = ThRR_grid(beta_Ridge, n_alphas, max_nb_variables)
        param_grid.append(dict(alpha = [alpha_ridge2D[i]],
                                            thresh = alpha_th))

    sr_test = LSThRR(tol=tol,fit_intercept = fit_intercept)
    gs = GridSearchCV(sr_test, cv = cv, param_grid = param_grid, n_jobs = n_jobs)
    gs.fit(X, y)
    #    index_ThRRCV = np.where(alpha_th == gs.best_params_['thresh'])[0]
    index_ThRRCV = gs.best_params_
    coef_ThRRCV = gs.best_estimator_.coef_
    return coef_ThRRCV, index_ThRRCV


def ThRR2DAVp(X, y, alpha_ridge2D, a_param, n_alphas,
                    max_nb_variables, tol = 1e-7, fit_intercept = False):

    coefs_ThRR2D = RidgePath2D(X, y, alpha_ridge2D, n_alphas, max_nb_variables,
              fit_intercept = fit_intercept , tol =  tol)

    idx_list, idx_size = Support(coefs_ThRR2D)
    coef_ThRR2DAVp, _, idx_ThRR2DAVp , support_th_for = AVp(X, y,
                         idx_list, idx_size, a_param)
    return coef_ThRR2DAVp, idx_ThRR2DAVp


def LassoCV_joe(X, y, alpha_grid, n_jobs=1, cv = 10, max_iter = 10000,
                tol = 1e-7, fit_intercept = False):
    param_grid = dict(alpha = alpha_grid)
    sr_test = Lasso(max_iter = max_iter, tol = tol, fit_intercept = fit_intercept)
    gs = GridSearchCV(sr_test, cv = cv, param_grid = param_grid, n_jobs = n_jobs)
    gs.fit(X, y)
    index_LassoCV = np.where(alpha_grid == gs.best_params_['alpha'])[0]
    coef_LassoCV = gs.best_estimator_.coef_
    return coef_LassoCV, index_LassoCV


def LSLassoCV_joe(X, y, alpha_grid, n_jobs=1, cv = 10, max_iter = 10000,
                tol = 1e-7, fit_intercept = False):
    param_grid = dict(alpha = alpha_grid)
    sr_test = LSLasso(max_iter = max_iter, tol = tol, fit_intercept = fit_intercept)
    gs = GridSearchCV(sr_test, cv = cv, param_grid = param_grid, n_jobs = n_jobs)
    gs.fit(X, y)
    index_LSLassoCV = np.where(alpha_grid == gs.best_params_['alpha'])[0]
    coef_LSLassoCV = gs.best_estimator_.coef_
    return coef_LSLassoCV, index_LSLassoCV


def LassoIC_joe(X, y, coefs_Lasso,eps_machine=1e-12, method_type='bic'):
    n_samples=X.shape[0]
    R = y[:, np.newaxis] - np.dot(X, coefs_Lasso) # residuals
    mean_squared_error = np.mean(R ** 2, axis=0)
    df = Support(coefs_Lasso,eps_machine)[1]
    if method_type=='bic':
        K = np.log(n_samples) # AIC
    elif method_type == 'aic':
     K = 2 # AIC
    else:
        raise ValueError('criterion should be either bic or aic')
    criterion=K*df + n_samples * np.log(mean_squared_error)
    index_LassoIC = np.argmin(criterion)
    coef_LassoIC =coefs_Lasso[:, index_LassoIC]

    return coef_LassoIC, index_LassoIC