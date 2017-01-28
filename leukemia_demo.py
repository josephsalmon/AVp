# -*- coding: utf-8 -*-
"""
Created on Fri Sept 25 11:08:33 2015

@author: jo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # for plots
from matplotlib import rc
from sklearn import preprocessing
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from general_functions import LSLassoCV_joe, LassoCV_joe, LassoAVp, \
    LassoIC_joe, LassoBIC
from noise_estimation_functions import SqrtLasso_SZ
import time
from lasso_scenarii import ScenarioEquiCor
from sklearn.datasets.mldata import fetch_mldata
from scipy import sparse
import seaborn as sns
from general_functions import QAgg

# from sklearn.grid_search import GridSearchCV

params = {'axes.labelsize': 30,
          'text.fontsize': 30,
          'legend.fontsize': 30,
          'xtick.labelsize': 30,
          'ytick.labelsize': 30,
          'text.usetex': True,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)

sns.set_style("white")
# sns.set_context("poster", font_scale=1)
sns.set_palette("colorblind")
rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)

##############################################################################
# Parameters settings

# n_samples = 40
# n_features =50
s = 10         # sparsity index = 5
sig_noise = 1  # np.sqrt(5*6)     # sig_noise = 0.1
rho = 0.5   # rho = 0.1
n_confounding_var = 10
scenario = 'ScenarioEquiCor'
# scenario = 'ScenarioPowDecayCor'
# other scenarii:
# 'ScenarioMEG'
# 'ScenarioEquiCor'
# 'ScenarioShaoDengI'
# 'ScenarioEquiCor'
# 'ScenarioShaoDengI'
# 'ScenarioPowDecayCor
noise_type = 'Normal'  # 'Laplace', 'Normal'
tol = 1e-7  # tol for the lasso type method. not for the Sqrt Lasso.
max_iter = 1000  # maximum iteration in coordinate descent algorithms
n_alphas = 100

avp_cste = 1  # 1.75
fit_intercept = False
eps_support = 1e-12
eps_lasso_grid = 1e-3
sig_precision = 1e-2  # parameter to stop sig_hat computation
max_iter_sig = 20  # weird cycles... no convergence?


do_Lasso = True  # False
do_ThRR = False  # True
saving = True    # True

dataset_id = 1
if dataset_id == 0:  # riboflavin dataset
    train_df = pd.read_csv("../../RiboflavinData/DataSets/riboflavin.csv",
                           index_col=0,)
    train_df = train_df.transpose()
    train_df.columns = train_df.columns
    y = train_df['q_RIBFLV']
    X = train_df.drop(train_df.columns[[0]], axis=1)
    y = y.as_matrix()
    X = X.as_matrix()


elif dataset_id == 1:  # riboflavin dataset
    dataset_name = 'leukemia'
    # dataset_name = 'arcene nips'
    # dataset_name = 'dorothea'
    # dataset_name = 'gisette'
    data = fetch_mldata(dataset_name)
    X = data.data
    y = data.target
    X = X.astype(float)
    y = y.astype(float)
    # y /= linalg.norm(y)

    if sparse.issparse(X):
        pass
        # col_norms = np.array([linalg.norm(x.toarray()) for x in X.T])
        # inplace_column_scale(X, 1. / col_norms)
    else:
        X /= np.sqrt(np.sum(X ** 2, axis=0))  # Standardize data
        mask = np.sum(np.isnan(X), axis=0) == 0
        if np.any(mask):
            X = X[:, mask]

elif dataset_id == -1:
    y, beta, X = ScenarioEquiCor(n_samples=200, n_features=1000,
                                 sig_noise=1,
                                 rho=0.5, s=10,
                                 normalize=True,
                                 noise_type=noise_type)


X /= np.sqrt(np.sum(X ** 2, axis=0))

print X.shape


def error_l2(beta, X_test, y_test):
    return metrics.mean_squared_error(np.dot(X_test, beta), y_test)

cv_fold = 10
nb_replica = 1

error_l2_mat = np.zeros([nb_replica, 5])
time_l2_mat = np.zeros([nb_replica, 5])

for i in range(nb_replica):
    np.random.seed(seed=i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.46)
    print y_test.shape
    std_scale = preprocessing.StandardScaler().fit(X_train)

    X_train_std = X_train
    X_test_std = X_test

    n_samples, n_features = X_train_std.shape

    # X_train_std = std_scale.transform(X_train)
    # X_test_std = std_scale.transform(X_test)

    alpha_grid = _alpha_grid(X_train_std, y_train, n_alphas=n_alphas,
                             l1_ratio=1.0,
                             fit_intercept=False, eps=eps_lasso_grid)
    # EWA
    start_lassobicb = time.time()
    coef_ini = np.zeros(n_features)
    alpha_sz = np.sqrt(2 * np.log(n_features) / n_samples)
    lasso_tol = 1e-1
    sig_precision = 1e-2  # parameter to stop sig_hat computation
    coef_SqrtLasso_SZ, sig_hat_SZ = SqrtLasso_SZ(X, y, alpha_sz, 20,
                                                 sig_precision,
                                                 lasso_tol, coef_ini)
    a_param_estimated = 14 * sig_hat_SZ ** 2
    coef_lassobicb, index_LassoBIC = LassoBIC(X, y, alpha_grid,
                                              max_iter, tol,
                                              a_param_estimated)
    end_lassobicb = time.time()
    timebicb = end_lassobicb - start_lassobicb
    print('BIC (Bellec) done')

    print('Standard done')
    start_lassobic = time.time()
    coef_lassobic, _ = LassoIC_joe(X_train_std, y_train, alpha_grid,
                                   tol=tol, fit_intercept=False,
                                   method_type='bic')
    end_lassobic = time.time()
    timing_lassobic = end_lassobic - start_lassobic
    print('BIC done')

    ###########################################################################
    # AVp-procedure: LassoAVp with estimated noise

    start_lassocv = time.time()
    coef_lassocv, _ = LassoCV_joe(X_train_std, y_train, alpha_grid, n_jobs=1,
                                  cv=cv_fold, max_iter=max_iter,
                                  tol=tol, fit_intercept=False)
    end_lassocv = time.time()
    timing_lassocv = end_lassocv - start_lassocv

    start_lslassocv = time.time()
    coef_lslassocv, _ = LSLassoCV_joe(X_train_std, y_train, alpha_grid,
                                      n_jobs=1,
                                      cv=cv_fold, max_iter=max_iter,
                                      tol=tol, fit_intercept=False)
    end_lslassocv = time.time()
    timing_lslassocv = end_lslassocv - start_lslassocv
    print('CV done')

    # AVp
    start_lassoavp = time.time()
    coef_ini = np.zeros(n_features)
    alpha_sz = np.sqrt(2 * np.log(n_features) / n_samples)
    lasso_tol = 1e-1
    sig_precision = 1e-2  # parameter to stop sig_hat computation
    coef_SqrtLasso_SZ, sig_hat_SZ = SqrtLasso_SZ(X, y, alpha_sz, 20,
                                                 sig_precision,
                                                 lasso_tol, coef_ini)
    a_param_estimated = avp_cste * sig_hat_SZ ** 2
    coef_LassoAVp, index_LassoAVp = LassoAVp(X, y, alpha_grid,
                                             max_iter, tol,
                                             a_param_estimated)
    end_lassoavp = time.time()
    timeavp = end_lassoavp - start_lassoavp

    ###########################################################################
    # Q-Aggregation with sigma knowledge
    start_Qagg = time.time()
    y_QAgg = QAgg(X, y, alpha_grid, max_iter, tol, a_param_estimated)
    end_Qagg = time.time()
    timing_Qagg = end_Qagg - start_Qagg
    ###########################################################################

    print('AVp done')

    print "AVp       ", error_l2(coef_LassoAVp, X_test_std, y_test),\
          'time', timeavp
    print "LassoCV  ", error_l2(coef_lassocv, X_test_std, y_test), \
          'time', timing_lassocv
    print "LslassoCV", error_l2(coef_lslassocv, X_test_std, y_test), \
          'time', timing_lslassocv
    print "LassoBIC ", error_l2(coef_lassobic, X_test_std, y_test), \
          'time', timing_lassobic
    print "LSLassoBIC", error_l2(coef_lassobicb, X_test_std, y_test),\
          'time', timebicb
    print "Qagg",

    if dataset_id == -1:

        print "----"

        print "AVpr       ", error_l2(coef_LassoAVp, X_test_std,
                                      np.dot(X_test, beta)), 'time', timeavp
        print "LassoCV  ", error_l2(coef_lassocv, X_test_std,
                                    np.dot(X_test, beta)),
        'time', timing_lassocv
        print "LslassoCV", error_l2(coef_lslassocv, X_test_std,
                                    np.dot(X_test, beta)), 'time',
        timing_lslassocv
        print "LassoBIC ", error_l2(coef_lassobic, X_test_std,
                                    np.dot(X_test, beta)), 'time',
        timing_lassobic
        print "LSLassoBIC       ", error_l2(coef_lassobicb, X_test_std,
                                      np.dot(X_test, beta)), 'time', timebicb

    error_l2_mat[i, :] = [error_l2(coef_LassoAVp, X_test_std, y_test),
                          error_l2(coef_lassobicb, X_test_std, y_test),
                          error_l2(coef_lassocv, X_test_std, y_test),
                          error_l2(coef_lslassocv, X_test_std, y_test),
                          error_l2(coef_lassobic, X_test_std, y_test)
                          ]

    time_l2_mat[i, :] = [timeavp, timebicb, timing_lassocv, timing_lslassocv,
                         timing_lassobic]

print error_l2_mat
print time_l2_mat
print np.mean(error_l2_mat, 0)
print np.mean(time_l2_mat, 0)


columns_name = ["AVpr", "lslassoBIC", "lasso (CV = " + str(cv_fold) + ")",
                "lslasso (CV =" + str(cv_fold) + ")", "lassoBIC"]

time_df = pd.DataFrame(time_l2_mat, columns=columns_name)
prediction_df = pd.DataFrame(error_l2_mat, columns=columns_name)

time_df.to_csv("performance/timing_leukemia.csv")
prediction_df.to_csv("performance/prediction_leukemia.csv")


dirname = "srcimages/"
# columns_name = ["AV${}_\text{Pr}$", "LASSO (CV = " + str(cv_fold) + ")",
#                 "LSLASSO (CV =" + str(cv_fold) + ")", "Lasso BIC"]
###############################################################################
# You can start from here if the data has been already generated
###############################################################################
pred_df = pd.read_csv("performance/prediction_leukemia.csv")
tim_df = pd.read_csv("performance/timing_leukemia.csv")

tim_df = tim_df.drop(tim_df.columns[0], axis=1)
pred_df = pred_df.drop(pred_df.columns[0], axis=1)

pred_df.columns = columns_name
tim_df.columns = columns_name


file_format = ".pdf"

ttl = 'Lasso on the Leukemia dataset, ' +\
    r"$p={0}, n={1},r={2}$".format(n_features, n_samples, n_alphas)
fig1 = plt.figure(figsize=(10, 8))
ax = sns.barplot(np.array(columns_name), np.array(tim_df).squeeze(),
                 palette="Paired")
plt.xticks(rotation=45)
plt.ylabel('Timing in seconds')
plt.title(ttl, multialignment='center',fontsize=24)
plt.tight_layout()

# fig1.ylabel('ee')
filename = "Timing_leukemia"
image_name = dirname + filename + file_format
fig1.savefig(image_name)

fig2 = plt.figure(figsize=(10, 8))
ax = sns.barplot(np.array(columns_name), np.array(pred_df).squeeze(),
                 palette="Paired")
plt.ylabel("Prediction Error")
plt.xticks(rotation=45)
plt.title(ttl, multialignment='center', fontsize=24)
plt.tight_layout()
plt.show()

filename = "Prediction_leukemia"
image_name = dirname + filename + file_format
fig2.savefig(image_name)

# print pred_df
# print time_df
