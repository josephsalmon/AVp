# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:39:45 2014

@author: joseph salmon
"""
import time
import seaborn as sns
import numpy as np
import pylab as plt
#import inspect
from lasso_scenarii import ScenarioPowDecayCor #,ScenarioEquiCor
from noise_estimation_functions import SqrtLasso_SZ
from general_tools import PredictionError, EstimationError_2
from general_tools import EstimationError, FalseNegative, FalsePositive
from general_tools import Support, Refitting, ThRR_grid
from general_functions import LSLassoCV_joe, LassoCV_joe, LassoIC_joe
from general_functions import LSThRRCV, ThRRCV, RidgePath,ComputeRidgeCoef
from general_functions import LassoAVp, LassoAVp_for_display, ThRRAVp, ThRRAVp_for_display
from general_functions import QAgg

from sklearn.linear_model import lasso_path, RidgeCV, Ridge
from sklearn.linear_model.coordinate_descent import _alpha_grid

from matplotlib import rc


params = {'text.usetex': True}
plt.rcParams.update(params)
sns.set_context("poster")
sns.set_style("white")
sns.set_palette("colorblind")

current_palette = sns.color_palette()
rc('font', **{'family':'serif', 'serif':['Helvetica']})
rc('text', usetex=True)

##############################################################################
# Parameters settings
##############################################################################
np.random.seed(seed=666)
n_samples = 200#204# n_samples=10
n_features = 100#8000# n_features=9
s = 10         # sparsity index = 5
sig_noise = 1#np.sqrt(5*6)     # sig_noise = 0.1
rho = 0.5   # rho = 0.1
n_confounding_var = 10
scenario = 'ScenarioEquiCor' #'ScenarioMEG'#'ScenarioEquiCor' #'ScenarioShaoDengI'#'ScenarioEquiCor'#'ScenarioShaoDengI' #'ScenarioPowDecayCor
noise_type='Normal'#'Laplace', 'Normal'


tol = 1e-7 # tol for the lasso type method. not for the Sqrt Lasso.
max_iter =1000 # maximum iteration in coordinate descent algorithms

n_alphas = 20
eps_lasso_grid = 1e-3


# max number of variables for the ThRR case
max_nb_variables =np.floor(n_features/5)# for the ThRR
alpha_ridge = 1* (n_features)**(0.5)


AVp_constant = 1
fit_intercept = False
eps_support = 1e-12
cv_fold=10

# Trying AVp with the Lasso and/or Thresholded Ridge Regression
do_Lasso = True#False#False#True
do_ThRR = True#True

#Display options:
xaxis_val=np.arange(n_alphas)+1
legend_fontsize =20
title_fontsize = 24
subtitle_fontsize = 22
label_fontsize = 16
fig_size1=20
fig_size2=20

#display in consol some informations
verbose_res = True


##############################################################################
######      DATAT GENERATION
##############################################################################

start_datagen = time.time()
scenario == 'ScenarioPowDecayCor'
# Power decay of correlations: rho^|i-j|
scenario_str = "Correlations with power decay"
y, beta_true, X = ScenarioPowDecayCor(n_samples = n_samples,
                                      n_features = n_features,
                                      sig_noise = sig_noise,
                                      rho = rho, s = s, normalize = True,
                                      noise_type = noise_type)


alpha_grid = _alpha_grid(X, y , n_alphas = n_alphas, l1_ratio = 1.0,
                       fit_intercept = False, eps = eps_lasso_grid)

end_datagen = time.time()
print 'Timing data generation in s.'
print end_datagen- start_datagen
print '  '


##############################################################################
######      CHOICE OF THE AVp CONSTANT: noise estimation
##############################################################################

coef_ini=np.zeros(beta_true.shape)
alpha_SZ=np.sqrt(2*np.log(n_features)/n_samples)
Lasso_tol=1e-1
sig_precision=1e-2 # parameter to stop sig_hat computation update +-= old for sig_precision
#coef_SqrtLasso, sig_hat = SqrtLasso(X, y, 0.7, 1000, 1e-7,coef_ini)


###############################################################################
#######      Noise estimation with the Sca-Lasso (Sun and Zhang)
###############################################################################

start_SqrtLassoPath_SZ = time.time()
coef_SqrtLasso_SZ, sig_hat_SZ = SqrtLasso_SZ(X, y, alpha_SZ , max_iter, sig_precision, Lasso_tol, coef_ini)
end_SqrtLassoPath_SZ = time.time()
timing_SqrtLasso_SZ=end_SqrtLassoPath_SZ - start_SqrtLassoPath_SZ
print 'Timing for sigma_hat computation in s.'
print timing_SqrtLasso_SZ
print '  '
print 'sigma estimation'
print sig_hat_SZ


###############################################################################
#######      constant choice for AVp based on noise estimation
###############################################################################

a_param_estimated = AVp_constant*sig_hat_SZ**2
a_param_true = AVp_constant *sig_noise**2

##############################################################################
######      PART I: LASSO
##############################################################################


print ''
print '###############################################################'
print '###################    LASSO    ###############################'
print '###############################################################'
print ''


if do_Lasso == True:
    ##############################################################################
    # Lasso path
    start_LassoPath = time.time()
    _, coefs_Lasso, _ = lasso_path(X, y, alphas = alpha_grid,
                                   fit_intercept = False,
                                   return_models = False,
                                   max_iter = max_iter, tol = tol)


    end_LassoPath = time.time()
    timing_LassoPath = end_LassoPath- start_LassoPath

    ##############################################################################
    # LassoBIC
    start_LassoBIC = time.time()
    coef_LassoBIC, index_LassoBIC = LassoIC_joe(X, y, coefs_Lasso,
                                                eps_machine=1e-12,
                                                method_type='bic')
    end_LassoBIC  = time.time()
    timing_LassoBIC = end_LassoBIC - start_LassoBIC+timing_LassoPath




    ##############################################################################
    # LSLasso path
    start_LSLassoPath = time.time()
    coefs_LSLasso,_,_ = Refitting(coefs_Lasso,X,y, eps_machine=eps_support)
    end_LSLassoPath = time.time()
    timing_LSLassoPath = end_LSLassoPath- start_LSLassoPath

    ##############################################################################
    # LSLASSO BIC
    start_LSLassoBIC = time.time()
    coef_LSLassoBIC, index_LSLassoBIC = LassoIC_joe(X, y, coefs_LSLasso,
                                                eps_machine=1e-12,
                                                method_type='bic')
    end_LSLassoBIC  = time.time()
    timing_LSLassoBIC = end_LSLassoBIC - start_LSLassoBIC+timing_LassoPath+timing_LSLassoPath


    ##############################################################################
    # AVp-procedure: LassoAVp with true sigma knowledges
    matrix_for = LassoAVp_for_display(X, y, alpha_grid, max_iter, tol,
                                             a_param_true)
    blasso=Support(coefs_Lasso)[1]
    print 'support compared in the process'
    print blasso

    start_LassoAVp = time.time()
    coef_LassoAVp, index_LassoAVp = LassoAVp(X, y, alpha_grid, max_iter, tol,
                                             a_param_true)
    end_LassoAVp = time.time()
    timing_LassoAVp = end_LassoAVp - start_LassoAVp


    ##############################################################################
    # Q-Aggregation with sigma knowledge
    matrix_for = LassoAVp_for_display(X, y, alpha_grid, max_iter, tol,
                                             a_param_true)
    blasso=Support(coefs_Lasso)[1]
    print 'support compared in the process'
    print blasso

    y_QAgg = QAgg(X, y, alpha_grid, max_iter, tol,
                                             a_param_true)
    pred_error_QAgg = np.sum((y_QAgg - np.dot(X, beta_true)) ** 2) / n_samples


    ##############################################################################
    # AVp-procedure: LassoAVp with estimated noise

    start_LassoAVp_sighat = time.time()
    coef_LassoAVp_sighat, index_LassoAVp_sighat = LassoAVp(X, y, alpha_grid, max_iter, tol,
                                             a_param_estimated)
    end_LassoAVp_sighat = time.time()
    timing_LassoAVp_sighat = end_LassoAVp_sighat - start_LassoAVp_sighat+timing_SqrtLasso_SZ


    ##############################################################################
    # CV for the Lasso: LassoCV  (SCIKIT LEARN 0.14)
    start_LassoCV = time.time()
    coef_LassoCV, index_LassoCV = LassoCV_joe(X, y, alpha_grid, n_jobs=1,
                                               cv = cv_fold, max_iter = max_iter,
                                               tol = tol, fit_intercept = False)
    end_LassoCV = time.time()
    timing_LassoCV = end_LassoCV-start_LassoCV

    ##############################################################################
    # CV of (Lasso + LS): LSLassoCV (SCIKIT LEARN 0.15)
    start_LSLassoCV = time.time()

    coef_LSLassoCV, index_LSLassoCV = LSLassoCV_joe(X, y, alpha_grid, n_jobs=1,
                                               cv = cv_fold, max_iter = max_iter,
                                               tol = tol, fit_intercept = False)
    end_LSLassoCV = time.time()
    timing_LSLassoCV = end_LSLassoCV-start_LSLassoCV




    ###########################################################################
    # print results:
    ###########################################################################

    if verbose_res == True:

        print '###################    SPARSITY index     #####################'
        print
        print ['True sparsity', 'CV', 'BIC', 'LSBIC', 'CV+LS', 'LSLassoCV', 'AVp','AVp_sighat', 'Sca-Lasso']
        print [s, Support(coef_LassoCV)[0].shape[0],
              Support(coef_LassoBIC)[0].shape[0],
              Support(coef_LSLassoBIC)[0].shape[0],
              Support(coef_LSLassoCV)[0].shape[0],
              Support(coef_LassoAVp)[0].shape[0],
              Support(coef_LassoAVp_sighat)[0].shape[0],
              Support(coef_SqrtLasso_SZ)[0].shape[0]]
        print ''

        print '###################    PREDICTION ERROR    ####################'
        print ''
        print ['CV', 'BIC', 'LSBIC', 'LSLassoCV','AVp','AVp_sighat', 'Sca-Lasso']
        print '%.5f' %PredictionError(X, coef_LassoCV, beta_true),\
              '%.5f' %PredictionError(X, coef_LassoBIC, beta_true),\
              '%.5f' %PredictionError(X, coef_LSLassoBIC, beta_true),\
              '%.5f' %PredictionError(X, coef_LSLassoCV, beta_true),\
              '%.5f' %PredictionError(X, coef_LassoAVp, beta_true),\
              '%.5f' %PredictionError(X, coef_LassoAVp_sighat, beta_true),\
              '%.5f' %PredictionError(X, coef_SqrtLasso_SZ, beta_true)
        print ''

        print '########################    TIMING (in s.)   ##################'
        print ''
        print " CV, BIC, LSBIC, LSLassoCV, AVp, AVp_sighat, Sca-Lasso"
        print '%.5f' %timing_LassoCV,\
              '%.5f' %timing_LassoBIC,\
              '%.5f' %timing_LSLassoBIC,\
              '%.5f' %timing_LSLassoCV,\
              '%.5f' %timing_LassoAVp,\
              '%.5f' %timing_LassoAVp_sighat,\
              '%.5f' %timing_SqrtLasso_SZ
        print ''


    else:
        print 'No display'


    ###########################################################################
    # plot Prediction and Estimation error
    ###########################################################################

    rc('lines', linewidth = 4)
    fig3 = plt.figure(figsize = (fig_size1,fig_size2), dpi = 90)
    xlabels = 'index of the supports'
    plt.suptitle(scenario_str + r" $(p={0}, n={1}, s={2}, \sigma={3}, \rho={4}, a={5}\sigma^2,r={6})$".format(n_features,
                 n_samples, s, sig_noise, rho, AVp_constant, n_alphas),
                 fontsize = title_fontsize, multialignment = 'center')

    ax1 = fig3.add_subplot(321)
    ax1.plot(xaxis_val,PredictionError(X, coefs_Lasso, beta_true), '--', color=current_palette[0], label = "lasso")
    ax1.plot(xaxis_val,PredictionError(X, coefs_LSLasso, beta_true), ':',color=current_palette[1], label = "lslasso")
    ax1.plot(index_LassoCV+1, PredictionError(X, coef_LassoCV, beta_true),
             'v',color=current_palette[0], markersize = 20, label = "lassoCV")
    ax1.plot(index_LSLassoCV+1, PredictionError(X, coef_LSLassoCV, beta_true),
             color='blue', marker='^', linestyle='None', markersize = 20, label = "lslassoCV")
    ax1.plot(index_LassoAVp+1, PredictionError(X, coef_LassoAVp, beta_true),
             'rD', markersize = 9, label = r"lassoAV${}_{\mbox{{p}}}$")
    ax1.plot(index_LassoAVp_sighat+1, PredictionError(X, coef_LassoAVp_sighat, beta_true),
             '*',color=current_palette[2], markersize = 24, label = r"lassoAV${}_{\mbox{{p}}}$ with $\hat\sigma$")
    ax1.set_yscale('symlog')
    plt.xlabel(xlabels, fontsize = label_fontsize)
    ax1.set_ylim(0,1.3*np.max(PredictionError(X, coefs_Lasso, beta_true)))
    plt.ylabel(r"$\|X \beta -X \hat \beta \|_2^2/n$",fontsize = label_fontsize )
    plt.title('Prediction Error', fontsize = subtitle_fontsize )


    ax2 = fig3.add_subplot(322)
    ax2.plot(xaxis_val,EstimationError_2(coefs_Lasso, beta_true), '--', color=current_palette[0], label = "lasso")
    ax2.plot(xaxis_val,EstimationError_2(coefs_LSLasso, beta_true),
             ':',color=current_palette[1], label = "lslasso")
    ax2.plot( index_LassoCV+1, EstimationError_2(coef_LassoCV, beta_true),
             'v',color=current_palette[0], label = "lassoCV", markersize = 20)
    ax2.plot(index_LSLassoCV+1, EstimationError_2(coef_LSLassoCV, beta_true),
             color='blue', marker='^', linestyle='None',markersize = 20, label = "lslassoVB")
    ax2.plot(index_LassoAVp+1, EstimationError_2(coef_LassoAVp, beta_true),
             'rD', markersize = 9, label = r"lassoAV${}_{\mbox{{p}}}$")
    ax2.plot(index_LassoAVp_sighat+1, EstimationError_2(coef_LassoAVp_sighat, beta_true),
             '*',color=current_palette[2], markersize = 24, label = r"lassoAV${}_{\mbox{{p}}}$ with $\hat\sigma$")
    plt.title('Estimation Error ($\ell_2$)', fontsize = subtitle_fontsize )
    plt.xlabel(xlabels, fontsize = label_fontsize)
    plt.ylabel(r"$\|\beta - \hat \beta \|_2$",fontsize = label_fontsize )

    ax3 = fig3.add_subplot(323)
    ax3.plot(xaxis_val,EstimationError(coefs_Lasso, beta_true), '--', color=current_palette[0], label = "lasso")
    ax3.plot(xaxis_val,EstimationError(coefs_LSLasso, beta_true), ':',color=current_palette[1], label = "lasso")
    ax3.plot(index_LassoCV+1, EstimationError(coef_LassoCV, beta_true),
             'v',color=current_palette[0], label = "lassoCV", markersize = 20)
    ax3.plot(index_LSLassoCV+1, EstimationError(coef_LSLassoCV, beta_true),
             color='blue', marker='^', linestyle='None' ,markersize = 20, label = "lslassoCV")
    ax3.plot(index_LassoAVp+1, EstimationError(coef_LassoAVp, beta_true),
             'rD', markersize = 9, label = r"lassoAV${}_{\mbox{{p}}}$")
    ax3.plot(index_LassoAVp_sighat+1, EstimationError(coef_LassoAVp_sighat, beta_true),
             '*',color=current_palette[2], markersize = 24, label = r"lassoAV${}_{\mbox{{p}}}$ with $\hat\sigma$")
    plt.xlabel(xlabels, fontsize = label_fontsize)
    ylabels = r"$\|\beta - \hat \beta \|_\infty$"
    plt.ylabel(ylabels, fontsize = label_fontsize )
    plt.title('Estimation Error ($\ell_\infty$)', fontsize = subtitle_fontsize )

    ax4 = fig3.add_subplot(324)
    ax4.plot(xaxis_val,FalsePositive(coefs_Lasso, beta_true), '--', color=current_palette[0], label = "lasso")
    ax4.plot(xaxis_val,FalsePositive(coefs_LSLasso, beta_true), ':',color=current_palette[1], label = "lslasso")
    ax4.plot( index_LassoCV+1, FalsePositive(coef_LassoCV, beta_true),
             'v',color=current_palette[0], label = "lassoCV", markersize = 20)
    ax4.plot(index_LSLassoCV+1, FalsePositive(coef_LSLassoCV, beta_true),
             color='blue', marker='^', linestyle='None', markersize = 20, label = "lslassoCV")
    ax4.plot(index_LassoAVp+1, FalsePositive(coef_LassoAVp, beta_true),
             'rD', markersize = 9, label = r"lassoAV${}_{\mbox{{p}}}$")
    ax4.plot(index_LassoAVp_sighat+1, FalsePositive(coef_LassoAVp_sighat, beta_true),
             '*',color=current_palette[2], markersize = 24, label =r"lassoAV${}_{\mbox{{p}}}$ with $\hat\sigma$")
    plt.xlabel(xlabels, fontsize = label_fontsize)
    ylabels = r"Number of false positive"
    plt.ylabel(ylabels, fontsize = label_fontsize)
    plt.title('False positive', fontsize = subtitle_fontsize )

    ax5 = fig3.add_subplot(325)
    ax5.plot(xaxis_val,FalseNegative(coefs_Lasso, beta_true), '--', color=current_palette[0], label = "lasso")
    ax5.plot(xaxis_val,FalseNegative(coefs_LSLasso, beta_true), ':',color=current_palette[1], label = "lslasso")
    ax5.plot( index_LassoCV+1, FalseNegative(coef_LassoCV, beta_true),
             'v',color=current_palette[0], markersize = 20, label = "lassoCV")
    ax5.plot(index_LSLassoCV+1, FalseNegative(coef_LSLassoCV, beta_true),
             color='blue', marker='^', linestyle='None', markersize = 20, label = "lslassoCV")
    ax5.plot(index_LassoAVp+1, FalseNegative(coef_LassoAVp, beta_true),
             'rD', markersize = 9, label = r"lassoAV${}_{\mbox{{p}}}$")
    ax5.plot(index_LassoAVp_sighat+1, FalseNegative(coef_LassoAVp_sighat, beta_true),
             '*',color=current_palette[2], markersize = 24, label =r"lassoAV${}_{\mbox{{p}}}$ with $\hat\sigma$")
    plt.xlabel(xlabels, fontsize = label_fontsize)
    ylabels = r"Number of false negative"
    plt.ylabel(ylabels, fontsize = label_fontsize)
    plt.title('False negative', fontsize = subtitle_fontsize )
    plt.legend(numpoints = 1)


    ax6 = fig3.add_subplot(326)
    ax6.plot(coef_LassoCV, 'g', label = "lassoCV")
    ax6.plot(coef_LSLassoCV, 'b--', label = "lslassoCV")
    ax6.plot(coef_LassoAVp_sighat, 'r', label = r"lassoAV${}_{\mbox{{p}}}$ with $\hat\sigma$")
    plt.xlabel('Coefficients indices',fontsize = label_fontsize )
    plt.title('Coefficients amplitudes', fontsize = subtitle_fontsize )
    plt.legend(prop={'size':legend_fontsize})
    plt.subplots_adjust(hspace = 0.2, wspace = 0.2, bottom = 0.1 )
    plt.show()



###############################################################################
######      PART II: RIDGE + THRESHOLDING
###############################################################################


print ''
print '###############################################################'
print '###################    RIDGE    ###############################'
print '###############################################################'
print ''



if do_ThRR == True:
    ###########################################################################
    # ThRR path
    ###########################################################################
    start_path_ThRR = time.time()
    beta_Ridge = ComputeRidgeCoef(X, y, alpha_ridge,
                                  fit_intercept = fit_intercept, tol =  tol)
    alpha_th = ThRR_grid(beta_Ridge, n_alphas, max_nb_variables)
    coefs_ThRR = RidgePath(X, y, alpha_ridge, alpha_th,
              fit_intercept = fit_intercept , tol =  tol)

    end_path_ThRR = time.time()

    coefs_ThRR_LS, _, _ = Refitting(coefs_ThRR, X, y)
    timing_path_ThRR = end_path_ThRR-start_path_ThRR
    [a,b]=Support(coefs_ThRR)
    print 'support compared in the process'
    print b


    RidgeCV_pred = RidgeCV(fit_intercept = False, alphas=[0.01,0.1, 1.0, 10.0, 100,1000,10000])
    RidgeCV_pred.fit(X, y)
    coef_RidgeCV = RidgeCV_pred.coef_


    ###########################################################################
    # ThRRCV
    ###########################################################################
    start_ThRRCV = time.time()
    coef_ThRRCV, index_ThRRCV = ThRRCV(X, y, alpha_ridge,
                                             n_alphas, max_nb_variables, n_jobs=1, cv = cv_fold)
    end_ThRRCV = time.time()
    timing_ThRRCV = end_ThRRCV-start_ThRRCV


    ###########################################################################
    # SoftRidgeCV +LS
    ###########################################################################
    coef_ThRRCVpLS = np.reshape(coefs_ThRR_LS[..., index_ThRRCV],-1)

    ###########################################################################
    # LSThRRCV
    ###########################################################################
    start_LSThRRCV = time.time()
    coef_LSThRRCV, index_LSThRRCV = LSThRRCV(X, y, alpha_ridge,
                                                      n_alphas, max_nb_variables, n_jobs=1, cv = cv_fold)
    end_LSThRRCV = time.time()
    timing_LSThRRCV = end_LSThRRCV - start_LSThRRCV

    ###########################################################################
    # ThRRAVp
    ###########################################################################
    start_RidgeDispaly = time.time()
    matrix_for = ThRRAVp_for_display(X, y, alpha_ridge,
                                     a_param_true, n_alphas,
                                     max_nb_variables, tol = tol,
                                     fit_intercept = fit_intercept)
    end_RidgeDispaly = time.time()
    timing_ThRRDisplay = end_RidgeDispaly -start_RidgeDispaly

    start_ThRRAVp_partial = time.time()
    coef_ThRRAVp, index_ThRRAVp = ThRRAVp(X, y, alpha_ridge,
                                      a_param_true, n_alphas,
                                      max_nb_variables,
                                      tol = tol,
                                      fit_intercept = fit_intercept)
    end_ThRRAVp_partial = time.time()
    timing_ThRRAVp = end_ThRRAVp_partial-start_ThRRAVp_partial

    ###########################################################################
    # ThRRAVp_sighat
    ###########################################################################
    start_RidgeDispaly = time.time()
    matrix_for = ThRRAVp_for_display(X, y, alpha_ridge,a_param_estimated, n_alphas,
                    max_nb_variables, tol = tol,
                    fit_intercept = fit_intercept)
    end_RidgeDispaly = time.time()
    timing_ThRRDisplay = end_RidgeDispaly -start_RidgeDispaly

    start_ThRRAVp_partial = time.time()
    coef_ThRRAVp_sighat, index_ThRRAVp_sighat = ThRRAVp(X, y, alpha_ridge,
                                      a_param_estimated, n_alphas,
                                      max_nb_variables,
                                      tol = tol,
                                      fit_intercept = fit_intercept)
    end_ThRRAVp_partial = time.time()
    timing_ThRRAVp_sighat = end_ThRRAVp_partial-start_ThRRAVp_partial+timing_SqrtLasso_SZ


    ###########################################################################
    # plot Prediction and Estimation error
    ###########################################################################
    rc('lines', linewidth = 4)

    fig4 = plt.figure(figsize = (fig_size1,fig_size2), dpi = 90)
    xlabels = 'Thresholding index'
    plt.suptitle("Thrr:"+ r" $p={0}, n={1}, s={2}, \sigma={3}, \rho={4},r={6}$".format(n_features,
                 n_samples, s, sig_noise, rho, AVp_constant, n_alphas, alpha_ridge),
                 fontsize = title_fontsize, multialignment = 'center')
    ax1 = fig4.add_subplot(321)
    ax1.plot(xaxis_val,PredictionError(X, coefs_ThRR, beta_true), '--', color=current_palette[0],
             label="thrr")
    ax1.plot(xaxis_val,PredictionError(X, coefs_ThRR_LS, beta_true), ':',color=current_palette[1],
             label="lsthrr")
    ax1.plot(index_ThRRCV+1, PredictionError(X, coef_ThRRCV, beta_true),
             'v',color=current_palette[0], label = "thrrCV", markersize = 20)
    ax1.plot(index_LSThRRCV+1, PredictionError(X, coef_LSThRRCV, beta_true),
             'o',color=current_palette[1], markersize = 20, label = "lsthrrCV")
    ax1.plot(index_ThRRAVp_sighat+1, PredictionError(X, coef_LassoAVp_sighat, beta_true),
             '*',color=current_palette[2], markersize = 24, label = r"$\mbox{thrrAV}_{\mbox{{p}}}$")
    ax1.plot(index_ThRRAVp+1, PredictionError(X, coef_ThRRAVp, beta_true),
             'rD', markersize = 9, label = r"$\mbox{thrrAV}_{\mbox{{p}}}$ with $\hat\sigma$")
    ax1.set_yscale('symlog')
    ax1.set_ylim(0,1.3*np.max(PredictionError(X, coefs_ThRR, beta_true)))
    plt.xlabel(xlabels, fontsize = label_fontsize)
    plt.ylabel(r"$\|X \beta -X \hat \beta \|_2^2/n$")
    plt.title('Prediction Error', fontsize = subtitle_fontsize )


    ax2 = fig4.add_subplot(322)
    ax2.plot(xaxis_val,EstimationError_2(coefs_ThRR, beta_true), '--', color=current_palette[0],
             label = "thrr")
    ax2.plot(xaxis_val,EstimationError_2(coefs_ThRR_LS, beta_true), ':',color=current_palette[1],
             label = "lsthrr")
    ax2.plot(index_ThRRCV+1, EstimationError_2(coef_ThRRCV, beta_true),
             'v',color=current_palette[0], label = "ThRRCV", markersize = 20)
    ax2.plot(index_LSThRRCV+1, EstimationError_2(coef_LSThRRCV, beta_true),
             'o',color=current_palette[1], markersize = 20, label = "lsthrrCV")
    ax2.plot(index_ThRRAVp_sighat+1, EstimationError_2(coef_LassoAVp_sighat, beta_true),
             '*',color=current_palette[2], markersize = 24, label = r"$\mbox{thrrAV}_{\mbox{{p}}}$")
    ax2.plot(index_ThRRAVp+1, EstimationError_2(coef_ThRRAVp, beta_true),
             'rD', markersize = 9, label = r"$\mbox{thrrAV}_{\mbox{{p}}}$ with $\hat\sigma$")
    ax2.set_ylim(0,1.3*np.max(EstimationError_2(coefs_ThRR, beta_true)))
    plt.title('Estimation Error ($\ell_2$)', fontsize = subtitle_fontsize )
    plt.xlabel(xlabels, fontsize = label_fontsize)
    plt.ylabel(r"$\|\beta - \hat \beta \|_2$")


    ax3 = fig4.add_subplot(323)
    ax3.plot(xaxis_val,EstimationError(coefs_ThRR, beta_true), '--', color=current_palette[0], label="thrr")
    ax3.plot(xaxis_val,EstimationError(coefs_ThRR_LS, beta_true), ':',color=current_palette[1],
             label="lsthrr")
    ax3.plot(index_ThRRCV+1, EstimationError(coef_ThRRCV, beta_true),
             'v',color=current_palette[0], label = "thrrCV", markersize = 20)
    ax3.plot(index_LSThRRCV+1, EstimationError(coef_LSThRRCV, beta_true),
             'o',color=current_palette[1], markersize = 20, label = "lsthrrCV")
    ax3.plot(index_ThRRAVp_sighat+1, EstimationError(coef_ThRRAVp_sighat, beta_true),
             '*',color=current_palette[2], markersize = 24, label = r"$\mbox{thrrAV}_{\mbox{{p}}}$")
    ax3.plot(index_ThRRAVp+1, EstimationError(coef_ThRRAVp, beta_true),
             'rD', markersize = 9, label = r"$\mbox{thrrAV}_{\mbox{{p}}}$ with $\hat\sigma$ ")
    ax3.set_ylim(0,1.3*np.max(EstimationError(coefs_ThRR, beta_true)))
    plt.xlabel(xlabels, fontsize = label_fontsize)
    ylabels = r"$\|\beta - \hat \beta \|_\infty$"
    plt.ylabel(ylabels, fontsize = label_fontsize)
    plt.title('Estimation Error ($\ell_\infty$)', fontsize = subtitle_fontsize )


    ax4 = fig4.add_subplot(324)
    ax4.plot(xaxis_val,FalsePositive(coefs_ThRR, beta_true), '--', color=current_palette[0], label = "thrr")
    ax4.plot(xaxis_val,FalsePositive(coefs_ThRR_LS, beta_true),
             ':',color=current_palette[1], label = "lsthrr")
    ax4.plot( index_ThRRCV+1, FalsePositive(coef_ThRRCV, beta_true),
             'v',color=current_palette[0], label = "thrrCV", markersize = 20)
    ax4.plot(index_LSThRRCV+1, FalsePositive(coef_LSThRRCV, beta_true),
             'o',color=current_palette[1], markersize = 20, label = r"lsthrrCV")
    ax4.plot(index_ThRRAVp_sighat+1, FalsePositive(coef_ThRRAVp_sighat, beta_true),
             '*',color=current_palette[2], markersize = 24, label = r"$\mbox{thrrAV}_{\mbox{{p}}}$")
    ax4.plot(index_ThRRAVp+1, FalsePositive(coef_ThRRAVp, beta_true),
             'rD', markersize = 9, label = r"$\mbox{thrrAV}_{\mbox{{p}}}$ with $\hat\sigma$ ")
    plt.xlabel(xlabels, fontsize = label_fontsize)
    ylabels = r"Number of false positive"
    plt.ylabel(ylabels, fontsize = label_fontsize)
    plt.title('False positive', fontsize = subtitle_fontsize )
    ax4.set_ylim(-0.02,1.3*np.max(FalsePositive(coefs_ThRR, beta_true)))


    ax5 = fig4.add_subplot(325)
    ax5.plot(xaxis_val,FalseNegative(coefs_ThRR, beta_true), '--', color=current_palette[0], label = "thrr")
    ax5.plot(xaxis_val,FalseNegative(coefs_ThRR_LS, beta_true),':',color=current_palette[1],
             label = "lsthrr")
    ax5.plot( index_ThRRCV+1, FalseNegative(coef_ThRRCV, beta_true),
             'v',color=current_palette[0], label = "thrrCV", markersize = 20)
    ax5.plot(index_LSThRRCV+1, FalseNegative(coef_LSThRRCV, beta_true),
             'o',color=current_palette[1], markersize = 20,  label = "lsthrrCV")
    ax5.plot(index_ThRRAVp_sighat+1, FalseNegative(coef_ThRRAVp_sighat, beta_true),
             '*',color=current_palette[2], markersize = 24, label = r"$\mbox{thrrAV}_{\mbox{{p}}}$")
    ax5.plot(index_ThRRAVp+1, FalseNegative(coef_ThRRAVp, beta_true),
             'rD', markersize = 9, label = r"$\mbox{thrrAV}_{\mbox{{p}}}$  with $\hat\sigma$ ")
    plt.xlabel(xlabels, fontsize = label_fontsize)
    ax5.set_ylim(-0.3,1.3*np.max(FalseNegative(coefs_ThRR, beta_true)))
    ylabels = r"Number of false negative"
    plt.ylabel(ylabels, fontsize = label_fontsize)
    plt.title('False negative', fontsize = subtitle_fontsize )
    plt.legend(numpoints = 1, fontsize = legend_fontsize)

    ax6 = fig4.add_subplot(326)
    ax6.plot(coef_ThRRCV, '--',color=current_palette[0], label = "thrrCV")
    ax6.plot(coef_LSThRRCV, ':',color=current_palette[1], label = "lsthrrCV")
    ax6.plot(coef_ThRRAVp_sighat, color=current_palette[2], label = r"$\mbox{thrrAV}_{\mbox{{p}}}$ with $\hat\sigma$")
    plt.xlabel('Coefficients',fontsize = label_fontsize)
    plt.title('Recovered signal',fontsize = subtitle_fontsize )
    plt.legend(prop={'size':legend_fontsize})
    plt.subplots_adjust(hspace = 0.2, wspace = 0.2, bottom = 0.1 )
    plt.show()

    ###########################################################################
    # print results
    ###########################################################################
    verbose_res = True
    if verbose_res == True:
        print a[index_ThRRAVp]


        print '####################  SPARSITY index  #########################'
        print ''
        print ['True Sparsity', 'CV', 'ThRRCV+LS', 'LSThRRCV', 'ThRRAVp','ThRRAVp_sighat']
        print [s, Support(coef_ThRRCV)[0].shape[0],
              Support(coef_ThRRCVpLS)[0].shape[0],
              Support(coef_LSThRRCV)[0].shape[0],
              Support(coef_ThRRAVp)[0].shape[0],
              Support(coef_ThRRAVp_sighat)[0].shape[0]]
        print ''
        print '###############################################################'
        print '###################    PREDICTION ERROR    ####################'
        print '###############################################################'
        print ''
        print ['RidgeCV','ThRRCV', 'ThRRCV+LS', 'LSThRRCV', 'ThRRAVp', 'ThRRAVp_sighat']
        print '%.5f' %PredictionError(X, coef_RidgeCV, beta_true),\
               '%.5f' %PredictionError(X, coef_ThRRCV, beta_true),\
               '%.5f' %PredictionError(X, coef_ThRRCVpLS, beta_true),\
               '%.5f' %PredictionError(X, coef_LSThRRCV, beta_true),\
               '%.5f' %PredictionError(X, coef_ThRRAVp, beta_true),\
               '%.5f' %PredictionError(X, coef_ThRRAVp_sighat, beta_true)
        print ''
        print '###############################################################'
        print '###################    TIMING     #############################'
        print '###############################################################'
        print ''
        print "TIMING: ThRRCV, LSThRRCV, ThRRAVp, ThRRAVp_sighat"
        print '%.5f' %timing_ThRRCV,\
              '%.5f' %timing_LSThRRCV,\
              '%.5f' %timing_ThRRAVp,\
              '%.5f' %timing_ThRRAVp_sighat

    else:
        print 'no display'


##############################################################################
# print GLOBAL comaparison:
##############################################################################
if do_Lasso == True & do_ThRR == True:

    print ''
    print '###############################################################'
    print '###################    GLOBAL    ##############################'
    print '###############################################################'
    print ''
    print '#####################   SPARSITY index  #######################'
    print ''
    print ['True s', 'LassoCV', 'LSLassoCV', 'LassoAVp',
           'ThRRCV', 'LSThRRCV', 'ThRRAVp']
    print[s, Support(coef_LassoCV)[0].shape[0],
        Support(coef_LSLassoCV)[0].shape[0],
        Support(coef_LassoAVp_sighat)[0].shape[0],
        Support(coef_ThRRCV)[0].shape[0],
        Support(coef_LSThRRCV)[0].shape[0],
        Support(coef_ThRRAVp_sighat)[0].shape[0]]
    print ''
    print '#####################   PREDICTION ERROR  #####################'
    print ''
    print ['LassoCV', 'LSLassoCV', 'LassoAVp_sighat',
           'ThRRCV', 'LSThRRCV', 'ThRRAVp_sighat']
    print '%.5f' %PredictionError(X, coef_LassoCV, beta_true),\
           '%.5f' %PredictionError(X, coef_LSLassoCV, beta_true),\
           '%.5f' %PredictionError(X, coef_LassoAVp_sighat, beta_true),\
           '%.5f' %PredictionError(X, coef_ThRRCV, beta_true),\
           '%.5f' %PredictionError(X, coef_LSThRRCV, beta_true),\
           '%.5f' %PredictionError(X, coef_ThRRAVp_sighat, beta_true)
    print ''
    print '#####################   TIMING   ##############################'
    print ''
    print 'LassoAVp_sighat, LSLassoCV, ThRRAVp_sighat, LSThRRCV'
    print '%.5f' %timing_LassoAVp_sighat,\
          '%.5f' %timing_LSLassoCV,\
          '%.5f' %timing_ThRRAVp_sighat,\
          '%.5f' %timing_LSThRRCV
