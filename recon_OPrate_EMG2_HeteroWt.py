# -*- coding: utf-8 -*-
"""
Maximum likelihood estimation of the rate-constant map
of the ortho-positronium (OP) lifespan pdf and its
mixture-weight map.

The pdf is assumed to contain a mixture of two EMG distributions
describing a fast deacy process (DA and PP)
and the slower OP decay process.
The rate-constant map of the fast decay is assumed known.

Created on Sat Dec 23, 2023
@author: Chien-Min Kao
"""
import numpy as np
from numba import jit
from scipy.sparse import spdiags
from numpy import linalg as LA

import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('-r',
                    '--run',
                    help='run id',
                    type=int,
                    default=1)
parser.add_argument('-n',
                    '--niters',
                    help='max number of iterations',
                    type=int,
                    default=50)
parser.add_argument('-o',
                    '--to_save',
                    help='iteration number to save outputs',
                    type=str,
                    default="3,4,all-log")
parser.add_argument('-t',
                    '--TOF',
                    help='TOF FWHM (ps)',
                    type=int,
                    default=0)
parser.add_argument('--hdir',
                    help='system matrix directory',
                    type=str,
                    default=".")
parser.add_argument('--wdir',
                    help='working directory',
                    type=str,
                    default=".")
# args = parser.parse_args(["--wdir", "./full", "-r", "0", "-o", "all-log,11,12"])
args = parser.parse_args()
run = args.run
max_iters = args.niters
to_save = [max_iters]
for ss in args.to_save.split(','):
    if ss.lower()=="all":
        to_save.extend(np.arange(max_iters)+1)
    elif ss.lower()=="all-log":
        for e in np.power(10, np.arange(int(np.log10(max_iters))+1)):
            to_save.extend([1*e,2*e,5*e])
    else:
        to_save.extend([int(ss)])
to_save = np.array(to_save)
to_save = np.unique(to_save[to_save<=max_iters])
wdir = "." if args.wdir=="None" else args.wdir
hdir = "." if args.hdir=="None" else args.hdir
if not os.path.exists(wdir): os.makedirs(wdir)

from config import TOF_FWHM as default_TOF_FWHM, getTOFSettings
TOF_FWHM = int(default_TOF_FWHM if (args.TOF<=0) else args.TOF)
TOF_std, N_TOF, TOF_edges, TOF_bins = getTOFSettings(TOF_FWHM)

print (f"OP rate; EMG2, fixed fast rate: {run = }, {max_iters = }, {TOF_FWHM = }ps, {to_save = }")

#-----------------------------------------
# load data and H matrix
from scipy.io import loadmat
datapath = f"{wdir}/{TOF_FWHM}ps/" + str(run) + '/data/'

## read the exactly known maps of the activity and the exactly known
## maps of the mixture weight and rate constant for DA, PP and OP decays
events_raw_filename = datapath + 'events_raw_phantom5.mat'
f_map_true = loadmat(events_raw_filename)['f_map']
f_vec_true = np.reshape(f_map_true, (-1), order='C')
DAwt_map_true, DArate_map_true = loadmat(events_raw_filename)['DA_map']
DAwt_vec_true = np.reshape(DAwt_map_true, (-1), order='C')
DArate_vec_true = np.reshape(DArate_map_true, (-1), order='C')
PPwt_map_true, PPrate_map_true = loadmat(events_raw_filename)['PP_map']
PPwt_vec_true = np.reshape(PPwt_map_true, (-1), order='C')
PPrate_vec_true = np.reshape(PPrate_map_true, (-1), order='C')
OPwt_map_true, OPrate_map_true = loadmat(events_raw_filename)['OP_map']
OPwt_vec_true = np.reshape(OPwt_map_true, (-1), order='C')
OPrate_vec_true = np.reshape(OPrate_map_true, (-1), order='C')
N_pix = np.size(f_vec_true)

# load estimated f map
f_est_path = f"{wdir}/{TOF_FWHM}ps/" + str(run) + "/recon_f_TOF_OSEM/"
f_est_filename = f_est_path + "recon_phantom5.mat"
f_map_est = loadmat(f_est_filename)['f_map_recons']
ROI = f_map_true>0
f_map_est = ROI*f_map_est[np.shape(f_map_est)[0]-1]
f_map_est = np.reshape(f_map_est, (-1), order='C')

f_vec_true = f_map_est

from config import N_det

# read the H matrix as a CSC for efficient computation
# of the likelihood function
H_path = f"{hdir}/h_matrix/"
# H = loadmat(H_path+f"H_TOF_PB_{TOF_FWHM}ps.mat")['H'].tocsc()
H = loadmat(H_path+f"H_TOF_Siddon_{TOF_FWHM}ps.mat")['H'].tocsc()

# read data
events_measured_filename = datapath + f"events_measured_phantom5_{TOF_FWHM}ps.mat"
W = loadmat(events_measured_filename)['W']
i1_511 = np.int32(W[:,0])
i2_511 = np.int32(W[:,1])
k_511  = np.int32(W[:,2])
igamma = np.int32(W[:,3])
DT = W[:,4]
tau_m = []
## correct for the travel-time difference between
## the prompt gamma and annihilation photons
from config import det_xs, det_ys
from constants import c
for i1, i2, k, ig, dtgamma in zip(i1_511, i2_511, k_511, igamma, DT):
    x1, y1 = det_xs[i1], det_ys[i1]
    x2, y2 = det_xs[i2], det_ys[i2]
    xg, yg = det_xs[ig], det_ys[ig]
    D12 = np.sqrt((x2-x1)**2+(y2-y1)**2)
    beta = c*TOF_bins[k]/D12
    x = (x1+x2)/2-beta*(x2-x1)
    y = (y1+y2)/2-beta*(y2-y1)
    alpha_gamma = np.sqrt((xg-x)**2+(yg-y)**2)
    tau_m.append(dtgamma-(D12/2-alpha_gamma)/c)
tau_m = np.array(tau_m)
tau_sigma = np.sqrt(3)*TOF_std/2

# get projection matrix corresponding to the list-mode data
## ck: get vectorized channel index for events from multi-indices
ck = np.ravel_multi_index((i1_511,i2_511,k_511),
                          (N_det,N_det,N_TOF),
                          order='C')
## build H_{c_k,j} all j, Eq. (4)
# Hck = H[ck,:]
## build H_{c+k,j}f_j for all j, Eq. (4)&(5)
# Hf = Hck.dot(spdiags(f_vec_true,0,N_pix,N_pix))
Hf = (H[ck,:]).dot(spdiags(f_vec_true,0,N_pix,N_pix))

#-----------------------------------------
## Theoretically, if an event occur the row of the HF
## associated with it cannot be entirely zero because
## that means the probability of its occurrence is zero,
## an obvious violation. However, due to discretization
## in measurement and in computing H, this can happen.
## With real data, it can also occur due to scatter or random.
## In this case, either f shall be modified.
to_delete1 = np.array(np.nonzero(Hf.sum(axis=1)==0)[0], dtype=np.int32)
to_delete2 = np.array(np.nonzero(tau_m<0)[0], dtype=np.int32)
to_delete = np.unique(np.concatenate([to_delete1, to_delete2], axis=0))
## remove violating events and rebuild ck, Hck and Hf
i1_511 = np.delete(i1_511, to_delete)
i2_511 = np.delete(i2_511, to_delete)
k_511 = np.delete(k_511, to_delete)
tau_m = np.delete(tau_m, to_delete)
ck = np.ravel_multi_index((i1_511,i2_511,k_511),
                          (N_det,N_det,N_TOF),
                          order='C')
# Hck = H[ck,:]
# Hf = Hck.dot(spdiags(f_vec_true,0,N_pix,N_pix))
Hf = (H[ck,:]).dot(spdiags(f_vec_true,0,N_pix,N_pix))
Hf.eliminate_zeros()

### pre-compute transpose, stored as CSR for efficient computing
### the gradient of the likelihood with respect to lambda
Hf_t = Hf.transpose().tocsr()

#-----------------------------------------
# functions for computing the likelihood and its gradient

## numba jit does not recognize sparse matrix.
## Therefore, we produce a wrap for doing matrix operations
## using numpy to allow acceleration by numba.
## The matrix must tbe a CSC matrix.

from scipy.sparse import isspmatrix_csc, isspmatrix_csr
from constants import nan_killer
from utils import erf

def merge2(x1,x2):
    """merge two inputs arrays into one array and return"""
    return np.concatenate((x1,x2))

def split2(x):
    """split and return an input numpy array into two equal-sized arrays"""
    n = np.size(x)//2
    return x[:n], x[n:]

@jit(nopython=True, parallel=True)
def _event_likelihood(data, row_indices, col_ptr, tau, sigma, wts, rates):
    ysum = np.zeros(np.shape(tau)[0], dtype=np.float64)
    nonzero_pixels = np.nonzero(np.logical_and(wts!=0,rates!=0))[0]
    s2 = sigma*sigma
    for col in nonzero_pixels:
        if (col_ptr[col]==col_ptr[col+1]): continue
        rows = row_indices[col_ptr[col]:col_ptr[col+1]]
        wt, rate = wts[col], rates[col]
        t1 = tau[rows]-s2*rate/2
        t2 = (tau[rows]-s2*rate)/(np.sqrt(2)*sigma)
        d = wt*rate*np.exp(-rate*t1)*(1+erf(t2))/2
        ysum[rows] += data[col_ptr[col]:col_ptr[col+1]]*d
    return ysum+nan_killer

def event_likelihood(Hf, tau_m, sigma, fastRates, wts, rates):
    # check if HF is a csc matrix
    if not isspmatrix_csc(Hf):
        raise TypeError("Expect Hf a CSC matrix")
    nr, nc = Hf.shape
    if (nr!=np.size(tau_m)) or (nc!=np.size(fastRates)) or \
        (nc!=np.size(wts) or (nc!=np.size(rates))):
        raise TypeError("Hf mismatch data dimension")
    re  = _event_likelihood(Hf.data, Hf.indices, Hf.indptr, tau_m, sigma, wts, rates)
    re += _event_likelihood(Hf.data, Hf.indices, Hf.indptr, tau_m, sigma, 1-wts, fastRates)
    return re

def loglikelihood(Hf, tau_m, sigma, fastRates, wts, rates):
    return np.sum(np.log(event_likelihood(Hf, tau_m, sigma, fastRates, wts, rates)))

@jit(nopython=True, parallel=True)
def _gradient_loglikehood(data, col_indices, row_ptr, tau, sigma, evlik,
                          fastRates, wts, rates):
    n = np.shape(rates)[0]
    grad_wts = np.zeros(n, dtype=np.float64)
    grad_rates = np.zeros(n, dtype=np.float64)
    s2 = sigma*sigma
    for row in np.arange(n):
        if (row_ptr[row]==row_ptr[row+1]): continue
        cols = col_indices[row_ptr[row]:row_ptr[row+1]]
        hrow = data[row_ptr[row]:row_ptr[row+1]]
        wt, rate = wts[row], rates[row]
        t1 = tau[cols]-s2*rate/2
        t2 = tau[cols]-s2*rate
        t3 = t2/(np.sqrt(2)*sigma)
        h = (1+erf(t3))/2
        d  = (1-rate*t2)*h
        d -= sigma*rate*np.exp(-t3**2)/np.sqrt(2*np.pi)
        d *= wt*np.exp(-rate*t1)/evlik[cols]
        grad_rates[row] = np.sum(hrow*d)
        e = rate*np.exp(-rate*t1)*h
        frate = fastRates[row]
        t1 = tau[cols]-s2*frate/2
        t2 = tau[cols]-s2*frate
        t3 = t2/(np.sqrt(2)*sigma)
        e -= frate*np.exp(-frate*t1)*(1+erf(t3))/2
        e /= evlik[cols]
        grad_wts[row] = np.sum(hrow*e)
    return np.concatenate((grad_wts,grad_rates))

def gradient_loglikehood(Hf, Hf_t, tau_m, sigma, fastRates, wts, rates):
    # check if HF is a CSC matrix
    if not isspmatrix_csc(Hf):
        raise TypeError("Expect Hf a CSC matrix")
    # check if HF_t is a CSR matrix
    if not isspmatrix_csr(Hf_t):
        raise TypeError("Expect Hf_t a CSR matrix") 
    nr, nc = Hf.shape
    nc1, nr1 = Hf_t.shape
    if (nr!=nr1) or (nc!=nc1):
        raise TypeError("Hf and Hf_t dimension mismatch")
    if (nr!=np.size(tau_m)) or (nc!=np.size(fastRates)) or \
        ((nc!=np.size(wts)) or (nc!=np.size(rates))):
        raise TypeError("Hf (Hf_t) and data dimension mistmatch")
    evlik = event_likelihood(Hf, tau_m, sigma, fastRates, wts, rates)
    return _gradient_loglikehood(Hf_t.data, Hf_t.indices, Hf_t.indptr,
                            tau_m, sigma, evlik, fastRates, wts, rates)
    
#---------------------------------------
# solve
from scipy.optimize import minimize
from utils import NMSE

## define function to minimize and its gradient
def fun(x, Hf, Hf_t, tau, sigma, fastRates):
    wts, rates = split2(x)
    return -loglikelihood(Hf, tau, sigma, fastRates, *split2(x))
def jac(x, Hf, Hf_t, tau, sigma, fastRates):
    wts, rates = split2(x)
    return -gradient_loglikehood(Hf, Hf_t, tau, sigma, fastRates, *split2(x))

## prepare initial solution
OPrate_vec_recon = np.ones_like(OPrate_vec_true, dtype=np.float64)*0.5
OPwt_vec_recon = np.ones_like(OPwt_vec_true, dtype=np.float64)*0.4
OPrate_vec_recon[np.logical_not(f_vec_true>0)] = 0
OPwt_vec_recon[np.logical_not(f_vec_true>0)] = 0
fastRate_vec_true = np.array(DArate_vec_true, dtype=np.float64)
N_obj = np.shape(OPrate_map_true)[0]

niter, ftol = 1, 1e-9
rate_map_saved, wt_map_saved = [], []
likelihood_logged, NMSE_logged, grad_logged = [], [], []
crc_logged = []
bv_logged = []
alr_logged = []
std_alr_logged = []

def calculate_crc(oprate_vec_true, rates):
    ind_circle1 = (oprate_vec_true == 0.6)
    ind_circle2 = (oprate_vec_true == 0.4)
    ind_background = (oprate_vec_true == 0.5)
    crc1 = (np.mean(rates[ind_circle1]) / np.mean(rates[ind_background]) - 1) / (0.6 / 0.5 - 1)
    crc2 = (np.mean(rates[ind_circle2]) / np.mean(rates[ind_background]) - 1) / (0.4 / 0.5 - 1)
    mean_crc = np.mean([crc1, crc2])
    return mean_crc

def calculate_bv(oprate_vec_true, rates):
    ind_background = (oprate_vec_true == 0.5)
    bv = np.std(rates[ind_background]) / np.mean(rates[ind_background])
    return bv
    
#OPrate_vec_true = np.reshape(OPrate_map_true, (-1), order='C')
ind_circle1 = (OPrate_vec_true == 0.6)
ind_circle2 = (OPrate_vec_true == 0.4)
ind_background = (OPrate_vec_true == 0.5)

def calculate_abs_log_ratio(rate_vec_true, rates, ind_circle1, ind_circle2, ind_background):
    alr1 = np.absolute(np.log(np.mean(rates[ind_circle1]) / np.mean(rates[ind_background])))
    alr2 = np.absolute(np.log(np.mean(rates[ind_circle2]) / np.mean(rates[ind_background])))
    alr = np.mean([alr1, alr2])
    return alr

def log_results(xc):
    global niter, likelihood_logged, NMSE_logged, grad_logged, crc_logged, bv_logged, alr_logged, std_alr_logged
    global rate_map_saved, wt_map_saved
    global Hf, Hf_t, tau_m, tau_sigma
    global fastRate_vec_true, OPrate_vec_true, OPwt_vec_true
    likelihood_logged.append(-fun(xc, Hf, Hf_t, tau_m, tau_sigma, fastRate_vec_true))
    wts, rates = split2(xc)
    NMSE_logged.append(NMSE(rates, OPrate_vec_true))
    grad_logged.append(LA.norm(jac(xc, Hf, Hf_t, tau_m, tau_sigma, fastRate_vec_true)))
    crc_logged.append(calculate_crc(OPrate_vec_true, rates))
    bv_logged.append(calculate_bv(OPrate_vec_true, rates))
    alr_logged.append(calculate_abs_log_ratio(OPrate_vec_true, rates, ind_circle1, ind_circle2, ind_background))
    std_alr_logged.append(alr_logged[-1]/bv_logged[-1])
    print("step=", niter, "log_likelihood=", likelihood_logged[-1], "NMSE=", NMSE_logged[-1], "crc:", crc_logged[-1], "bv:", bv_logged[-1], "alr:", alr_logged[-1], "std_alr:", std_alr_logged[-1])
    if (niter in to_save):
        a = np.reshape(np.copy(rates),(N_obj,N_obj),order='C')
        rate_map_saved.append(a)
        a = np.reshape(np.copy(wts),(N_obj,N_obj),order='C')
        wt_map_saved.append(a)
    niter += 1
    return

## call minimizer
wt_bound = np.array([(0.2,0.4)]*np.size(OPwt_vec_recon))
rate_bound = np.array([(0,3)]*np.size(OPrate_vec_recon))
res = minimize(fun, merge2(OPwt_vec_recon, OPrate_vec_recon), jac=jac,
    args=(Hf, Hf_t, tau_m, tau_sigma, fastRate_vec_true), \
    method='L-BFGS-B', bounds=merge2(wt_bound, rate_bound), \
    callback=log_results, options={'maxiter': max_iters, 'ftol': ftol, 'disp': 10})
x_recon = res.x

niter -= 1
if (niter not in to_save):
    to_save = np.append(to_save, niter)
    wts, rates = split2(x_recon)
    rate_map_saved.append(np.reshape(rates,(N_obj,N_obj),order='C'))
    wt_map_saved.append(np.reshape(wts,(N_obj,N_obj),order='C'))
saved = np.unique(to_save[to_save<=niter])

#---------------------------------
# save and show results
import os
outpath = f"{wdir}/{TOF_FWHM}ps/" + str(run) + "/recon_OPrate_EMG2_FixedFastRate_with_Metrics/"
if not os.path.exists(outpath): os.makedirs(outpath)
figures_outpath = outpath + "figures/"
if not os.path.exists(figures_outpath): os.makedirs(figures_outpath)

from scipy.io import savemat
savemat(outpath+'recon_phantom5.mat',
        {'iters': saved,
         'OPrate_map_recon': rate_map_saved,
         'OPwt_map_recon': wt_map_saved,
         'OPrate_map_true': OPrate_map_true,
         'OPwt_map_true': OPwt_map_true,
         'likelihood': likelihood_logged,
         'NMSE': NMSE_logged,
         'gradient': grad_logged,
         'CRC': crc_logged,
         'BV': bv_logged,
         'ALR': alr_logged,
         'std_ALR': std_alr_logged})
         
# calculate the change in log likelihood         
diff_logl = [(likelihood_logged[i+1] - likelihood_logged[i])/likelihood_logged[i] for i in range(len(likelihood_logged)-1)]

import matplotlib.pyplot as plt

# plot log likelihood, log NMSE, and log |gradient| vs iteration number
likelihood_logged = np.array(likelihood_logged)
NMSE_logged = np.array(NMSE_logged)
grad_logged = np.array(grad_logged)

# plot log likelihood, log NMSE, and log |gradient| vs iteration number
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
ax1.set_title("likelihood")
ax1.plot(np.arange(niter)+1, likelihood_logged-np.min(likelihood_logged)+0.1)
ax1.set_yscale("log", base=10)

ax2.plot(np.arange(niter)+1, NMSE_logged)
ax2.set_title("NMSE")
ax2.set_yscale("log", base=10)

# Find the minimum NMSE value and its corresponding iteration
min_NMSE_value = np.min(NMSE_logged)
min_NMSE_iteration = np.argmin(NMSE_logged) + 1  # Adding 1 to convert from 0-based index to 1-based index

# Plotting NMSE with a red dot at the minimum point
ax2.plot(np.arange(niter)+1, NMSE_logged)
ax2.scatter(min_NMSE_iteration, min_NMSE_value, color='red')  # Red dot at the minimum point

# Annotate the red dot with its coordinates
ax2.annotate(f'Min: ({min_NMSE_iteration}, {min_NMSE_value:.2e})',
             xy=(min_NMSE_iteration, min_NMSE_value),
             xytext=(min_NMSE_iteration + 2, 1.1*min_NMSE_value),
             color='red')

ax3.plot(np.arange(niter)+1, grad_logged)
ax3.set_yscale("log", base=10)
ax3.set_title("gradient magnitude")

# Find the minimum NMSE value and its corresponding iteration
min_grad_value = np.min(grad_logged)
min_grad_iteration = np.argmin(grad_logged) + 1  # Adding 1 to convert from 0-based index to 1-based index

# Plotting NMSE with a red dot at the minimum point
ax3.plot(np.arange(niter)+1, grad_logged)
ax3.scatter(min_grad_iteration, min_grad_value, color='red')  # Red dot at the minimum point

# Annotate the red dot with its coordinates
ax3.annotate(f'Min: ({min_grad_iteration}, {min_grad_value:.2e})',
             xy=(min_grad_iteration, min_grad_value),
             xytext=(min_grad_iteration + 2, 1.1*min_grad_value),
             color='red')

plt.gcf().canvas.manager.set_window_title("likelihood, NMSE, grad")
plt.savefig(figures_outpath+"likelihood_NMSE_gradient.png")

# plot relative change in likelihood vs iteration number
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6,3))
ax1.set_title("in log10")
ax1.plot(np.arange(2, niter + 1), diff_logl)
ax1.set_yscale("log", base=10)

ax2.set_title("no log trans")
ax2.plot(np.arange(2, niter + 1), diff_logl)

#ax3.set_title("rel diff of log10(logl)")
#ax3.plot(np.arange(2, niter + 1), diff_log10logl)

plt.savefig(figures_outpath+"diff_in_log_likelihood.png")

# plot CRC and BV
crc_logged = np.array(crc_logged)
bv_logged = np.array(bv_logged)
alr_logged = np.array(alr_logged)
std_alr_logged = np.array(std_alr_logged)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
ax1.set_title("CRC")
ax1.plot(np.arange(niter)+1, crc_logged)

ax2.plot(np.arange(niter)+1, bv_logged)
ax2.set_title("BV")

ax3.plot(np.arange(niter)+1, alr_logged)
ax3.set_title("ALR")

ax4.plot(np.arange(niter)+1, std_alr_logged)
ax4.set_title("Std ALR")

# Label maximum point in CRC plot
max_crc_index = np.argmax(crc_logged)+1
max_crc_value = np.max(crc_logged)
ax1.scatter(max_crc_index, max_crc_value, color='red')
ax1.annotate(f'Max: ({max_crc_index}, {max_crc_value:.2f})', xy=(max_crc_index, max_crc_value),
             xytext=(max_crc_index + 2, max_crc_value),
             color='red')
             
# Find the index where CRC curve starts to decrease
decrease_start_index = np.where(np.diff(crc_logged) < 0)[0][0] + 1

# Label the point on the CRC plot
decrease_start_value = crc_logged[decrease_start_index-1]
ax1.scatter(decrease_start_index, decrease_start_value, color='blue')
ax1.annotate(f'Decrease: ({decrease_start_index}, {decrease_start_value:.2f})',
             xy=(decrease_start_index, decrease_start_value),
             xytext=(decrease_start_index + 2, decrease_start_value*0.9),
             color='blue')

# Label minimum point in BV plot
min_bv_index = np.argmin(bv_logged) + 1
min_bv_value = np.min(bv_logged)
ax2.scatter(min_bv_index, min_bv_value, color='red')
ax2.annotate(f'Min: ({min_bv_index}, {min_bv_value:.2f})', xy=(min_bv_index, min_bv_value),
             xytext=(min_bv_index + 2, min_bv_value*1.1),
             color='red')
             
# Label maximum point in ALR plot
max_alr_index = np.argmax(alr_logged) + 1
max_alr_value = np.max(alr_logged)
ax3.scatter(max_alr_index, max_alr_value, color='red')
ax3.annotate(f'Max: ({max_alr_index}, {max_alr_value:.2f})', xy=(max_alr_index, max_alr_value),
             xytext=(max_alr_index + 2, max_alr_value),
             color='red')
             
# Label maximum point in std ALR plot
max_std_alr_index = np.argmax(std_alr_logged) + 1
max_std_alr_value = np.max(std_alr_logged)
ax4.scatter(max_std_alr_index, max_std_alr_value, color='red')
ax4.annotate(f'Max: ({max_std_alr_index}, {max_std_alr_value:.2f})', xy=(max_std_alr_index, max_std_alr_value),
             xytext=(max_std_alr_index + 2, max_std_alr_value),
             color='red')

plt.savefig(figures_outpath+"CRC_BV.png")

# show true f and OP rate and weight
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
t = ax1.imshow(f_map_true)
fig.colorbar(t, ax=ax1, shrink=0.8, pad=0.002)
ax1.set_title('activity')
t = ax2.imshow(OPrate_map_true)
fig.colorbar(t, ax=ax2, shrink=0.8, pad=0.002)
ax2.set_title('OP rate')
t = ax3.imshow(OPwt_map_true)
fig.colorbar(t, ax=ax3, shrink=0.8, pad=0.002)
ax3.set_title('OP weight')

plt.gcf().canvas.manager.set_window_title("true activity, rate, mixture-weight images")
plt.savefig(figures_outpath + "activity_OPrate_OPweight_true.png")

# show saved reconstruction results
for (n,rate,wt) in zip(saved, rate_map_saved, wt_map_saved):
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6,3))
    t = ax1.imshow(rate)
    ax1.set_title('rate')
    fig.colorbar(t, ax=ax1, shrink=0.8, pad=0.002)
    t = ax2.imshow(wt)
    ax2.set_title('weight')
    fig.colorbar(t, ax=ax2, shrink=0.8, pad=0.002)
    plt.suptitle(f"reconstructed maps of OP rate and weight, iter {n}")
    plt.gcf().canvas.manager.set_window_title(f"iter {n}")
    plt.savefig(figures_outpath + f"OPrate_OPweight_recon_iter{n}.png")
