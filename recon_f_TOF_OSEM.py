# -*- coding: utf-8 -*-
"""
TOF reconstruction of activity map using OSEM
Created on Sun Feb  6 11:18:11 2022
@author: Zheyuan Zhu
Edited: Chien-Min Kao, 10/2023, Zhuo Chen 1/2024
"""
import numpy as np

import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('-r',
                    '--run',
                    help='run id',
                    type=int,
                    default=1)
parser.add_argument('-S',
                    '--subsets',
                    help='number of subsets',
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
parser.add_argument('--wdir',
                    help='working directory',
                    type=str,
                    default=".")
parser.add_argument('--hdir',
                    help='system matrix directory',
                    type=str,
                    default=".")
# args = parser.parse_args(["--wdir", "./full", "-r", "0", "-o", "all-log,10"])
args = parser.parse_args()
run = args.run
nsubsets = args.subsets
max_iters = args.niters//nsubsets + (1 if args.niters%nsubsets else 0)
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

print (f"f, TOF, OSEM: {run = }, {nsubsets = }, {max_iters = }, {TOF_FWHM = }ps, {to_save = }")

# define setsets
from config import N_det
from numpy.random import permutation
N_LOS = N_det*N_det*N_TOF
all_LOSs = permutation(np.arange(0, N_LOS))
if (nsubsets<=1):
    subsets = [all_LOSs]
else:
    m = np.size(all_LOSs)//nsubsets
    subsets = []
    for s in range(0,nsubsets-1):
        subsets.append(all_LOSs[s*m:(s+1)*m])
    subsets.append(all_LOSs[(nsubsets-1)*m:])

# load true, H matrix and data
from scipy.io import loadmat
datapath = f"{wdir}/{TOF_FWHM}ps/" + str(run) + '/data/'

## load true image
f_map_true = loadmat(datapath + 'events_raw.mat')['f_map']
f_vec_true = np.reshape(f_map_true, (-1), order='C')

## load TOF matrix
# Hfilename = f"{hdir}/h_matrix/H_TOF_PB_{TOF_FWHM}ps.mat"
# Hfilename = f"{hdir}/h_matrix/H_TOF_{TOF_FWHM}ps.mat"
Hfilename = f"{hdir}/h_matrix/H_TOF_Siddon_{TOF_FWHM}ps.mat"
H = loadmat(Hfilename)['H'].tocsr()

## load data
events_raw_filename = datapath + 'events_raw.mat'
events_measured_filename = datapath + f"events_measured_{TOF_FWHM}ps.mat"
W = loadmat(events_measured_filename)['W']
i1_511 = np.int32(W[:,0])
i2_511 = np.int32(W[:,1])
k_511 = np.int32(W[:,2])

# convert to histogram data
iis = np.ravel_multi_index((i1_511,i2_511,k_511), (N_det,N_det,N_TOF), order='C')
g = np.zeros((N_LOS,))
for ii in iis: g[ii] += 1
N_events = np.nansum(g)
print (f"{N_events = }")

#-----------------------------------------
# solve

# stopping conditions and step size
rel_diff_tol = 1e-8

# initial solution
f_vec_recon = np.ones_like(f_vec_true)*(float(N_events)/np.size(f_vec_true))
f_vec_recon_old = np.copy(f_vec_recon)
N_obj = np.int32(np.sqrt(np.size(f_vec_recon)))

from utils import NMSE
from constants import nan_killer

# prepare quantities for subsets
Nsub = [np.size(subsets[s]) for s in range(nsubsets)]
Hsub = [H[subsets[s],:] for s in range(nsubsets)]
gsub =[g[subsets[s]] for s in range(nsubsets)]
Ssub = [A.transpose().dot(np.ones((n,)))+nan_killer \
        for (A,n) in zip(Hsub, Nsub)]

    
rel_diff, niter = 1, 1
images_saved, likelihood_logged, NMSE_logged = [], [], []
crc_logged = []
bv_logged = []
rel_diff_logged = []
alr_logged = []

circle, background, zeros = np.sort(np.unique(f_vec_true))[::-1]
print("Event count in circles:", circle, "Event count in background:", background)
def calculate_crc(rate_vec_true, rates):
    ind_circle = (rate_vec_true == circle)
    ind_background = (rate_vec_true == background)
    crc = (np.mean(rates[ind_circle]) / np.mean(rates[ind_background]) - 1) / (circle / background - 1)
    return crc

def calculate_bv(rate_vec_true, rates):
    ind_background = (rate_vec_true == background)
    bv = np.std(rates[ind_background]) / np.mean(rates[ind_background])
    return bv
    
OPwt_map_true, OPrate_map_true = loadmat(events_raw_filename)['OP_map']
OPrate_vec_true = np.reshape(OPrate_map_true, (-1), order='C')
ind_circle1 = (OPrate_vec_true == 0.6)
ind_circle2 = (OPrate_vec_true == 0.4)
ind_background = (OPrate_vec_true == 0.5)

def calculate_abs_log_ratio(rate_vec_true, rates, ind_circle1, ind_circle2, ind_background):
    alr1 = np.absolute(np.log(np.mean(rates[ind_circle1]) / np.mean(rates[ind_background])))
    alr2 = np.absolute(np.log(np.mean(rates[ind_circle2]) / np.mean(rates[ind_background])))
    alr = np.mean([alr1, alr2])
    return alr
    
while (rel_diff>=rel_diff_tol) and (niter<=max_iters):
    f_vec_recon_old = f_vec_recon.copy()
    for (A,y,s) in zip(Hsub,gsub,Ssub):
        Af = A.dot(f_vec_recon)
        yy = np.copy(y)
        yy[Af!=0] /= Af[Af!=0]
        yy[Af==0] = (yy[Af==0]==0)
        f_vec_recon *= A.transpose().dot(yy)/s
    Hf = H.dot(f_vec_recon)
    logy = np.nansum(g*np.log(Hf+nan_killer)-Hf)
    likelihood_logged.append(logy)
    NMSE_logged.append(NMSE(f_vec_recon, f_vec_true))
    rel_diff = NMSE(f_vec_recon, f_vec_recon_old)
    rel_diff_logged.append(rel_diff)
    crc_logged.append(calculate_crc(f_vec_true, f_vec_recon))
    bv_logged.append(calculate_bv(f_vec_true, f_vec_recon))
    alr_logged.append(calculate_abs_log_ratio(f_vec_true, f_vec_recon, ind_circle1, ind_circle2, ind_background))
    print("step=", niter, "log_likelihood=", logy, "rel_diff=", rel_diff, "NMSE=", NMSE_logged[-1], "crc:", crc_logged[-1], "bv:", bv_logged[-1], "alr:", alr_logged[-1])
    if (niter in to_save):
        im = np.reshape(np.copy(f_vec_recon),(N_obj,N_obj),order='C')
        images_saved.append(im)
    niter += 1
print("step=", niter, "log_likelihood=", logy, "rel_diff=", rel_diff, "NMSE=", NMSE_logged[-1], "crc:", crc_logged[-1], "bv:", bv_logged[-1], "alr:", alr_logged[-1])
niter -= 1
if (niter not in to_save):
    to_save = np.append(to_save, niter)
    images_saved.append(np.reshape(f_vec_recon,(N_obj,N_obj),order='C'))
saved = np.unique(to_save[to_save<=niter])

import os
outpath = f"{wdir}/{TOF_FWHM}ps/" + str(run) + "/recon_f_TOF_OSEM/"
if not os.path.exists(outpath): os.makedirs(outpath)
figures_outpath = outpath + "figures/"
if not os.path.exists(figures_outpath): os.makedirs(figures_outpath)
    
# save results
from scipy.io import savemat
savemat(outpath + "recon.mat",
        {"iters": saved,
         "f_map_recons": images_saved,
         "f_map_true": f_map_true,
         'NMSE': NMSE_logged,
         'likelihood': likelihood_logged,
         'rel_diff': rel_diff_logged,
         'CRC': crc_logged,
         'BV': bv_logged,
         'ALR': alr_logged})

from numba import jit
from config import N_det, N_TOF
@jit(nopython=True, parallel=True)
def data2sino(data):
    N_RHO, N_PHI = N_det+1, N_det//2
    re = np.zeros((N_TOF, N_PHI, N_RHO))
    for (ii,g) in enumerate(data):
        if (g>0):
            i1, t = ii//(N_TOF*N_det), ii%(N_TOF*N_det)
            k, i2 = t%N_TOF, t//N_TOF
            ia = i1+i2+1
            ir = N_det//2-np.abs(i1-i2)
            if (ia>=N_det): ir = -ir
            ia = (ia%N_det)//2
            ir += N_det//2-1
            re[k,ia,ir] += g
    return re
    
# plots
import matplotlib.pyplot as plt

# plot log likelihood vs iteration number
likelihood_logged = np.array(likelihood_logged)
NMSE_logged = np.array(NMSE_logged)

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
ax1.plot(np.arange(niter)+1, likelihood_logged-np.min(likelihood_logged)+0.1)
ax1.set_yscale("log", base=10)
ax1.set_title("log likelihood")

ax2.plot(np.arange(niter)+1, NMSE_logged)
ax2.set_yscale("log", base=10)
ax2.set_title("NMSE")

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

plt.gcf().canvas.manager.set_window_title("likelihood, NMSE")
plt.savefig(figures_outpath + "f_likelihood_grad.png")

# plot relative change in likelihood vs iteration number
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
ax1.set_title("in log10")
ax1.plot(np.arange(niter)+1, rel_diff_logged)
ax1.set_yscale("log", base=10)

ax2.set_title("no log trans")
ax2.plot(np.arange(niter)+1, rel_diff_logged)

plt.savefig(figures_outpath+"rel_diff.png")

# plot CRC and BV
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
ax1.set_title("CRC")
ax1.plot(np.arange(niter)+1, crc_logged)

ax2.plot(np.arange(niter)+1, bv_logged)
ax2.set_title("BV")

ax3.plot(np.arange(niter)+1, alr_logged)
ax3.set_title("ALR")

# Label maximum point in CRC plot
max_crc_index = np.argmax(crc_logged)+1
max_crc_value = np.max(crc_logged)
ax1.scatter(max_crc_index, max_crc_value, color='red')
ax1.annotate(f'Max: ({max_crc_index}, {max_crc_value:.2f})', xy=(max_crc_index, max_crc_value),
             xytext=(max_crc_index + 2, max_crc_value),
             color='red')
             
# Find the index where CRC curve starts to decrease
#decrease_start_index = np.where(np.diff(crc_logged) < 0)[0][0] + 1

# Label the point on the CRC plot
#decrease_start_value = crc_logged[decrease_start_index-1]
#ax1.scatter(decrease_start_index, decrease_start_value, color='blue')
#ax1.annotate(f'Decrease: ({decrease_start_index}, {decrease_start_value:.2f})',
#             xy=(decrease_start_index, decrease_start_value),
#             xytext=(decrease_start_index + 2, decrease_start_value*0.9),
#             color='blue')

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

plt.savefig(figures_outpath+"CRC_BV.png")

# show selected TOF data as images
fig, axes = plt.subplots(3,5,figsize=(10,6))
fig.tight_layout()
gsino = data2sino(g)
f_vec_recon = np.reshape(images_saved[-1],(-1,))
HF_recon_sino = data2sino(H.dot(f_vec_recon))
HF_true_sino = data2sino(H.dot(f_vec_true))
dphi = 180./np.shape(gsino)[1]
axes[0][0].set_ylabel('g')
axes[1][0].set_ylabel('Hf_recon')
axes[2][0].set_ylabel('Hf_true')
for i in np.arange(5):
    ia = i*30
    angle = ia*dphi
    axes[0][i].imshow(gsino[:,ia,N_det//2-30:N_det//2+31], aspect='auto')
    axes[0][i].axes.xaxis.set_visible(False)
    axes[0][i].get_xaxis().set_ticks([])
    axes[0][i].get_yaxis().set_ticks([])
    axes[0][i].set_title(f"{angle = :3.1f}")
    axes[1][i].imshow(HF_recon_sino[:,ia,N_det//2-30:N_det//2+31], aspect='auto')
    axes[1][i].get_xaxis().set_ticks([])
    axes[1][i].get_yaxis().set_ticks([])
    axes[2][i].imshow(HF_true_sino[:,ia,N_det//2-30:N_det//2+31], aspect='auto')
    axes[2][i].get_xaxis().set_ticks([])
    axes[2][i].get_yaxis().set_ticks([])
plt.savefig(figures_outpath + "g_Hf.png")

# show true image
plt.figure()
plt.imshow(f_map_true)
plt.colorbar()
plt.clim([0,None])
plt.title("true f")
plt.gcf().canvas.manager.set_window_title("true activity")
plt.savefig(figures_outpath + "f_map_true.png")

# show svaed reconstruction images at saved iterations
for (n,im) in zip(saved, images_saved):
    plt.figure()
    plt.imshow(im)
    plt.colorbar()
    plt.clim([0,None])
    plt.title(f"recon f, iter={n}")
    plt.gcf().canvas.manager.set_window_title(f"iter {n}")
    plt.savefig(figures_outpath + f"f_map_recon_iter{n}.png")