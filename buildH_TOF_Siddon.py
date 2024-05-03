# -*- coding: utf-8 -*-
"""
Produce nonTOF projection matrix by Siddon's raytracing
All units are listed below unless otherwise noted:
    spatial: cm
    temporal: ns
@author: Zheyuan Zhu
Edited: Chien-Min Kao, 10/2023
"""

import numpy as np

#%% get TOF FWHM
import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t',
                    '--TOF',
                    help='TOF FWHM (ps)',
                    type=int,
                    default=0)
parser.add_argument('--hdir',
                    help='system matrix directory',
                    type=str,
                    default=".")
args = parser.parse_args()
hdir = "." if args.hdir=="None" else args.hdir

outpath = f"{hdir}/h_matrix/"
if not os.path.exists(outpath): os.makedirs(outpath)

from config import TOF_FWHM as default_TOF_FWHM, getTOFSettings
TOF_FWHM = int(default_TOF_FWHM if (args.TOF<=0) else args.TOF)
TOF_std, N_TOF, TOF_edges, TOF_bins = getTOFSettings(TOF_FWHM)

print (f"{TOF_FWHM = }ps")

#%% calculat H matrix

from config import N_obj, N_det, det_xs, det_ys
from config import image_xedges as xedges, image_yedges as yedges
from utils import SiddonTOF_pp

N_LOR = N_det*N_det
N_LOS = N_LOR*N_TOF

# define object to produce TOF weights

N_intersect = int(N_obj)
iis = np.zeros(N_LOS*N_intersect).astype(np.int32)
jjs = np.zeros(N_LOS*N_intersect).astype(np.int32)
weights = np.zeros(N_LOS*N_intersect).astype(np.float32)
start_ind = 0
for ii in range(N_LOR):
    print (f"{ii}/{N_LOR}",end=', ')
    i1, i2 = np.unravel_index(ii, (N_det,N_det), order='C')
    i1s = np.full((N_TOF,), i1, dtype=np.int32)
    i2s = np.full((N_TOF,), i2, dtype=np.int32)
    ks = np.arange(N_TOF)
    _iis = np.ravel_multi_index((i1s,i2s,ks), (N_det,N_det,N_TOF), order='C')
    p1, p2 = (det_xs[i1],det_ys[i1]), (det_xs[i2],det_ys[i2])
    for (ix, iy, wts, _) in zip(*SiddonTOF_pp(p1, p2, xedges, yedges, \
                                TOF_std, TOF_edges, nsubsamples=10)):
        jjs[start_ind:start_ind+N_TOF] = \
            np.ravel_multi_index((ix,iy), (N_obj,N_obj), order='F')
        iis[start_ind:start_ind+N_TOF] = _iis
        weights[start_ind:start_ind+N_TOF] = wts
        start_ind += N_TOF
# complete all LORs, delete unused elements in iis, jjs, weights
iis = iis[:start_ind]
jjs = jjs[:start_ind]
weights = weights[:start_ind]
# create H matrix
from scipy.sparse import coo_matrix, diags
H = coo_matrix((weights,(iis,jjs)), shape=(N_LOS, N_obj*N_obj))

# wheter to normalize all column sums (sensitivity to pixels) to 1
normalize_H = True
if normalize_H:
    H_sum = np.asarray(H.sum(axis=0)).ravel()
    inv_H_norm = 1/H_sum
    H = H.dot(diags(inv_H_norm))

# save H matrix
from scipy.io import savemat
Hfilename = f"H_TOF_Siddon_{TOF_FWHM}ps.mat"
savemat(outpath + Hfilename, {'H':H})

