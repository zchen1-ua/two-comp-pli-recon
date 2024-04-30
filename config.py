"""
Configuration file for positron lifetime imaging simulation

Make a copy of this config_sample.py file as config.py, and modify config.py
for experimentation as config.py is not tracked by git.

All units are listed below unless otherwise noted:
    spatial: cm
    temporal: ns

@author: Zheyuan Zhu
"""

import numpy as np
from utils import get_grid
from constants import c

#%% Observation properties
T         = 1    # duration of observation in seconds
eta_decay = 0.01 # 10% of the decays give rise to detectable signals

#%% Scanner configuration
N_det       = 364                      # number of detectors on detector ring
                                       # (default: 560, GE scanner)
D           = 57.2                     # diameter of the detector ring, in cm 
                                       # (default: 88cm, GE scanner)
# define detector angular edges
det_theta_edges, det_theta_bins = get_grid(N_det, 2*np.pi/N_det, center=False)
det_xs, det_ys = (D/2)*np.cos(det_theta_bins), (D/2)*np.sin(det_theta_bins)

#%% Scanner configuration
TOF_FWHM    = 200                      # TOF res in FWHM, in picoseconds
                                       # Qi's paper: 570 ps
                                       
## compute TOFstd in ns, N_TOF, TOF_edges, and TOF_bins accordingly
gettimebinsize = lambda FWHM: FWHM/2/1e3
def getTOFSettings(FWHM):
    TOF_std = FWHM/2.3548/1e3
    timebinsize = gettimebinsize(FWHM)
    N_TOF = 2*(int(np.floor(D/(c*timebinsize/2)))//2)+1
    TOF_edges, TOF_bins = get_grid(N_TOF, timebinsize)
    return TOF_std, N_TOF, TOF_edges, TOF_bins

# get settings for current the TOF FWHM
TOF_std, N_TOF, TOF_edges, TOF_bins = getTOFSettings(TOF_FWHM)

# define time bins for tau measurements
min_tau, max_tau = -2, 20
def getTauSettings(FWHM):
    timebinsize = gettimebinsize(FWHM)
    N_tau_bins = int((max_tau-min_tau)/timebinsize)
    tau_edges, tau_bins = get_grid(N_tau_bins, timebinsize, center=False)
    tau_bins += min_tau
    tau_edges += min_tau
    return N_tau_bins, tau_edges, tau_bins
N_tau_bins, tau_edges, tau_bins = getTauSettings(TOF_FWHM)

#%% image configuration
dx    = 0.327  # pixel size in cm 
               # (default: 0.327, from Qi's paper)
N_obj = 41     # number of pixels along x and y

image_xedges, image_xbins = get_grid(N_obj, dx)
image_yedges, image_ybins = get_grid(N_obj, dx)

#%% parameters for computing system matrix H
N_em_angle_samples  = int(5*N_det)  # Number of directions for emissions
N_pixel_subsample   = 100           # Number of samples for a pixel