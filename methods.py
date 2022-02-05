#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2022 Yauhen Tratsiak. All rights reserved.
# Authors: Yauhen Tratsiak <ytratsia@utk.edu>
# License: GPLv3 (GNU General Public License Version 3)
#          https://www.gnu.org/licenses/quick-guide-gplv3.html
#
# This file is part of TSL deconvolute software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

'''
This file contains different methods that are required for `TSLFit` class. These methods include but not 
limeted with peaks finder algorithm, baseline algorithm, simple algorithm for curve smoothing.
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve

def peaks_finder (signal, window=13, koef = 3):
    peaks = []
    half_w = int(window/2)
    for i in range(window,len(signal)-window,1):
        mean_l = np.mean(signal[i-window-1:i])
        mean_r = np.mean(signal[i+1:i+window+1])
        mean_c = np.mean(signal[i-half_w:i+half_w+1])
        if (mean_l < mean_c and mean_c > mean_r):
            peaks.append(i)
    upd_peaks = []
    for i in range(len(peaks)-1):
        if peaks[i+1]-peaks[i] > window*koef:
            upd_peaks.append(peaks[i])
    return np.array(upd_peaks)

def find_i(x, y, init):
    i = 0
    res = False
    while i<len(x) or not res:
        if int(x[i]) >= init:
            res = True
            return i
        else:
            i += 1
    
    return None

def upd_data(x, y, val, stop = 0):
    i = find_i(x, y, val)
    if stop:
        yi = find_i(x, y, stop)
    else:
        yi = None
    if i is not None:
        if i < len(x):
            start = i
        else:
            start = 0
    else:
        start = 0
    if yi is not None:
        if yi < len(x):
            stp = yi
        else:
            stp = len(x)
    else:
        stp = len(x)
    return x[start:stp], y[start:stp]

def smooth(y, window = 13):
    upd_y = y.copy()
    half_w = int(window/2)
    for i in range(half_w, len(y)-half_w, 1):
        m_val = np.mean(y[i-half_w:i+half_w])
        upd_y[i] = m_val
    return upd_y

def baseline_als(y, lam=10e+5, p=1e-3, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

if "__name__" == "__main__":
    pass