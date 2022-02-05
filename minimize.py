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
This script contains class `TSLFit` for deconvolution and fitting TSL curves using first-order equation.
The initial peak positions are required, they can be found automatically or set manually. Primitive automatic
peak identification does not work properly, better to use the manual function. The initial curve also may be
smoothed, if required. The function for the subtraction of baseline is also available. Corresponding 
functionality provided by corresponding methods and used approaches can be found in their descriptions.
'''

import numpy as np
import scipy.special as sp
import TSLDataReader as tdr
import matplotlib.pyplot as plt
import methods
from scipy.optimize import least_squares

    

class TSLFit():
    def __init__(self, temp, curve, start_t = 20, stop_t = 250, peaks_data = [], peaks = 'auto', smooth = False, baseline = True, bl_params = []):
        '''
        `temp` - array of temperature values, in K. Should be the same dimension as `curve`
        `curve` - array of intensity values. Should be the same dimension as `temp`
        `start_t` - start temperature point in the `curve` that we are interesting in
        `stop_t`- last temperature point in the `curve` that we are interested in
        `peaks_data` - possible values of the peaks positions in the `curve`, in K
        `peaks` - two values available, 'auto' or 'manual'. The automatic peaks looking for algorithm will be 
        used in the case 'auto', while the data from 'peaks_data' is used for 'manual'
        `smooth` - boolean value, True if the smooth of `curve` is required
        `baseline` - boolean value, True if the baseline substraction is required
        `bl_params` - specific parameters for the baseline algorithm may be set here

        '''
        self.t ,self.curve = methods.upd_data(temp, curve, start_t, stop_t)
        if smooth:
            self.curve = methods.smooth(self.curve)
        if baseline:
            if len(bl_params) == 2:
                self.curve = self.curve - methods.baseline_als(self.curve, bl_params[0], bl_params[1])
            else:
                self.curve = self.curve - methods.baseline_als(self.curve)
        if peaks == 'auto':
            indexes = methods.peaks_finder(self.curve)
            self.peaks = self.t[indexes]
        elif peaks == 'manual' or len(peaks_data) > 0:
            self.peaks = np.abs(peaks_data)
        else:
            print('Incorrect approach for peaks identification has been used. The default value "auto" will be used.')
            indexes = methods.peaks_finder(self.curve)
            self.peaks = self.t[indexes]
        self.init_params = self.generate_initial_func_parameters()

    def target_func(self, params):
        '''
        Setting of target function for minimisation task. In our case the function y'-y is used and works well
        '''
        minim = self.curve.copy()
        bkg = self.signal_builder(params)
        sign = np.sum(bkg, axis=0)
        return minim - sign
    
    def fit(self):
        '''
        Main method that starts the minimisation algorithm. As a main algorithm the `least_squares` from `scipy`
        with bounds is used.
        '''
        bnds = self.generate_bounds()
        res = least_squares(self.target_func, self.init_params[0], bounds = bnds, gtol=1e-8, max_nfev = 10000, ftol=1e-8, xtol=1e-8)
        if res.success:
            self.calcr2(res.x)
        else:
            print('Fitting was unsuccessful, try again wint new parameters')
        return res

    def peak_function(self, prm): #T, E, I
        '''
        Function for description the first order equation. The primary article with this equation is here
        https://iopscience.iop.org/article/10.1088/0022-3727/31/19/037
        '''
        Tm, E, Im = prm
        k = .000086173303
        dm = (2.0*k*(Tm))/E
        T=self.t
        I_t = Im*np.exp(1.0 +(E/(k*T))*((T-Tm)/Tm)-((T*T)/(Tm*Tm))*np.exp((E/(k*T))*((T-Tm)/Tm))*(1.0-((2.0*k*T)/E))-dm)
        return I_t
    
    def signal_builder(self, params):
        '''
        This method is used for building the glow curves for all peaks in accordance with their parameters. The 
        `peak_function` is used for building single peak.
        '''
        signal = []
        params = params.reshape(-1, 3)
        for param in params:
            signal.append(self.peak_function(param))
        return np.array(signal)

    def generate_initial_func_parameters(self):
        '''
        Generation the initial parameters for all peaks for fitting is set here. These values will have been 
        adjusted during fit.
        '''
        initial_param = np.zeros((len(self.peaks), 3))
        initial_param[:,0] = self.peaks
        initial_param[:,1] = 0.1
        initial_param[:,2] = 1
        return initial_param.reshape(1,-1)
    
    def generate_bounds(self):
        '''
        Generation boundings for all initial parameters. They are limiting possible values of initial parameters
        during fit procedure.
        '''
        bnds = [[min(self.t)-1, max(self.t)+1],[1e-5, 4],[0, np.inf]]*len(self.peaks)
        bnds= np.array(bnds)
        return (bnds[:,0], bnds[:,1])  

    def calcr2(self, params):
        '''
        Calculation of R^2 coefficient for evaluation of fitting quality
        '''
        deconv = self.signal_builder(params)
        fitted_curve = np.sum(deconv, axis = 0)
        adj_matrix = np.corrcoef(self.curve, fitted_curve)
        r2 = adj_matrix[0, 1]
        print('R2 correlation coefficient is ', r2)

    



if __name__ == "__main__":
    fname = 'Und/Cs3Cu2I5 medGain 1_55kV (2atempt2)'
    data = tdr.TSLDataReader(fname).read_data() #read TSL data file and data extraction
    temp = data[:,0]
    curve = data[:,3]
    plt.plot(temp, curve)
    plt.show()
    #lanch the TSLFit for curve doconvolution
    tst = TSLFit(temp, curve, peaks = 'manual', peaks_data= [28,33,38,43, 50,72,88,125], start_t = 20, stop_t=140, smooth = True, baseline = True)
    #[31.4, 35.1, 38, 39.5, 41, 43, 44, 50, 52.5, 55.6, 82.7, 109.7, 124.7, 136.8, 143.6, 147, 170.9, 188.1, 190, 221.2]
    #result of deconvolution
    res = tst.fit()
    sign = tst.signal_builder(res.x)
    #plot initial curve and set of peaks those are result of deconvolution
    plt.plot(tst.t, tst.curve)
    for s in sign:
        plt.plot(tst.t, s, c='r')
    plt.plot(tst.t, np.sum(sign, axis = 0))
    plt.show()

   


