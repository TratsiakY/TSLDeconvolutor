import numpy as np
import scipy.special as sp
import TSLDataReader as tdr
import matplotlib.pyplot as plt
import methods
from scipy.optimize import least_squares
import result_processing as rp
    

class TSLFit():
    def __init__(self, temp, curve,  Beta = 3, start_t = 20, stop_t = 250, peaks_data = [], peaks = 'auto', smooth = False, baseline = True, bl_params = []):
        self.t ,self.curve = methods.upd_data(temp, curve, start_t, stop_t)
        if smooth:
            self.curve = methods.smooth(self.curve)
        if baseline:
            if len(bl_params) == 2:
                self.curve = self.curve - methods.baseline_als(self.curve, bl_params[0], bl_params[1])
            else:
                self.curve = self.curve - methods.baseline_als(self.curve)
        self.beta = Beta
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
        minim = self.curve.copy()
        bkg = self.signal_builder(params)
        sign = np.sum(bkg, axis=0)
        return minim - sign
    
    def fit(self):
        bnds = self.generate_bounds()
        res = least_squares(self.target_func, self.init_params[0], bounds = bnds, gtol=1e-8, max_nfev = 10000, ftol=1e-8, xtol=1e-8)
        if res.success:
            self.calcr2(res.x)
        else:
            print('Fitting was unsuccessful, try again wint new parameters')
        return res

    def peak_function(self, prm): #T, E, I
        Tm, E, Im = prm
        k = .000086173303
        dm = (2.0*k*(Tm))/E
        T=self.t
        I_t = Im*np.exp(1.0 +(E/(k*T))*((T-Tm)/Tm)-((T*T)/(Tm*Tm))*np.exp((E/(k*T))*((T-Tm)/Tm))*(1.0-((2.0*k*T)/E))-dm)
        return I_t
    
    def signal_builder(self, params):
        signal = []
        params = params.reshape(-1, 3)
        for param in params:
            signal.append(self.peak_function(param))
        return np.array(signal)

    def generate_initial_func_parameters(self):
        initial_param = np.zeros((len(self.peaks), 3))
        initial_param[:,0] = self.peaks
        initial_param[:,1] = 0.1
        initial_param[:,2] = 1
        return initial_param.reshape(1,-1)
    
    def generate_bounds(self):
        bnds = [[min(self.t)-1, max(self.t)+1],[1e-5, 4],[0, np.inf]]*len(self.peaks)
        bnds= np.array(bnds)
        return (bnds[:,0], bnds[:,1])  

    def calcr2(self, params):
        deconv = self.signal_builder(params)
        fitted_curve = np.sum(deconv, axis = 0)
        adj_matrix = np.corrcoef(self.curve, fitted_curve)
        r2 = adj_matrix[0, 1]
        print('R2 correlation coefficient is ', r2)

    



if __name__ == "__main__":
    fname = 'CsCuI_In_att1_medgain'
    data = tdr.TSLDataReader(fname).read_data()
    temp = data[:,0]
    curve = data[:,3]
    plt.plot(temp, curve)
    plt.show()
    tst = TSLFit(temp, curve, peaks = 'manual', peaks_data= [82, 103, 109, 118, 140, 148, 167, 212, 257], start_t = 40, stop_t=290, smooth = True, baseline = True)
    #[31.4, 35.1, 38, 39.5, 41, 43, 44, 50, 52.5, 55.6, 82.7, 109.7, 124.7, 136.8, 143.6, 147, 170.9, 188.1, 190, 221.2]
    res = tst.fit()
    sign = tst.signal_builder(res.x)
    plt.plot(tst.t, tst.curve)
    for s in sign:
        plt.plot(tst.t, s, c='r')
    plt.plot(tst.t, np.sum(sign, axis = 0))
    plt.show()
    rdm = rp.DataProcessing(res.x, tst.t, tst.curve, sign, 3, fname)

    # print('Final results')
    # print(res.message)
    dat = rdm.update_data()
    # plt.scatter(dat[:,0], dat[:,1])
    # plt.show()
    # plt.scatter(dat[:,0], dat[:,2])
    # plt.show()
    # plt.scatter(dat[:,0], dat[:,3])
    # plt.show()
    rdm.unload_peaks()
    result = np.concatenate((np.expand_dims(temp, axis = 0).T, np.expand_dims(curve, axis = 0).T), axis = 1)
    np.savetxt(fname+'_initdata', result, fmt ='%.3f', delimiter=' ', newline='\n')



