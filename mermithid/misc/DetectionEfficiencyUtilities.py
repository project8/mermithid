

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.stats import binom
from scipy import constants
from statsmodels.stats.proportion import proportion_confint
import helper_functions.alis_power as ap
import helper_functions.plot_methods as pm


def binomial_interval(n, k, alpha=1 / 3.1514872):

    if isinstance(n, list) or type(n) is np.ndarray:

        mean = np.zeros(len(n))
        interval = np.zeros((2, len(n)))
        interval_mean = np.zeros(len(n))
        for i in range(len(n)):
            interval[0][i], interval[1][i] = proportion_confint(k[i], n[i], method='wilson', alpha=alpha)
            rv = binom(n[i], k[i]*1./n[i])
            mean[i] = rv.mean()*1./n[i]
            interval_mean[i] = np.mean(interval.T[i])
        interval_error = [interval_mean - interval[0], interval[1]-interval_mean]

    else:
        l_int, u_int = proportion_confint(k, n, method='wilson', alpha=alpha)
        rv = binom(n, k*1./n)
        mean = rv.mean()*1./n
        interval_mean = np.mean([l_int, u_int])
        interval_error = [interval_mean - l_int, u_int-interval_mean]
    return mean, interval_mean, interval_error


def freq2en(f, mixfreq=24.5e9):
    """
    convert IF frequency in MHz to energy in keV
    """
    B=0.9578170819250281
    f = f*1e6
    emass = constants.electron_mass/constants.e*constants.c**2
    gamma = (constants.e*B)/(2.0*np.pi*constants.electron_mass) * 1/(f+mixfreq)

    return (gamma -1)*emass*1e-3


def en2freq(E, Theta=None, mixfreq=24.5e9):
    """
    convert energy in keV to IF frequency in MHz
    """
    B=0.9578170819250281
    E = E*1e3
    if Theta==None:
        Theta=np.pi/2
    e = constants.e
    c = constants.c

    emass = constants.electron_mass/constants.e*constants.c**2
    gamma = E/(emass)+1

    return ((constants.e*B)/(2.0*np.pi*constants.electron_mass) * 1/gamma)*1e-6-mixfreq*1e-6


def power_efficiency(f, plot=False, savepath='.'):
    """
    Toy model efficiency using pheno paper power vs. energy and arbitrary threshold
    """
    #print('Toy model efficiency')
    threshold = 8
    N = int(1e6)
    efficiency = np.zeros(len(f))
    efficiency_error = np.zeros((len(f),2))
    power_factor = ap.Power(f)/ap.Power([1407e6+24.5e9])

    #if plot:
    #    N = int(1e6)

    data0 = np.random.exponential(2, N)

    if plot:
        plt.figure(figsize=(7,5))
        plt.hist(data0, histtype='step', bins=100, color=pm.sns_color[0])
        plt.axvline(threshold, label='Detection threshold', color=pm.sns_color[4])
        plt.ylabel('N')
        plt.legend()
        plt.xlabel('SNR')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'hypothetical_detection.png'), dpi=200, transparent=True)
        plt.savefig(os.path.join(savepath, 'hypothetical_detection.pdf'), dpi=200, transparent=True)

    for i in range(len(f)):

        data = data0*power_factor[i]
        efficiency[i], _, efficiency_error[i] = binomial_interval(len(data), len(data[data>threshold]))


    if plot:
        plt.figure(figsize=(7,5))
        plt.plot(freq2en(f*1e-6-24.5e3), efficiency, color=pm.sns_color[0])
        #plt.fill_between(freq2en(f*1e-6-24.5e3), efficiency-efficiency_error.T[0], efficiency+efficiency_error.T[1], color=pm.sns_color[0], alpha=0.5)
        plt.ylabel('Efficiency')
        plt.xlabel('Energy [keV]')
        plt.tight_layout()

        #print((efficiency[-1]-efficiency[0])/(freq2en(f*1e-6-24.5e3)[-1]-freq2en(f*1e-6-24.5e3)[0])/efficiency[0])
        plt.savefig(os.path.join(savepath, 'hypothetical_efficiency.png'), dpi=200, transparent=True)
        plt.savefig(os.path.join(savepath, 'hypothetical_efficiency.pdf'), dpi=200, transparent=True)

    return efficiency/np.mean(efficiency), efficiency_error.T/np.mean(efficiency)

def interpolated_efficiency(f, snr_efficiency_dict, alpha=1):
    """
    turn efficiencies at fixed frequencies into callable efficiency for arbitrary frequency
    alpha scales uncertainty
    """

    # only use good index (where fits in efficiency analysis didn't fail)
    index_0 = snr_efficiency_dict['good_fit_index']

    x =  np.array(snr_efficiency_dict['frequency'])[index_0]
    y =  np.array(snr_efficiency_dict['tritium_rates'])[index_0]
    y = [y for _,y in sorted(zip(x,y))]
    y_err_0 = np.array(snr_efficiency_dict['tritium_rates_error'][0])[index_0]*alpha
    y_err_1 = np.array(snr_efficiency_dict['tritium_rates_error'][1])[index_0]*alpha
    y_err_0 = [y_err for _,y_err in sorted(zip(x,y_err_0))]
    y_err_1 = [y_err for _,y_err in sorted(zip(x,y_err_1))]
    x = sorted(x)

    yp = np.interp(f, x, y, left=0, right=0)
    yp_error_0 = np.interp(f, x, y_err_0, left=1, right=1)
    yp_error_1 = np.interp(f, x, y_err_1, left=1, right=1)


    return np.array(yp), np.array([yp_error_0, yp_error_1])

def pseudo_interpolated_efficiency(f, snr_efficiency_dict, alpha=1):
    """
    gaussian random efficiency in uncertainty intervall
    """
    if isinstance(f, float):
        y, y_error = interpolated_efficiency(f, snr_efficiency_dict,alpha)
        pseudo_y = np.random.randn()*y_error + y
    else:
        y, y_error = interpolated_efficiency(f, snr_efficiency_dict, alpha)

        pseudo_y = np.random.randn(len(f))
        pseudo_y[pseudo_y<0]*=np.array(y_error)[0][pseudo_y<0]
        pseudo_y[pseudo_y>0]*=np.array(y_error)[1][pseudo_y>0]


    return pseudo_y+y, y_error


def integrated_efficiency(f, snr_efficiency_dict, df=None , mix_freq=24.5e9, centers = True):
    """
    pseudo integrates interpolated efficiency over bin width.
    integral is approximated by just averaging a few points over the bin, because its much faster

    f: frequency bins in absolute frequencies (>25GHz), units is Hz
    df: width around bin centers that should be integrated over. Only used if f is float
    mix_freq: required because efficiency analysis was done in IF frequencies
    center: if true assume f is bin centers, if False assume f is bin edges
    """


    # number of points summed in "integration"
    N = 10

    # inteprolate efficiency and efficiency uncertainty

    frequency = np.array(snr_efficiency_dict['frequency'])[snr_efficiency_dict['good_fit_index']]
    efficiency, efficiency_error = interpolated_efficiency(frequency, snr_efficiency_dict)

    interp_eff = interp1d(frequency-24.5e9, efficiency, fill_value=0, bounds_error=False)
    interp_eff_error_up = interp1d(frequency-24.5e9, efficiency_error[1], fill_value=1, bounds_error=False)
    interp_eff_error_down = interp1d(frequency-24.5e9, efficiency_error[0], fill_value=1, bounds_error=False)


    # bin width
    if isinstance(f, float):
        if df==None:
            raise Exception('No integration width given')


        # if df is smaller than FSS bin width, do not integrate but just return interpolation

        if df > 2e6:
            x = np.linspace(f-mix_freq-df/2, f-mix_freq+df/2, N)
            eff = np.sum(interp_eff(x))/N
            eff_err_down = np.sum(interp_eff_error_down(x))/N
            eff_err_up = np.sum(interp_eff_error_up(x))/N
            return eff, [eff_err_down, eff_err_down]

        else:
            eff = interp_eff(f-mix_freq)
            eff_error_down = interp_eff_error_down(f-mix_freq)
            eff_error_up = interp_eff_error_up(f-mix_freq)

    else: # f ist list of array
        df = f[1]-f[0]

        # if df is smaller than FSS bin width, do not integrate but just return interpolation
        if df < 2e6:
            if centers:
                x = f - mix_freq
            else:
                x = f[0:-1]+0.5*df - mix_freq
            eff = interp_eff(x)
            eff_error_down = interp_eff_error_down(x)
            eff_error_up = interp_eff_error_up(x)
            eff_err = [eff_error_down, eff_error_up]

        # else pseudo integrate
        else:
            if centers:
                M = len(f)
                x = np.zeros((M, N))
                bin_centers = f-mix_freq
            else:
                M = len(f)-1
                x = x = np.zeros((M, N))
                bin_centers = f[0:-1]+0.5*df-mix_freq

            for i in range(M):
                x[i] = np.linspace(bin_centers[i]-df/2, bin_centers[i]+df/2, N)


            eff = interp_eff(x)
            eff_err = [interp_eff_error_down(x), interp_eff_error_up(x)]


            eff =  np.sum(eff, axis=1)/N
            #eff_err = [np.sqrt(np.sum(eff_err[0]**2, axis=1))/np.sqrt(N),
            #           np.sqrt(np.sum(eff_err[1]**2, axis=1))/np.sqrt(N)]
            eff_err = [np.mean(eff_err[0], axis=1), np.mean(eff_err[1], axis=1)]


        return eff, eff_err



def pseudo_integrated_efficiency(f, df, snr_efficiency_dict, alpha=1):
    if isinstance(f, float):
        y, y_error = integrated_efficiency(f, snr_efficiency_dict, df)
        pseudo_y = np.random.randn()*y_error# + y
    else:
        y, y_error = integrated_efficiency(f, snr_efficiency_dict, df)

        pseudo_y = np.random.randn(len(f))
        #print(pseudo_y)
        pseudo_y[pseudo_y<0]*=y_error[0][pseudo_y<0]
        pseudo_y[pseudo_y>0]*=y_error[1][pseudo_y>0]
        #print(pseudo_y)
    return pseudo_y+y, y_error


def binomial_interval(n, k, alpha=1 / 3.1514872):

    if isinstance(n, list) or type(n) is np.ndarray:

        mean = np.zeros(len(n))
        interval = np.zeros((2, len(n)))
        interval_mean = np.zeros(len(n))
        for i in range(len(n)):
            interval[0][i], interval[1][i] = proportion_confint(k[i], n[i], method='wilson', alpha=alpha)
            rv = binom(n[i], k[i]*1./n[i])
            mean[i] = rv.mean()*1./n[i]
            interval_mean[i] = np.mean(interval.T[i])
        interval_error = [interval_mean - interval[0], interval[1]-interval_mean]

    else:
        l_int, u_int = proportion_confint(k, n, method='wilson', alpha=alpha)
        rv = binom(n, k*1./n)
        mean = rv.mean()*1./n
        interval_mean = np.mean([l_int, u_int])
        interval_error = [interval_mean - l_int, u_int-interval_mean]
    return mean, interval_mean, interval_error