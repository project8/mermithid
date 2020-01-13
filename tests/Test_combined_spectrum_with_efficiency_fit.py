'''
This scripts aims at testing Tritium specific processors.
Author: C. Claessens
Date: Jul 31 2019
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

from mermithid.processors.TritiumSpectrum import DistortedTritiumSpectrumLikelihoodSampler, KuriePlotFitter
from morpho.processors.plots import Histogram
from morpho.processors.IO import IOJSONProcessor
from mermithid.misc.Constants import seconds_per_year, tritium_endpoint
from mermithid.misc import Constants

from ROOT import TH1F
from ROOT import TMath

from morpho.processors.sampling.PyStanSamplingProcessor import PyStanSamplingProcessor
from morpho.processors.plots.APosterioriDistribution import APosterioriDistribution

import matplotlib.pyplot as plt
import numpy as np

import importlib.machinery
modulename = importlib.machinery.SourceFileLoader('modulename','/host-mermithid/mermithid/processors/TritiumSpectrum/DistortedTritiumSpectrumLikelihoodSampler.py').load_module()
from modulename import DistortedTritiumSpectrumLikelihoodSampler

def ProduceAndReadData(counts=10000):


    specGen_config = {
        "volume": 7e-6*1e-2, # [m3]
        "density": 3e17, # [1/m3]
        "duration": 1.*seconds_per_year()/12.*1, # [s]
        "neutrino_mass" :0, # [eV]
        "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
        "frequency_window": [-100e6, +100e6], #[Fmin, Fmax]
        "energy_or_frequency": "frequency",
        # "energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
        "background": 1e-5, # [counts/eV/s]
        "energy_resolution": 5,# [eV]
        "frequency_resolution": 1e6,# [Hz]
        "mode": "generate",
        "varName": "F",
        "iter": counts,
        "interestParams": ["F"],
        "fixedParams": {"m_nu": 0},
        "options": {"snr_efficiency": True, "channel_efficiency":False, "smearing": True},
        "snr_efficiency_coefficients": [-265.03357206889626, 6.693200670990694e-07, -5.795611253664308e-16, 1.5928835520798478e-25, 2.892234977030861e-35, -1.566210147698845e-44], #[-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
        "channel_central_frequency": 1400e6,
        "mixing_frequency": 24.5e9
    }
    json_read = {
        "filename" : "/host/efficiency_data.json",
        "variables" : ["f", "efficiency", "efficiency_error"],
        "action" : "read"
    }
    histo_plot = {
        "variables": ["F"],
        "n_bins_x": 100,
        "title": "spectrum",
        "range": [1320e6, 1480e6]
    }


    specGen = DistortedTritiumSpectrumLikelihoodSampler("specGen")
    histo = Histogram("histo")
    json_reader = IOJSONProcessor("jsonReader")

    specGen.Configure(specGen_config)
    histo.Configure(histo_plot)
    json_reader.Configure(json_read)

    specGen.Run()
    json_reader.Run()
    tritium_data = specGen.data
    efficiency_data = json_reader.data
    print(efficiency_data.keys())
    print('Number of events: {}'.format(len(tritium_data["F"])))
    print('Number of efficiency bins: {}'.format(len(efficiency_data["f"])))
    result_E = {"KE": specGen.Energy(tritium_data["F"])}
    result_E["is_sample"] = tritium_data["is_sample"]


    result_mixed = tritium_data
    result_mixed["F"] = [f-24.5e9 for f in result_mixed["F"]]
    histo.data = result_mixed
    histo.Run()

    return result_E, efficiency_data

def AnalyzeData(tritium_data, efficiency_data, nbins=20, iterations=100, f_min = 24.5e9+1320e6, f_max = 24.5e9+1480e6, make_plots=False):
    i = 0
    KE_min = Energy(f_max)
    KE_max = Energy(f_min)

    energy_resolution_precision = 0.01
    energy_resolution = 25
    meanSigma = energy_resolution

    nCounts = len(tritium_data["KE"])
    nbins_eff = len(efficiency_data["f"])
    print(len(efficiency_data["f"]))
    print(len(efficiency_data["efficiency"]))
    print(len(efficiency_data["efficiency_error"]))

    bins = np.linspace(KE_min, KE_max, nbins+1)
    hist_data, bins = np.histogram(tritium_data["KE"], bins=bins)
    bin_centers = bins[0:-1]+0.5*(bins[1]-bins[0])



    input_data = {"N": list(hist_data), "KE": list(bin_centers), "F": list(np.array(efficiency_data["f"])*1e-9), "Eff": efficiency_data["efficiency"], "Eff_err": efficiency_data["efficiency_error"]}
    print(input_data)

    plt.figure()
    plt.errorbar(input_data['KE'], input_data['N'], yerr=np.sqrt(input_data['N']), fmt='o')
    plt.savefig('binned_spectrum.png')


    # Gamma function: param alpha and beta
    # alpha and beta aren't the mean and std
    # mu = alpha/beta**2
    # sig = sqrt(alpha)/beta
    pystan_config = {
        "model_code": "/host/models/tritium_efficiency_phase_II_analyzer.stan",
        "input_data": {
            "runtime": 1.*seconds_per_year()/12.,
            "KEmin": KE_min,
            "KEmax": KE_max,
            "sigma_alpha": 1./(energy_resolution_precision**2),
            "sigma_beta": 1/(energy_resolution_precision**2*meanSigma),
            "Q_ctr": tritium_endpoint(),
            "Q_std": 80.,
            # "A_s_alpha": 0.1,
            # "A_s_beta": 12000.,
            # "A_b_log_ctr": -16.1181,
            # "A_b_log_std": 1.1749,
            #        bkgd_frac_b: 3.3567
            "Nbins_T": nbins,
            "Nbins_Eff": nbins_eff,
            "mixing_freq": 24.5e9,
            "B_field": 0.95777194923080811
        },
        "iter": max(iterations, 7000),
        "warmup": 5000,
        "n_jobs": 1,
        "interestParams": ['Q', 'sigma', 'A', 'A_s', 'A_b', 'coeff0', 'coeff1', 'KE_sample', 'spectrum_fit', 'F_sample', 'Eff_fitted', 'KE_sample_converted_to_F'],
        "function_files_location": "/host/models"
    }

    pystanProcessor = PyStanSamplingProcessor("pystanProcessor")
    pystanProcessor.Configure(pystan_config)
    pystanProcessor.data = input_data
    pystanProcessor.Run()
    sampling_result = pystanProcessor.results
    print(sampling_result.keys())


    plt.figure()
    f = np.array(input_data['F'])
    c0 = np.mean(np.array(sampling_result['coeff0'])[pystan_config['warmup']:])
    c1 = np.mean(np.array(sampling_result['coeff1'])[pystan_config['warmup']:])
    plt.plot(f, c0+ f* c1)
    plt.errorbar(input_data['F'], input_data['Eff'], yerr=input_data['Eff_err'], fmt='o')
    plt.savefig('fitted_efficiency.png')

    plt.figure()

    e = np.array(sampling_result['KE_sample'])[pystan_config['warmup']:]
    f = np.array(sampling_result['KE_sample_converted_to_F'])[pystan_config['warmup']:]
    a = np.array(sampling_result['spectrum_fit'])[pystan_config['warmup']:]

    plt.subplot(121)
    plt.plot(e, a/np.sum(a)/len(bin_centers)*len(e)*np.sum(hist_data), '.')
    plt.errorbar(input_data['KE'], input_data['N'], yerr=np.sqrt(input_data['N']), fmt='o')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Counts')

    plt.subplot(122)
    plt.plot(f, a/np.sum(a)/len(bin_centers)*len(e)*np.sum(hist_data), '.')
    #plt.errorbar(input_data['KE'], input_data['N'], yerr=np.sqrt(input_data['N']), fmt='o')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig('fitted_spectrum.png')

    if make_plots:
        plot_name = 'aposteriori_distribution_{}_{}_{}_{}'.format(
            nCounts, energy_resolution, energy_resolution_precision, i)
        plot_Sampling(sampling_result, plot_name)

def Energy(f, B=None, Theta=None):
    #print(type(F))
    if B==None:
        B = 0.95777194923080811
    emass_kg = Constants.m_electron()*Constants.e()/(Constants.c()**2)
    if isinstance(f, list):
        gamma = [(Constants.e()*B)/(2.0*TMath.Pi()*emass_kg) * 1/(f) for f in F]
        return [(g -1)*Constants.m_electron() for g in gamma]
    else:
        gamma = (Constants.e()*B)/(2.0*TMath.Pi()*emass_kg) * 1/(f)
        return (gamma -1)*Constants.m_electron()


def plot_Sampling(data, name='aposteriori_distribution'):
    aposteriori_config = {
        "n_bins_x": 100,
        "n_bins_y": 100,
        "variables": ['sigma', 'Q', 'A', 'A_s', 'A_b', 'coeff0', 'coeff1'],
        "title": name,
        "output_path": "results"
    }
    aposterioriPlotter = APosterioriDistribution("posterioriDistrib")
    aposterioriPlotter.Configure(aposteriori_config)
    aposterioriPlotter.data = data
    aposterioriPlotter.Run()

def DoCombinedAnalysis():
    print("Generate fake distorted tritium data and load efficiency data from json file")
    tritium_data, efficiency_data = ProduceAndReadData()
    AnalyzeData(tritium_data, efficiency_data, make_plots=True)
if __name__ == '__main__':

    DoCombinedAnalysis()
