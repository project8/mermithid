'''
This scripts aims at testing Tritium specific processors.
Author: A. Ziegler
Date: Aug 31 2019
'''

import unittest
import sys, os

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

from mermithid.processors.TritiumSpectrum import TritiumSpectrumProcessor, DistortedTritiumSpectrumLikelihoodSampler
from morpho.processors.plots import Histogram
from mermithid.misc.Constants import seconds_per_year, tritium_endpoint
from mermithid.misc import Constants

import ROOT
from ROOT import TH1F
from ROOT import TMath
from ROOT import RooFit
import numpy as np

from morpho.processors.sampling.PyStanSamplingProcessor import PyStanSamplingProcessor
from mermithid.processors.misc.EfficiencyCorrector import EfficiencyCorrector
from morpho.processors.plots.APosterioriDistribution import APosterioriDistribution

import matplotlib.pyplot as plt

def ProduceData(counts=10000):


    specGen_config = {
        "volume": 7e-6*1e-2, # [m3]
        "density": 3e17, # [1/m3]
        "duration": 1.*seconds_per_year()/12.*1, # [s]
        "neutrino_mass" :0, # [eV]
        "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
        "frequency_window": [-100e6, +100e6], #[Fmin, Fmax]
        "energy_or_frequency": "frequency",
        #"energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
        "background": 1e-6, # [counts/eV/s]
        "energy_resolution": 5,# [eV]
        "frequency_resolution": 50e3,# [Hz]
        "mode": "generate",
        "varName": "F",
        "iter": counts,
        "interestParams": ["F"],
        "fixedParams": {"m_nu": 0},
        "options": {"snr_efficiency": False, "channel_efficiency":False, "smearing": True},
        "snr_efficiency_coefficients": [-265.03357206889626, 6.693200670990694e-07, -5.795611253664308e-16, 1.5928835520798478e-25, 2.892234977030861e-35, -1.566210147698845e-44], #[-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
        "channel_central_frequency": 1400e6,
        "mixing_frequency": 24.5e9
    }

    specGen = DistortedTritiumSpectrumLikelihoodSampler("specGen")
    specGen.Configure(specGen_config)
    specGen.Run()
    tritium_data = specGen.data
    print('Number of events: {}'.format(len(tritium_data["F"])))

    return tritium_data

def CorrectData(input_data, nbins = 100, F_min = 24.5e9 + 1320e6,
F_max = 24.5e9 + 1480e6, mode = 'unbinned', asInteger = False,
energy_or_frequency = 'energy', histogram_or_dictionary = 'histogram'):

    effCorr_config = {

        "variables": "F",
        "range": [F_min, F_max],
        "n_bins_x": nbins,
        "title": "corrected_spectrum",
        #"efficiency": "-265.03357206889626 + 6.693200670990694e-07*(x-24.5e9) + -5.795611253664308e-16*(x-24.5e9)^2 + 1.5928835520798478e-25*(x-24.5e9)^3 + 2.892234977030861e-35*(x-24.5e9)^4 + -1.566210147698845e-44*(x-24.5e9)^5",
        "mode": mode,
        "energy_or_frequency": energy_or_frequency,
        "histogram_or_dictionary": histogram_or_dictionary,
        "asInteger": asInteger,

    }

    effCorr = EfficiencyCorrector("effCorr")
    effCorr.Configure(effCorr_config)
    effCorr.data = input_data
    effCorr.Run()
    result = effCorr.corrected_data

    return result

def DoStanFit(iter = 1):

    import json, io

    F_min, F_max = 24.5e9 + 1320e6, 24.5e9 + 1480e6
    KE_min = Energy(F_max)
    KE_max = Energy(F_min)
    nbins = 100

    fit_result = {'Q': [], 'sigmaQ': []}
    specGen = DistortedTritiumSpectrumLikelihoodSampler("specGen")
    effCorr = EfficiencyCorrector("effCorr")

    energy_resolution_precision = 0.01
    energy_resolution = 10
    meanSigma = energy_resolution
    iterations = 1
    specFit = PyStanSamplingProcessor("specFit")

    specGen_config = {
        "volume": 7e-6*1e-2, # [m3]
        "density": 3e17, # [1/m3]
        "duration": 1.*seconds_per_year()/12.*1, # [s]
        "neutrino_mass" :0, # [eV]
        "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
        "frequency_window": [-100e6, +100e6], #[Fmin, Fmax]
        "energy_or_frequency": "frequency",
        #"energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
        "background": 1e-6, # [counts/eV/s]
        "energy_resolution": 5,# [eV]
        "frequency_resolution": 50e3,# [Hz]
        "mode": "generate",
        "varName": "F",
        "iter": 10000,
        "interestParams": ["F"],
        "fixedParams": {"m_nu": 0},
        "options": {"snr_efficiency": True, "channel_efficiency":False, "smearing": True},
        "snr_efficiency_coefficients": [-265.03357206889626, 6.693200670990694e-07, -5.795611253664308e-16, 1.5928835520798478e-25, 2.892234977030861e-35, -1.566210147698845e-44], #[-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
        "channel_central_frequency": 1400e6,
        "mixing_frequency": 24.5e9
    }
    effCorr_config = {

        "variables": "F",
        "range": [F_min, F_max],
        "n_bins_x": nbins,
        "title": "corrected_spectrum",
        "efficiency": "-265.03357206889626 + 6.693200670990694e-07*(x-24.5e9) + -5.795611253664308e-16*(x-24.5e9)^2 + 1.5928835520798478e-25*(x-24.5e9)^3 + 2.892234977030861e-35*(x-24.5e9)^4 + -1.566210147698845e-44*(x-24.5e9)^5",
        "mode": 'unbinned',
        "energy_or_frequency": 'energy',
        "histogram_or_dictionary": 'dictionary',
        "asInteger": True,

    }

    effCorr.Configure(effCorr_config)

    for n in range(iter):
        print("Generate fake distorted tritium data and test efficiency correction.")
        specGen.Configure(specGen_config)
        specGen.Run()
        effCorr.data = specGen.data
        effCorr.Run()
        fit_config = {
            "model_code": "/Users/ziegler/docker_share/stan_models/tritium_phase_II_analyzer.stan",
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
                "Nbins": nbins,
                "KE": effCorr.corrected_data["KE"],
                "N": effCorr.corrected_data["N"]
            },
            "iter": max(iterations, 10000),
            "warmup": 5000,
            "n_jobs": 1,
            "interestParams": ['Q', 'sigma', 'A', 'A_s', 'A_b'],
            "function_files_location": "/Users/ziegler/docker_share/stan_models"
        }
        specFit.Configure(fit_config)
        specFit.Run()
        fit_result['Q'].append(np.mean(specFit.results['Q']))
        fit_result['sigmaQ'].append(np.mean(specFit.results['sigma']))

    temp = []
    if not os.path.isfile('results_stan_undistorted.json'):
        temp.append(fit_result)
        with open('results_stan_undistorted.json', mode='w') as outfile:
            json.dump(temp, outfile)
    else:
        with open('results_stan_undistorted.json') as infile:
            file_data = json.load(infile)

        file_data.append(fit_result)
        with open('results_stan_undistorted.json', mode='w') as outfile:
            json.dump(file_data, outfile)

def DoRooFit(iter = 1):

    import json, io

    F_min, F_max = 24.5e9 + 1320e6, 24.5e9 + 1480e6
    KE_min = Energy(F_max)
    KE_max = Energy(F_min)
    nbins = 100

    fit_result = {'Q': [], 'sigmaQ': [], 'chi2': []}
    specGen = DistortedTritiumSpectrumLikelihoodSampler("specGen")
    effCorr = EfficiencyCorrector("effCorr")
    specFit = TritiumSpectrumProcessor("SpecFit")

    specGen_config = {
        "volume": 7e-6*1e-2, # [m3]
        "density": 3e17, # [1/m3]
        "duration": 1.*seconds_per_year()/12.*1, # [s]
        "neutrino_mass" :0, # [eV]
        "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
        "frequency_window": [-100e6, +100e6], #[Fmin, Fmax]
        "energy_or_frequency": "frequency",
        #"energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
        "background": 1e-6, # [counts/eV/s]
        "energy_resolution": 5,# [eV]
        "frequency_resolution": 50e3,# [Hz]
        "mode": "generate",
        "varName": "F",
        "iter": 10000,
        "interestParams": ["F"],
        "fixedParams": {"m_nu": 0},
        "options": {"snr_efficiency": True, "channel_efficiency":False, "smearing": True},
        "snr_efficiency_coefficients": [-265.03357206889626, 6.693200670990694e-07, -5.795611253664308e-16, 1.5928835520798478e-25, 2.892234977030861e-35, -1.566210147698845e-44], #[-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
        "channel_central_frequency": 1400e6,
        "mixing_frequency": 24.5e9
    }
    effCorr_config = {

        "variables": "F",
        "range": [F_min, F_max],
        "n_bins_x": nbins,
        "title": "corrected_spectrum",
        "efficiency": "-265.03357206889626 + 6.693200670990694e-07*(x-24.5e9) + -5.795611253664308e-16*(x-24.5e9)^2 + 1.5928835520798478e-25*(x-24.5e9)^3 + 2.892234977030861e-35*(x-24.5e9)^4 + -1.566210147698845e-44*(x-24.5e9)^5",
        "mode": 'unbinned',
        "energy_or_frequency": 'energy',
        "histogram_or_dictionary": 'histogram',
        "asInteger": False,

    }
    fit_config = {

        "volume": 7e-6*1e-2, # [m3]
        "density": 3e17, # [1/m3]
        "duration": 1.*seconds_per_year()/12.*1, # [s]
        "neutrino_mass" :0, # [eV]
        #"energy_window": [KE_min,KE_max], # [KEmin,KEmax]
        "background": 1e-6, # [counts/eV/s]
        "energy_resolution": 36,# [eV]
        "mode": "fit",
        "varName": "KE",
        "iter": 100,
        "interestParams": ["endpoint"],
        "paramRange": {
            "KE": [KE_min, KE_max]
        },
        "fixedParams": {
            "m_nu": 0,
            "widthSmearing": 36
        },
        #"fixed_m_nu": False,
        "binned": True,
        "make_fit_plot": False

        #"frequency_window": [-100e6, +100e6], #[Fmin, Fmax]
        #"energy_or_frequency": "frequency",
        #"energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
        #"frequency_resolution": 50e3,# [Hz]
        #"options": {"snr_efficiency": True, "channel_efficiency":False, "smearing": True},
        #"snr_efficiency_coefficients": [-265.03357206889626, 6.693200670990694e-07, -5.795611253664308e-16, 1.5928835520798478e-25, 2.892234977030861e-35, -1.566210147698845e-44], #[-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
        #"channel_central_frequency": 1400e6,
        #"mixing_frequency": 24.5e9

    }

    effCorr.Configure(effCorr_config)
    specFit.Configure(fit_config)

    for n in range(iter):
        print("Generate fake distorted tritium data and test efficiency correction.")
        specGen.Configure(specGen_config)
        specGen.Run()
        effCorr.data = specGen.data
        effCorr.Run()
        specFit.data = effCorr.corrected_data
        specFit.Run()
        fit_result['Q'].append(specFit.result['endpoint'])
        fit_result['sigmaQ'].append(specFit.result['error_endpoint'])
        fit_result['chi2'].append(specFit.result['chi2'])

    temp = []
    if not os.path.isfile('results_roofit_undistorted.json'):
        temp.append(fit_result)
        with open('results_roofit_undistorted.json', mode='w') as outfile:
            json.dump(temp, outfile)
    else:
        with open('results_roofit_undistorted.json') as infile:
            file_data = json.load(infile)

        file_data.append(fit_result)
        with open('results_roofit_undistorted.json', mode='w') as outfile:
            json.dump(file_data, outfile)

def Energy(f, B=None, Theta=None):
    #print(type(F))
    if B==None:
        B = 0.95777194923080811
    emass_kg = Constants.m_electron()*Constants.e()/(Constants.c()**2)
    if isinstance(f, list):
        gamma = [(Constants.e()*B)/(2.0*TMath.Pi()*emass_kg) * 1/(F) for F in f]
        return [(g -1)*Constants.m_electron() for g in gamma]
    else:
        gamma = (Constants.e()*B)/(2.0*TMath.Pi()*emass_kg) * 1/(f)
        return (gamma -1)*Constants.m_electron()

def DoAnalysis(iter = 1):

    DoRooFit()
    DoStanFit()

if __name__ == '__main__':

    DoAnalysis()
