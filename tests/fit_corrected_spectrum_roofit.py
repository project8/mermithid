'''
This scripts aims at testing Tritium specific processors.
Author: A. Ziegler
Date: Aug 31 2019
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

from mermithid.processors.TritiumSpectrum import TritiumSpectrumProcessor, DistortedTritiumSpectrumLikelihoodSampler
from morpho.processors.plots import Histogram
from mermithid.misc.Constants import seconds_per_year, tritium_endpoint
from mermithid.misc import Constants

from ROOT import TH1F
from ROOT import TMath

#from morpho.processors.sampling.PyStanSamplingProcessor import PyStanSamplingProcessor
from mermithid.processors.misc.EfficiencyCorrector import EfficiencyCorrector
#from morpho.processors.plots.APosterioriDistribution import APosterioriDistribution

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
        "options": {"snr_efficiency": True, "channel_efficiency":False, "smearing": True},
        "snr_efficiency_coefficients": [-265.03357206889626, 6.693200670990694e-07, -5.795611253664308e-16, 1.5928835520798478e-25, 2.892234977030861e-35, -1.566210147698845e-44], #[-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
        "channel_central_frequency": 1400e6,
        "mixing_frequency": 24.5e9
    }

    specGen = DistortedTritiumSpectrumLikelihoodSampler("specGen")
    #histo = Histogram("histo")

    specGen.Configure(specGen_config)
    #histo.Configure(histo_plot)

    specGen.Run()
    tritium_data = specGen.data
    print('Number of events: {}'.format(len(tritium_data["F"])))
    #result_E = {"KE": specGen.Energy(tritium_data["F"])}
    #result_E["is_sample"] = tritium_data["is_sample"]


    #result_mixed = tritium_data
    #result_mixed["F"] = [f-24.5e9 for f in result_mixed["F"]]
    #histo.data = result_mixed
    #histo.Run()

    return tritium_data


def CorrectData(input_data, nbins = 100, F_min = 24.5e9 + 1320e6, F_max = 24.5e9 + 1480e6, mode = "binned"):

    effCorr_config = {

        "variables": "F",
        "range": [F_min, F_max],
        "n_bins_x": nbins,
        "title": "corrected_spectrum",
        "efficiency": "-265.03357206889626 + 6.693200670990694e-07*(x-24.5e9) + -5.795611253664308e-16*(x-24.5e9)^2 + 1.5928835520798478e-25*(x-24.5e9)^3 + 2.892234977030861e-35*(x-24.5e9)^4 + -1.566210147698845e-44*(x-24.5e9)^5",
        "mode": mode,

    }

    histo = TH1F("histo", "histo", nbins, F_min, F_max)
    corr_histo = TH1F("corr_hist", "corr_histo", nbins, Energy(F_max), Energy(F_min))

    for val in input_data["F"]:
        histo.Fill(val)

    bin_centers = []
    counts = []

    for i in range(nbins):
        counts.append(int(histo.GetBinContent(i)))
        bin_centers.append(histo.GetBinCenter(i))

    binned_data = {"N": counts, "F": bin_centers}

    effCorr = EfficiencyCorrector("effCorr")
    effCorr.Configure(effCorr_config)
    effCorr.data = binned_data
    effCorr.Run()

    if mode == "binned":
        corrected_E_data = {"KE": Energy(effCorr.corrected_data["F"])}
        corrected_E_data["N"] = effCorr.corrected_data["N"]

        for i in range(len(corrected_E_data["KE"])):
            corr_histo.SetBinContent(i, corrected_E_data["N"][-i])
        return corr_histo

    elif mode == "unbinned":
        corrected_E_data = {"KE": Energy(effCorr.corrected_data["F"])}
        return corrected_E_data

def AnalyzeData(tritium_data, nbins=100, iterations=100, F_min = 24.5e9 + 1320e6, F_max = 24.5e9 + 1480e6):

    KE_min = Energy(F_max)
    KE_max = Energy(F_min)

    #energy_resolution_precision = 0.01
    energy_resolution = 5
    #meanSigma = energy_resolution

    #nCounts = len(tritium_data["KE"])

    fit_config = {

        "volume": 7e-6*1e-2, # [m3]
        "density": 3e17, # [1/m3]
        "duration": 1.*seconds_per_year()/12.*1, # [s]
        "neutrino_mass" :0, # [eV]
        #"energy_window": [KE_min,KE_max], # [KEmin,KEmax]
        "background": 1e-6, # [counts/eV/s]
        "energy_resolution": energy_resolution,# [eV]
        "mode": "fit",
        "varName": "KE",
        "iter": iterations,
        "interestParams": ["KE"],
        "paramRange": {
            "KE": [KE_min, KE_max]
        },
        "fixedParams": {
            "m_nu": 0,
            "widthSmearing": energy_resolution
        },
        #"fixed_m_nu": False,
        "binned": True,
        "make_fit_plot": True

        #"frequency_window": [-100e6, +100e6], #[Fmin, Fmax]
        #"energy_or_frequency": "frequency",
        #"energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
        #"frequency_resolution": 50e3,# [Hz]
        #"options": {"snr_efficiency": True, "channel_efficiency":False, "smearing": True},
        #"snr_efficiency_coefficients": [-265.03357206889626, 6.693200670990694e-07, -5.795611253664308e-16, 1.5928835520798478e-25, 2.892234977030861e-35, -1.566210147698845e-44], #[-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
        #"channel_central_frequency": 1400e6,
        #"mixing_frequency": 24.5e9

    }

    specFit = TritiumSpectrumProcessor("SpecFit")
    specFit.Configure(fit_config)
    specFit.data = tritium_data
    specFit.Run()

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

def DoAnalysis():
    print("Generate fake distorted tritium data and test efficiency correction.")
    tritium_data = ProduceData()
    corrected_tritium_data = CorrectData(tritium_data)
    AnalyzeData(corrected_tritium_data)
if __name__ == '__main__':

    DoAnalysis()
