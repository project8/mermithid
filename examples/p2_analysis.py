'''
This script analyzes Project 8 Phase II data to extract the Tritium endpoint energy.
Author: X. Huyan, C. Claessens
Date: April 6, 2020
'''

import matplotlib.pyplot as plt
import argparse

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

from mermithid.misc.Constants import seconds_per_year, tritium_endpoint
from mermithid.processors.IO import MultiChannelCicadaReader
from mermithid.processors.IO import IOCicadaProcessor
from mermithid.processors.TritiumSpectrum.FakeDataGenerator import FakeDataGenerator
from mermithid.processors.TritiumSpectrum import DistortedTritiumSpectrumLikelihoodSampler
from mermithid.processors.misc.TritiumAndEfficiencyBinner import TritiumAndEfficiencyBinner
from mermithid.processors.misc.ComplexLineShape import ComplexLineShape

def ReadMultiChannelData(
    filepath=["/pathto/tritium_run_3_channel_a_some_runs.root",
              "/pathto/tritium_run_3_channel_b_some_runs.root",
              "/pathto/tritium_run_3_channel_c_some_runs.root"]):
    reader_config = {
        "action": "read",
        "N_channels": 3,
        "filename": filepath,
        "rf_roi_min_freqs": [25803125000.0, 25862500000.0, 25921875000.0],
        "channel_transition_freqs": [[0,1.38623121e9+24.5e9],
                                     [1.38623121e9+24.5e9, 1.44560621e9+24.5e9],
                                    [1.44560621e9+24.5e9, 50e9]],
        "object_type": "TMultiTrackEventData",
        "object_name": "multiTrackEvents:Event",
        "use_katydid": False,
        "variables": ['StartFrequency']
    }

    reader = MultiChannelCicadaReader("reader")
    reader.Configure(reader_config)
    reader.Run()
    results = reader.data
    return results
    '''
    The returned "results" above is a dictionary that includes:
    1) raw start frequency from each channel
    2) shifted (by rf_roi_min_freqs) frequency from each channel, with the frequency overlap removed
    3) a list of shifted frequency from all channels with frequency overlap removed (these are the data to analyze)
    '''

def GenerateFakeData(
    eff_path="/pathto/combined_energy_corrected_eff_at_quad_trap_frequencies.json",
    sl_path="/pathto/simplified_scattering_params.txt"):
    specGen_config = {
        "apply_efficiency": False,
        "efficiency_path": eff_path,
        "simplified_lineshape_path": sl_path,
        "detailed_or_simplified_lineshape": "detailed",
        "use_lineshape": True, # if False only gaussian smearing is applied
        "return_frequency": True,
        "scattering_sigma": 18.6,
        "scattering_prob": 0.77,
        "B_field": 0.9578186017836624,
        "S": 4000,
        "n_steps": 1000,
        "A_b": 1e-10
    }

    specGen = FakeDataGenerator("specGen")
    specGen.Configure(specGen_config)
    specGen.Run()
    results = specGen.results
    return results
    '''
    The returned "results" above is a dictionary that includes:
    1) keys: K (energy), F (frequency)
    2) mapped values: energy, frequency
    '''

def EfficiencyBinning(tritium_data):
    tritiumAndEfficiencyBinner_config = {
        "energy_or_frequency": 'frequency',
        "variables": "F",
        "title": "corrected_spectrum",
        "efficiency": "-265.03357206889626 + 6.693200670990694e-07*(x-24.5e9) + -5.795611253664308e-16*(x-24.5e9)^2 + 1.5928835520798478e-25*(x-24.5e9)^3 + 2.892234977030861e-35*(x-24.5e9)^4 + -1.566210147698845e-44*(x-24.5e9)^5",
        'bins': np.linspace(24.5e9+1300e6, 24.5e9+1550e6, 15),
        'fss_bins': False # If fss_bins is True, bins is ignored and overridden
     }

    tritiumAndEfficiencyBinner = TritiumAndEfficiencyBinner("tritiumAndEfficiencyBinner")
    tritiumAndEfficiencyBinner.Configure(tritiumAndEfficiencyBinner_config)
    tritiumAndEfficiencyBinner.data = tritium_data
    tritiumAndEfficiencyBinner.Run()
    results = tritiumAndEfficiencyBinner.results
    return results

def FitComplexLineshape():
    reader_config = {
        "action": "read",
        "filename": "/host/ShallowTrap8603-8669.root",
        "object_type": "TMultiTrackEventData",
        "object_name": "multiTrackEvents:Event",
        "use_katydid": False,
        "variables": ['StartTimeInAcq','StartFrequency']
    }

    complexLineShape_config = {
        'bins_choice': np.linspace(0,90e6,1000),
        'gases': ["H2","Kr"],
        'max_scatters': 20,
        'num_points_in_std_array': 10000,
        'RF_ROI_MIN': 25850000000.0,
        'B_field': 0.957810722501,
        'shake_spectrum_parameters_json_path': '/host/shake_spectrum_parameters.json',
        'path_to_osc_strengths_files': '/host/'
    }

    reader = IOCicadaProcessor("reader")
    complexLineShape = ComplexLineShape("complexLineShape")
    reader.Configure(reader_config)
    complexLineShape.Configure(complexLineShape_config)
    reader.Run()
    data = reader.data
    complexLineShape.data = data
    complexLineShape.Run()
    results = complexLineShape.results
    return results

def AnalyzeData(frequency_data, cl_fit_parameters, sl_path="/pathto/simplified_scattering_params.txt", efficiency, efficiency_error):
'''
All analysis results are saved and the following plots need to be saved to watch "live":
1) tritium spectrum
2) tritium spectrum fits
3) tritium endpoint posterior
'''

def DoAnalysis(real_data=True):
    tritium_data = ReadMultiChannelData() if real_data else GenerateFakeData()
    cl_output = FitComplexLineshape()
    eff_output = EfficiencyBinning(tritium_data)
    AnalyzeData(tritium_data, cl_output, eff_output)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description="This script analyzes Project 8 Phase II data to extract the Tritium endpoint energy"
    )
    REAL_DATA_NAME = "real"
    arg_parser.add_argument(
        "-d",
        "--data-type",
        choices=(REAL_DATA_NAME, "fake"),
        required=True,
        help="Type of data to analyze."
    )
    DoAnalysis(arg_parser.parse_args().data_type == REAL_DATA_NAME)
