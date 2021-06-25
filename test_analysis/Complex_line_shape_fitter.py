'''
Reads in data and fits it with complex lineshape model.
Author: E. Machado, Y.-H. Sun, E. Novitski
Date: 4/8/20
'''

import numpy as np
import unittest
import matplotlib.pyplot as plt
import ROOT as r
import os

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

class ComplexLineShapeTests(unittest.TestCase):

    def test_complex_lineshape(self):
        from mermithid.processors.IO import IOCicadaProcessor
        from mermithid.processors.misc.MultiGasComplexLineShape import MultiGasComplexLineShape


        reader_config = {
            "action": "read",
            "filename": "/host/shallow_trap_high_stats_above_10600_channel_a_concat.root",
            "object_type": "TMultiTrackEventData",
            "object_name": "multiTrackEvents:Event",
            "use_katydid": False,
            "variables": ['StartTimeInAcq','StartFrequency']
        }

        complexLineShape_config = {
            'bins_choice': np.linspace(0e6, 100e6, 1000),
            'gases': ["H2", "He"], # Ar, Kr
            'max_scatters': 20,
            'fixed_scatter_proportion': True,
            # When fixed_scatter_proportion is True, set the scatter proportion for the gases below
            'gas_scatter_proportion': [0.8, 0.2],#0.827, 0.076, 0.068, 0.028 # 0.75, 0.25
            'partially_fixed_scatter_proportion': False,
            'free_gases': ["H2", "He"],
            'fixed_gases': ["Ar", "Kr"],
            'scatter_proportion_for_fixed_gases': [0.018, 0.039],
            'fixed_survival_probability': False,
            # When option fixed_survival_probability is True, assign the survival probability below
            'survival_prob': 15/16., # assuming total cross section for elastic scattering is 1/10 of inelastic scattering
            # configure the resolution functions: simulated_resolution, gaussian_resolution, gaussian_lorentzian_composite_resolution, elevated_gaussian, composite_gaussian, composite_gaussian_pedestal_factor, composite_gaussian_scaled, simulated_resolution_scaled, 'simulated_resolution_scaled_fit_scatter_peak_ratio', 'gaussian_resolution_fit_scatter_peak_ratio'
            'resolution_function': 'gaussian_resolution_fit_scatter_peak_ratio',
            # specific choice of parameters in the gaussian lorentzian composite resolution function
            'recon_eff_param_a': 0.005569990343215976,
            'recon_eff_param_b': 0.351,
            'recon_eff_param_c': 0.546,
            'ratio_gamma_to_sigma': 0.8,
            'gaussian_proportion': 1.,
            # if the resolution function is composite gaussian
            'sigma_array': [5.01, 13.33, 15.40, 11.85],
            'A_array': [0.076, 0.341, 0.381, 0.203],
            #parameter for simulated resolution scaled resolution 
            'fit_recon_eff': False,
            #parameters for simulated resolution scaled with scatter peak ratio fitted
            #choose the parameters you want to fix from ['B field','amplitude','width scale factor', 'survival probability','scatter peak ratio param b', 'scatter peak ratio param c'] plus the gas scatter fractions as ['H2 scatter fraction'],
            'fixed_parameter_names': ['survival probability', 'H2 scatter fraction'],
            'fixed_parameter_values': [1.0, 0.896],        
            # This is an important parameter which determines how finely resolved
            # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
            'num_points_in_std_array': 10000,
            'RF_ROI_MIN': 25850000000.0,
            # shake_spectrum_parameters.json and oscillator strength data can be found at https://github.com/project8/scripts/tree/master/yuhao/line_shape_fitting/data
            'shake_spectrum_parameters_json_path': '../mermithid/misc/shake_spectrum_parameters.json',
            'path_to_osc_strengths_files': '/host/',
            'path_to_scatter_spectra_file': '/host/',
            'path_to_ins_resolution_data_txt': '/host/res_all_conversion_max15.5_alltraps.txt'
        }

        b = IOCicadaProcessor("reader")
        b.Configure(reader_config)
        b.Run()
        data = b.data
        logger.info("Data extracted = {}".format(data.keys()))
        for key in data.keys():
            logger.info("{} -> size = {}".format(key,len(data[key])))
        
        complexLineShape = MultiGasComplexLineShape("complexLineShape")
        
        complexLineShape.Configure(complexLineShape_config)       

        complexLineShape.data = data

        complexLineShape.Run()

        results = complexLineShape.results
        logger.info(results['output_string'])

        # plot fit with shake spectrum
        plt.rcParams.update({'font.size': 15})
        plt.figure(figsize=(15,9))
        plt.step(
        results['bins_Hz']/1e9, results['data_hist_freq'],
        label = 'data\n total counts = {}\n'.format(len(data['StartFrequency']))
        )
        plt.plot(results['bins_Hz']/1e9, results['fit_Hz'], label = results['output_string'], alpha = 0.7)
        plt.legend(loc = 'upper left', fontsize = 12)
        plt.xlabel('frequency GHz')
        plot_title = 'fit ftc march with gases: {},\n resolution function: {},\n file for simulated resolution data: {}'.format(complexLineShape_config['gases'], complexLineShape_config['resolution_function'], os.path.basename(complexLineShape_config['path_to_ins_resolution_data_txt']))
        if complexLineShape_config['resolution_function'] == 'composite_gaussian_scaled':
            plot_title = 'fit ftc march with gases: {},\n scatter proportion: {},\n resolution function: {},\n sigma_array: {},\n A_array: {},\n'.format(complexLineShape_config['gases'], complexLineShape_config['gas_scatter_proportion'], complexLineShape_config['resolution_function'], complexLineShape_config['sigma_array'], complexLineShape_config['A_array'])
        plt.title(plot_title)
        plt.tight_layout()
        plt.savefig('/host/plots/fit_shallow_trap_above_10600_with_gaussian_resolution.png'.format(len(complexLineShape_config['gases'])))

if __name__ == '__main__':

    unittest.main()