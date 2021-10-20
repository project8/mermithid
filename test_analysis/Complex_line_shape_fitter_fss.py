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
from scipy import integrate , signal, interpolate
import json

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

class ComplexLineShapeTests(unittest.TestCase):

    def test_complex_lineshape(self):
        from mermithid.processors.IO import IOCicadaProcessor
        from mermithid.processors.misc.MultiGasComplexLineShape import MultiGasComplexLineShape


        reader_config = {
            "action": "read",
            "filename": "/host/october_2019_kr_calibration_channel_b_merged.root",
            "object_type": "TMultiTrackEventData",
            "object_name": "multiTrackEvents:Event",
            "use_katydid": False,
            "variables": ['StartTimeInAcq','StartFrequency']
        }

        complexLineShape_config = {
            'bins_choice': np.linspace(0e6, 100e6, 1000),
            'gases': ["H2", "He", "Kr"], # "Ar", "Kr" # "Kr" for fss
            'fix_gas_composition': True,
            'fix_width_scale_factor': True,
            'factor': 0.4934,
            'scatter_fractions_for_gases': [0.950, 0.02],
            'max_scatters': 20,
            'fixed_scatter_proportion': True,
            # When fixed_scatter_proportion is True, set the scatter proportion for the gases below
            'gas_scatter_proportion': [0.8, 0.2],#0.827, 0.076, 0.068, 0.028 # 0.75, 0.25
            'partially_fixed_scatter_proportion': False,
            'free_gases': ["H2", "He"],
            'fixed_gases': ["Ar", "Kr"],
            'scatter_proportion_for_fixed_gases': [0.018, 0.039],
            'use_radiation_loss': True,
            'sample_ins_res_errors': False,
            'fixed_survival_probability': False,
            # When option fixed_survival_probability is True, assign the survival probability below
            'survival_prob': 15/16., # assuming total cross section for elastic scattering is 1/10 of inelastic scattering
            # configure the resolution functions: simulated_resolution, gaussian_resolution, gaussian_lorentzian_composite_resolution, elevated_gaussian, composite_gaussian, composite_gaussian_pedestal_factor, composite_gaussian_scaled, simulated_resolution_scaled, 'simulated_resolution_scaled_fit_scatter_peak_ratio', 'gaussian_resolution_fit_scatter_peak_ratio'
            'resolution_function': 'simulated_resolution_scaled_fit_scatter_peak_ratio',
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
            #choose the parameters you want to fix from ['B field','amplitude', 'width scale factor', 'survival probability','scatter peak ratio param b', 'scatter peak ratio param c'] plus the gas scatter fractions as ['H2 scatter fraction'],
            'fixed_parameter_names': ['survival probability', 'width scale factor', 'H2 scatter fraction', 'He scatter fraction'], #, 'width scale factor', 'H2 scatter fraction', 'He scatter fraction', 'Ar scatter fraction'
            'fixed_parameter_values': [1.0, 1.0, 0.950, 0.02],   #[1.0, 1.0, 0.886, 0.02, 0.06]   
            # This is an important parameter which determines how finely resolved
            # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
            'num_points_in_std_array': 4000,
            'RF_ROI_MIN': 24.5e9 + 1.40812680e+09 - 50e6, #25850000000.0
            # shake_spectrum_parameters.json and oscillator strength data can be found at https://github.com/project8/scripts/tree/master/yuhao/line_shape_fitting/data
            'shake_spectrum_parameters_json_path': '../mermithid/misc/shake_spectrum_parameters.json',
            'path_to_osc_strengths_files': '/host/',
            'path_to_scatter_spectra_file': '/host/',
            'path_to_ins_resolution_data_txt': '/host/trap_combined/all_res_cf12.400.txt'
        }

#         b = IOCicadaProcessor("reader")
#         b.Configure(reader_config)
#         b.Run()
#         data = b.data

        fss_real_data_path = '/host/analysis_results_fine_q300.json'
        with open(fss_real_data_path, 'r') as infile:
            data_selection = json.load(infile)['channel_a']['data_selection']
        
        start_freqs_vs_fss, run_durations_vs_fss, run_temps, track_lengths, event_lengths, slopes, nups, min_freqs = data_selection
        data = {}
        data['StartFrequency'] = np.array(start_freqs_vs_fss['0.0']) - 1.40812680e+09 + 50e6
        logger.info("Data extracted = {}".format(data.keys()))
        for key in data.keys():
            logger.info("{} -> size = {}".format(key,len(data[key])))
        
        complexLineShape = MultiGasComplexLineShape("complexLineShape")
        
        width_scale_factor_fit = {}
        width_scale_factor_fit_err = {}
        output_file = open('/host/fss_b_and_c_for_different_gas_compositions.txt', 'a')
        fixed_para_values_array = [[1.0, 1.0, 0.950, 0.02]]
        for i in range(len(fixed_para_values_array)):
            fixed_para_values = fixed_para_values_array[i]

            complexLineShape_config['path_to_ins_resolution_data_txt'] = '/host/fss_resolutions/all_res_cf{}.txt'.format('12.800')

            complexLineShape_config['fixed_parameter_values'] = fixed_para_values

            complexLineShape.Configure(complexLineShape_config)       

            complexLineShape.data = data

            complexLineShape.Run()

            results = complexLineShape.results
            
            logger.info(results['output_string'])
            #logger.info('\n'+str(results['correlation_matrix']))

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
            if complexLineShape_config['resolution_function'] == 'simulated_resolution_scaled_fit_scatter_peak_ratio':
                plot_title = 'data file:{},\n gases: {},\n resolution function: {}({}),\n fixed parameters:\n {}'.format(os.path.basename(reader_config['filename']),complexLineShape_config['gases'], complexLineShape_config['resolution_function'], os.path.basename(complexLineShape_config['path_to_ins_resolution_data_txt']), complexLineShape_config['fixed_parameter_names'])
            if complexLineShape_config['resolution_function'] == 'gaussian_resolution_fit_scatter_peak_ratio':
                plot_title = 'data file:{},\n gases: {},\n resolution function: {},\n fixed parameters:\n {}'.format(os.path.basename(reader_config['filename']),complexLineShape_config['gases'], complexLineShape_config['resolution_function'], complexLineShape_config['fixed_parameter_names'])
            plt.title(plot_title)
            plt.tight_layout()
            #plt.savefig('/host/plots/fit_FTC_march_with_simulated_resolution_cf{}_sp_1.0_width_factor_1.0.png'.format(file_cf))
            plt.savefig('/host/plots/fit_FSS_with_simulated_resolution_gas_composition_variation_{}.png'.format(i))# March_FTC
            output_file.write('H2 fraction: {}, He fraction: {}, b: {}, b_err: {}, c: {}, c_err: {}\n'.format(complexLineShape_config['fixed_parameter_values'][2], complexLineShape_config['fixed_parameter_values'][3], results['scatter_peak_ratio_b_fit'], results['scatter_peak_ratio_b_fit_err'], results['scatter_peak_ratio_c_fit'], results['scatter_peak_ratio_c_fit_err']))
        output_file.close()

        

if __name__ == '__main__':

    unittest.main()