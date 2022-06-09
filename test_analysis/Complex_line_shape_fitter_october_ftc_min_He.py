'''
Reads in data and fits it with complex lineshape model.
Author: E. Machado, Y.-H. Sun, E. Novitski
Date: 4/8/20
'''

import numpy as np
import unittest
import sys
sys.path.remove('/home/ys633/.local/lib/python3.6/site-packages')
import matplotlib.pyplot as plt
sys.path.append('/home/ys633/.local/lib/python3.6/site-packages')
import ROOT as r
import os
from scipy import integrate , signal, interpolate
import json
import time

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

class ComplexLineShapeTests(unittest.TestCase):

    def test_complex_lineshape(self):
        from mermithid.processors.IO import IOCicadaProcessor
        from mermithid.processors.misc.MultiGasComplexLineShape import MultiGasComplexLineShape


        reader_config = {
            "action": "read",
            "filename": "/home/ys633/lineshape_fitting/mermithid_share/october_2019_kr_calibration_channel_b_merged.root",
            "object_type": "TMultiTrackEventData",
            "object_name": "multiTrackEvents:Event",
            "use_katydid": False,
            "variables": ['StartTimeInAcq','StartFrequency']
        }

        complexLineShape_config = {
            'bins_choice': np.linspace(0e6, 100e6, 1000),
            'gases': ["H2", "He", "Ar", "Kr"], # "Ar", "Kr" # "Kr" for fss ["H2", "He", "Ar", "Kr"]
            'fix_gas_composition': True,
            'fix_width_scale_factor': False,
            'scatter_fractions_for_gases': [0.817, 0.07, 0.08],
            'factor': 0.4934,
            'max_scatters': 20,
            'fixed_scatter_proportion': True,
            # When fixed_scatter_proportion is True, set the scatter proportion for the gases below
            'gas_scatter_proportion': [0.8, 0.2],#0.827, 0.076, 0.068, 0.028 # 0.75, 0.25
            'partially_fixed_scatter_proportion': False,
            'free_gases': ["H2", "He"],
            'fixed_gases': ["Ar", "Kr"],
            'use_radiation_loss': True,
            'sample_ins_res_errors': False,
            'fixed_survival_probability': False,
            # When option fixed_survival_probability is True, assign the survival probability below
            'survival_prob': 15/16., # assuming total cross section for elastic scattering is 1/10 of inelastic scattering
            # configure the resolution functions: simulated_resolution, gaussian_resolution, gaussian_lorentzian_composite_resolution, elevated_gaussian, composite_gaussian, composite_gaussian_pedestal_factor, composite_gaussian_scaled, simulated_resolution_scaled, 'simulated_resolution_scaled_fit_scatter_peak_ratio', 'gaussian_resolution_fit_scatter_peak_ratio'
            'resolution_function': 'simulated_resolution_scaled_fit_scatter_peak_ratio2',
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
            'fixed_parameter_names': ['survival probability', 'width scale factor', 'H2 scatter fraction', 'He scatter fraction', 'Ar scatter fraction'], #, 'width scale factor', 'H2 scatter fraction', 'He scatter fraction', 'Ar scatter fraction'
            'fixed_parameter_values': [1.0, 1.0, (0.233+0.913)/2 + 0.674/2, 0.674/2 - 0.674/2, (0.051+0.104)/2],   #[1.0, 1.0, 0.886, 0.02, 0.06]   
            # This is an important parameter which determines how finely resolved
            # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
            'num_points_in_std_array': 4000,
            'RF_ROI_MIN': 25859375000.0, #24.5e9 + 1.40812680e+09 - 50e6, #25859375000.0, #24.5e9 + 1.40812680e+09 - 50e6, #25850000000.0
            # shake_spectrum_parameters.json and oscillator strength data can be found at https://github.com/project8/scripts/tree/master/yuhao/line_shape_fitting/data
            'shake_spectrum_parameters_json_path': '../mermithid/misc/shake_spectrum_parameters.json',
            'path_to_osc_strengths_files': '/home/ys633/lineshape_fitting/mermithid_share/',
            'path_to_scatter_spectra_file': '/home/ys633/lineshape_fitting/mermithid_share/',
            'path_to_ins_resolution_data_txt': '/host/October_FTC_resolution/all_res_cf14.300.txt',
            'rad_loss_path':'/home/ys633/lineshape_fitting/mermithid_share/',
            'path_to_quad_trap_eff_interp':'/home/ys633/lineshape_fitting/mermithid_share/quad_interps.npy'
        }

        b = IOCicadaProcessor("reader")
        b.Configure(reader_config)
        b.Run()
        data = b.data

#         fss_real_data_path = '/host/analysis_results_fine_q300.json'
#         with open(fss_real_data_path, 'r') as infile:
#             data_selection = json.load(infile)['channel_a']['data_selection']
#         
#         start_freqs_vs_fss, run_durations_vs_fss, run_temps, track_lengths, event_lengths, slopes, nups, min_freqs = data_selection
#         data = {}
#         data['StartFrequency'] = np.array(start_freqs_vs_fss['0.0']) - 1.40812680e+09 + 50e6
        logger.info("Data extracted = {}".format(data.keys()))
        for key in data.keys():
            logger.info("{} -> size = {}".format(key,len(data[key])))
        
        complexLineShape = MultiGasComplexLineShape("complexLineShape")
        
        complexLineShape.data = data

        #fixed_para_values_array = [[1.0, 1.0, 0.817, 0.07, 0.08], [1.0, 1.0, 0.886, 0.02, 0.06], [1.0, 1.0, 0.748, 0.12, 0.1], [1.0, 1.0, 0.777, 0.138, 0.06], [1.0, 1.0, 0.857, 0.002, 0.1], [1.0, 1.0, 0.845, 0.086, 0.1]]# [1.0, 1.0, 0.817, 0.07, 0.08], [1.0, 1.0, 0.886, 0.02, 0.06], [1.0, 1.0, 0.748, 0.12, 0.1], [1.0, 1.0, 0.777, 0.138, 0.06], [1.0, 1.0, 0.857, 0.002, 0.1]
        f_array = np.arange(0.4, 0.61, 0.01)
        # gas_variation_array = [[0.817, 0.07, 0.08], [0.886, 0.02, 0.06], [0.748, 0.12, 0.1], [0.777, 0.138, 0.06], [0.857, 0.002, 0.1], [0.845, 0.046, 0.08]]# [1.0, 1.0, 0.817, 0.07, 0.08], [1.0, 1.0, 0.886, 0.02, 0.06], [1.0, 1.0, 0.748, 0.12, 0.1], [1.0, 1.0, 0.777, 0.138, 0.06], [1.0, 1.0, 0.857, 0.002, 0.1]]
        # max_snr_array = ['13.000', '13.500', '14.000', '14.500', '15.000', '15.500', '16.000', '16.500']
        f = 0
        output_dict = {}
#        directories = os.listdir('/home/ys633/lineshape_fitting/mermithid_share/20211119_max_snr_sampling_traps_combined')
#        for directory in [directories[0]]:
        complexLineShape_config['path_to_ins_resolution_data_txt'] = '/home/ys633/lineshape_fitting/mermithid_share/averaged_resolutions/averaged_october_resolution.txt'
        #         if i == 10:
#            complexLineShape_config['path_to_ins_resolution_data_txt'] = '/host/October_FTC_resolution/all_res_cf14.300.txt'

#        complexLineShape_config['fixed_parameter_values'] = [1.0, 0.817, 0.07, 0.08]

        complexLineShape_config['factor'] = f

#        complexLineShape_config['scatter_fractions_for_gases'] = [0.817, 0.07, 0.08]

        complexLineShape.Configure(complexLineShape_config)       

        complexLineShape.Run()

        results = complexLineShape.results

        logger.info(results['output_string'])
        logger.info('\n'+str(results['correlation_matrix']))

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
        if complexLineShape_config['resolution_function'] == 'simulated_resolution_scaled_fit_scatter_peak_ratio' or complexLineShape_config['resolution_function'] == 'simulated_resolution_scaled_fit_scatter_peak_ratio2':
            plot_title = 'data file:{},\n gases: {},\n resolution function: {}({}),\n fixed parameters:\n {}'.format(os.path.basename(reader_config['filename']),complexLineShape_config['gases'], complexLineShape_config['resolution_function'], os.path.basename(complexLineShape_config['path_to_ins_resolution_data_txt']), complexLineShape_config['fixed_parameter_names'])
        if complexLineShape_config['resolution_function'] == 'gaussian_resolution_fit_scatter_peak_ratio':
            plot_title = 'data file:{},\n gases: {},\n resolution function: {},\n fixed parameters:\n {}'.format(os.path.basename(reader_config['filename']),complexLineShape_config['gases'], complexLineShape_config['resolution_function'], complexLineShape_config['fixed_parameter_names'])
        plt.title(plot_title)
        plt.tight_layout()
        #plt.savefig('/host/plots/fit_FTC_march_with_simulated_resolution_cf{}_sp_1.0_width_factor_1.0.png'.format(file_cf))
        plt.savefig('/home/ys633/lineshape_fitting/plots/fit_October_FTC_with_new_gas_fraction_min_He.png')# March_FTC
        output_dict['october max snr 14.300'] = results
        np.save('/home/ys633/lineshape_fitting/mermithid_share/october_max_snr_14.300_factor_0_new_gas_fraction_min_He.npy', output_dict)
#             time.sleep(600)
#             output_file = open('/host/october_res_upper_and_lower_bounds_results.txt', 'a')
#             output_file.write('{}\n\n {}\n\n\n'.format('lower bound', results['output_string']))
#             output_file.close()

        

if __name__ == '__main__':

    unittest.main()
