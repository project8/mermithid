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
import random

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

class ComplexLineShapeTests(unittest.TestCase):

    def test_complex_lineshape(self):
        from mermithid.processors.IO import IOCicadaProcessor
        from mermithid.processors.misc.MultiGasComplexLineShape import MultiGasComplexLineShape


        reader_config = {
            "action": "read",
            "filename": "/home/ys633/lineshape_fitting/mermithid_share/march_2020_kr_calibration_channel_b_merged.root",
            "object_type": "TMultiTrackEventData",
            "object_name": "multiTrackEvents:Event",
            "use_katydid": False,
            "variables": ['StartTimeInAcq','StartFrequency']
        }

        complexLineShape_config = {
            'bins_choice': np.linspace(0e6, 100e6, 1000),
            'gases': ["H2", "He", "Ar", "Kr"], # "Ar", "Kr" # "Kr" for fss
            'fix_gas_composition': True,
            'fix_width_scale_factor': True,
            'factor': 0.4934,
            'scatter_fractions_for_gases': [0.894],
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
            'fixed_parameter_values': [1.0, 1.0, (0.411+0.992)/2, 0.579/2, (0.003+0.007)/2],   #[1.0, 1.0, 0.886, 0.02, 0.06]   
            # This is an important parameter which determines how finely resolved
            # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
            'num_points_in_std_array': 4000,
            'RF_ROI_MIN': 25859375000.0, #24.5e9 + 1.40812680e+09 - 50e6, #25850000000.0
            # shake_spectrum_parameters.json and oscillator strength data can be found at https://github.com/project8/scripts/tree/master/yuhao/line_shape_fitting/data
            'shake_spectrum_parameters_json_path': '/home/ys633/lineshape_fitting/mermithid/mermithid/misc/shake_spectrum_parameters.json',
            'path_to_osc_strengths_files': '/home/ys633/lineshape_fitting/mermithid_share/',
            'path_to_scatter_spectra_file': '/home/ys633/lineshape_fitting/mermithid_share/',
            'path_to_ins_resolution_data_txt': '/home/ys633/lineshape_fitting/mermithid_share/March_FTC_resolution/all_res_cf15.200.txt',
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
        
        

        #gas_variation_array = [0.804, 0.844, 0.884, 0.924, 0.964, 0.984] #, [1.0, 1.0, 0.984]
#         max_snr_array = [ '16.000', '16.100', '16.200', '16.300', '16.400', '16.500', '16.600', '16.700', '16.800', '16.900',
#                         '17.000', '17.100', '17.200', '17.300', '17.400', '17.500', '17.600', '17.700', '17.800', '17.900',
#                         '18.000']
        output_dict = {}
        f = 0.4955
        N = int(5e6)
        dif = 0.01
        H2_min = 0.411
        H2_max = 0.996
        He_min = 0
        He_max = 0.579
        Ar_min = 0.003
        Ar_max = 0.007
        Kr_min = 0.004
        Kr_max = 0.009
        a = np.random.uniform(H2_min, H2_max, N)
        b = np.random.uniform(He_min, He_max, N)
        c = np.random.uniform(Ar_min, Ar_max, N)
        d = np.random.uniform(Kr_min, Kr_max, N)
        a_list = []
        b_list = []
        c_list = []
        d_list = []
        for i in range(N):
            if a[i] + b[i] + c[i] + d[i] > 1 - dif and a[i] + b[i] + c[i] + d[i] < 1 + dif:
                a_list.append(a[i]/(a[i] + b[i] + c[i] + d[i]))
                b_list.append(b[i]/(a[i] + b[i] + c[i] + d[i]))
                c_list.append(c[i]/(a[i] + b[i] + c[i] + d[i]))
                d_list.append(d[i]/(a[i] + b[i] + c[i] + d[i]))
        for i, H2_fraction, He_fraction, Ar_fraction, Kr_fraction  in zip(range(301), a_list[0:301], b_list[0:301], c_list[0:301], d_list[0:301]):

            complexLineShape_config['path_to_ins_resolution_data_txt'] = '/home/ys633/lineshape_fitting/mermithid_share/March_FTC_resolution/all_res_cf15.200.txt'

            logger.info('{} {} {} {} {}'.format(H2_fraction, He_fraction, Ar_fraction, Kr_fraction, H2_fraction+ He_fraction+ Ar_fraction + Kr_fraction))

            complexLineShape_config['fixed_parameter_values'] = [1.0, 1.0, H2_fraction, He_fraction, Ar_fraction]

            complexLineShape_config['factor'] = f

            complexLineShape.Configure(complexLineShape_config)       

            complexLineShape.data = data

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
            if complexLineShape_config['resolution_function'] == 'simulated_resolution_scaled_fit_scatter_peak_ratio2':
                plot_title = 'data file:{},\n gases: {},\n resolution function: {}({}),\n fixed parameters:\n {}'.format(os.path.basename(reader_config['filename']),complexLineShape_config['gases'], complexLineShape_config['resolution_function'], os.path.basename(complexLineShape_config['path_to_ins_resolution_data_txt']), complexLineShape_config['fixed_parameter_names'])
            if complexLineShape_config['resolution_function'] == 'gaussian_resolution_fit_scatter_peak_ratio':
                plot_title = 'data file:{},\n gases: {},\n resolution function: {},\n fixed parameters:\n {}'.format(os.path.basename(reader_config['filename']),complexLineShape_config['gases'], complexLineShape_config['resolution_function'], complexLineShape_config['fixed_parameter_names'])
            plt.title(plot_title)
            plt.tight_layout()
            #plt.savefig('/host/plots/fit_FTC_march_with_simulated_resolution_cf{}_sp_1.0_width_factor_1.0.png'.format(file_cf))
            plt.savefig('/home/ys633/lineshape_fitting/plots/fit_March_FTC_with_max_snr_15.300_factor_{}_gas_fraction_variation_{}.png'.format(f, i))# March_FTC
            output_dict['max snr 15.200 factor {} gas fraction variation {} H2 fraction {} He fraction {} Ar fraction {}'.format(f, i, H2_fraction, He_fraction, Ar_fraction)] = results
        np.save('/home/ys633/lineshape_fitting/mermithid_share/march_max_snr_15.200_factor_0.4955_gas_composition_variation.npy'.format(f), output_dict)

#             output_dict = np.load('/host/march_res_stat_upper_lower_bounds.npy', allow_pickle = True)
#             output_dict = output_dict.item()
#             output_dict['lower bound'] = results
#         np.save('/host/march_res_stat_upper_lower_bounds.npy', output_dict)        

if __name__ == '__main__':

    unittest.main()