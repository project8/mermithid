'''
Reads in data and fits it with complex lineshape model.
Author: E. Machado, Y.-H. Sun, E. Novitski
Date: 4/8/20
'''

import numpy as np
import unittest
import matplotlib.pyplot as plt
import ROOT as r

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

class ComplexLineShapeTests(unittest.TestCase):

    def test_complex_lineshape(self):
        from mermithid.processors.IO import IOCicadaProcessor
        from mermithid.processors.misc.MultiGasComplexLineShape import MultiGasComplexLineShape


        reader_config = {
            "action": "read",
            "filename": "/host/march_2020_kr_calibration_channel_b_merged.root",
            "object_type": "TMultiTrackEventData",
            "object_name": "multiTrackEvents:Event",
            "use_katydid": False,
            "variables": ['StartTimeInAcq','StartFrequency']
        }
        complexLineShape_config = {
            'bins_choice': np.linspace(0e6, 100e6, 1000),
            'gases': ["H2", "Kr"],
            'max_scatters': 20,
            'fixed_scatter_proportion': True,
            # configure the resolution functions: simulated_resolution, gaussian_resolution, gaussian_lorentzian_composite_resolution
            'resolution_function': 'gaussian_lorentzian_composite_resolution',
            # When fix_scatter_proportion is True, set the scatter proportion for gas1 below
            'gas_scatter_proportion': [0.61, 0.39],
            # This is an important parameter which determines how finely resolved
            # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
            'num_points_in_std_array': 10000,
            'RF_ROI_MIN': 25850000000.0,
            # shake_spectrum_parameters.json and oscillator strength data can be found at https://github.com/project8/scripts/tree/master/yuhao/line_shape_fitting/data
            'shake_spectrum_parameters_json_path': '../mermithid/misc/shake_spectrum_parameters.json',
            'path_to_osc_strengths_files': '/host/',
            'path_to_scatter_spectra_file': '/host/',
            'path_to_ins_resolution_data_txt': '/host/res_all_conversion_max25.txt'
        }

        b = IOCicadaProcessor("reader")
        complexLineShape = MultiGasComplexLineShape("complexLineShape")

        b.Configure(reader_config)
        complexLineShape.Configure(complexLineShape_config)

        b.Run()
        data = b.data
        logger.info("Data extracted = {}".format(data.keys()))
        for key in data.keys():
            logger.info("{} -> size = {}".format(key,len(data[key])))

        complexLineShape.data = data

        complexLineShape.Run()

        results = complexLineShape.results
        logger.info(results['output_string'])

        # plot fit with shake spectrum
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(15,9))
        plt.step(
        results['bins_Hz']/1e9, results['data_hist_freq'],
        label = 'data\n total counts = {}\n'.format(len(data['StartFrequency']))
        )
        plt.plot(results['bins_Hz']/1e9, results['fit_Hz'], label = results['output_string'], alpha = 0.7)
        plt.legend(loc = 'upper left', fontsize = 12)
        plt.xlabel('frequency GHz')
        plot_title = 'fit ftc march with {} gas scattering'.format(len(complexLineShape_config['gases']))
        if complexLineShape_config['fixed_scatter_proportion'] == True:
            str_gas_scatter_proportion = ''
            for i in range(len(complexLineShape_config['gases'])):
                str_gas_scatter_proportion += complexLineShape_config['gases'][i]
                str_gas_scatter_proportion += ': '
                str_gas_scatter_proportion += str(complexLineShape_config['gas_scatter_proportion'][i])
                str_gas_scatter_proportion += ' '
            plot_title += '\n with fixed scatter proportion \n {}'.format(str_gas_scatter_proportion)
        plt.title(plot_title)
        plt.tight_layout()
        plt.savefig('/host/plots/fit_FTC_march_with_composite_gaussian_lorentzian_resolution.png'.format(len(complexLineShape_config['gases'])))

if __name__ == '__main__':

    unittest.main()