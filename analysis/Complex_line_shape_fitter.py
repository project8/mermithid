'''
Reads in data and fits it with complex lineshape model.
Author: E. Machado, Y.-H. Sun, E. Novitski
Date: 4/8/20
'''

import numpy as np
import unittest
import matplotlib.pyplot as plt

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

class ComplexLineShapeTests(unittest.TestCase):

    def test_complex_lineshape(self):
        from mermithid.processors.IO import IOCicadaProcessor
        from mermithid.processors.misc.ComplexLineShape import ComplexLineShape
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint

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
            # This is an important parameter which determines how finely resolved
            # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
            'num_points_in_std_array': 10000,
            'RF_ROI_MIN': 25850000000.0,
            'B_field': 0.957810722501,
            'shake_spectrum_parameters_json_path': '/host/shake_spectrum_parameters.json',
            'path_to_osc_strengths_files': '/host/'
        }

        b = IOCicadaProcessor("reader")
        complexLineShape = ComplexLineShape("complexLineShape")

        b.Configure(reader_config)
        complexLineShape.Configure(complexLineShape_config)

        b.Run()
        data = b.data
        logger.info("Data extracted = {}".format(data.keys()))
        for key in data.keys():
            logger.info("{} -> size = {}".format(key,len(data[key])))

        #print(data['StartFrequency'])
        start_frequency_array = np.array(data['StartFrequency'])

        complexLineShape.data = data

        complexLineShape.Run()

        results = complexLineShape.results
        print(results['output_string'])
#        print(type(results['data_hist_freq']))
#        print(len(results['data_hist_freq']))

        # plot fit with shake spectrum
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(15,9))
        plt.step(
        results['bins_Hz'][0:-1]/1e9, results['data_hist_freq'],
        label = 'data\n total counts = {}\n'.format(len(data['StartFrequency']))
        )
        plt.plot(results['bins_Hz'][0:-1]/1e9, results['fit_Hz'], label = results['output_string'], alpha = 0.7)
        plt.legend(loc = 'upper left',fontsize = 12)
        plt.xlabel('frequency GHz')
        plt.title('fit with shake spectrum 2 gas scattering')
        plt.savefig('/host/plots/fit_shake_2_gas_0.png')

if __name__ == '__main__':
    unittest.main()
