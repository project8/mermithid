'''
This scripts aims at testing IO processors by reading files.
Author: M. Guigue
Date: Feb 27 2018
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

import matplotlib.pyplot as plt

class IOTests(unittest.TestCase):

    def test_Cicada(self):
        from mermithid.processors.IO import MultiChannelCicadaReader
        reader_config = {
            "action": "read",
            "N_channels": 3,
            "filename": ["/host/input_data/tritium_run_3_channel_a_some_runs.root",
                         "/host/input_data/tritium_run_3_channel_b_some_runs.root",
                         "/host/input_data/tritium_run_3_channel_c_some_runs.root"],
            "rf_roi_min_freqs": [25803125000.0, 25862500000.0, 25921875000.0],
            "channel_transition_freqs": [[0,1.38623121e9+24.5e9],
                                         [1.38623121e9+24.5e9, 1.44560621e9+24.5e9],
                                        [1.44560621e9+24.5e9, 50e9]],
            "object_type": "TMultiTrackEventData",
            "object_name": "multiTrackEvents:Event",
            "use_katydid": False,
            "variables": ['StartFrequency', 'StartTimeInRunC']
        }
        b = MultiChannelCicadaReader("reader")
        b.Configure(reader_config)
        b.Run()
        data = b.data
        logger.info("Data extracted = {}".format(data.keys()))
        for key in data.keys():
            logger.info("{} -> size = {}".format(key,len(data[key])))

        print(data.keys())

        plt.figure(figsize=(7,7))
        plt.subplot(211)
        n, bins, p = plt.hist(data['TrueStartFrequenciesMerged'], bins=100)
        plt.xlabel('Start frequencies')
        plt.ylabel('N')

        plt.subplot(212)
        plt.hist(data['a']['TrueStartFrequenciesCut'], bins=bins, label='channel a')
        plt.hist(data['b']['TrueStartFrequenciesCut'], bins=bins, label='channel b')
        plt.hist(data['c']['TrueStartFrequenciesCut'], bins=bins, label='channel c')
        plt.xlabel('Start frequencies')
        plt.ylabel('N')
        plt.legend()

        plt.tight_layout()
        plt.savefig('multi_channel_reader_test.png', dpi=200)

if __name__ == '__main__':
    unittest.main()