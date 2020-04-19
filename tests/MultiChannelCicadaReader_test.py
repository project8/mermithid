'''
This scripts aims at testing IO processors by reading files.
Author: C. Claessens
Date: April 08 2020
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np

class IOTests(unittest.TestCase):

    def test_MutliChannelReader(self):
        from mermithid.processors.IO import MultiChannelCicadaReader
        reader_config = {
            "action": "read",
            "channel_ids": ["a", "b", "c"],
            "filename": ["events_000007031_katydid_v2.13.0_concat.root",
                         "events_000007031_katydid_v2.13.0_concat.root",
                         "events_000007031_katydid_v2.13.0_concat.root"],
            "rf_roi_min_freqs": [25803125000.0, 25862500000.0, 25921875000.0],
            "channel_transition_freqs": [[0,1.38623121e9+24.5e9],
                                         [1.38623121e9+24.5e9, 1.44560621e9+24.5e9],
                                        [1.44560621e9+24.5e9, 50e9]],
            "object_type": "TMultiTrackEventData",
            "object_name": "multiTrackEvents:Event",
            "use_katydid": False,
            "variables": ['StartFrequency', 'StartTimeInRunC'],
            "merged_frequency_variable": "F"
        }
        b = MultiChannelCicadaReader("reader")
        b.Configure(reader_config)
        b.Run()
        data = b.data
        logger.info("Data extracted = {}".format(data.keys()))
        for key in data.keys():
            if key in reader_config["channel_ids"]:
                logger.info("Channel {} data:".format(key))
                logger.info(data[key].keys())
            else:
                logger.info("{} -> size = {}".format(key,len(data[key])))



        plt.figure(figsize=(7,7))
        plt.subplot(211)
        bins = np.linspace(min(data['F']), max(data['F']), 100)
        n, bins, p = plt.hist(data['F'], bins=bins)
        plt.xlabel('Start frequencies')
        plt.ylabel('N')

        plt.subplot(212)
        plt.hist(data['a']['TrueStartFrequenciesCut']-24.5e9, bins=bins-24.5e9, label='channel a: {} counts'.format(len(data['a']['TrueStartFrequenciesCut'])))
        plt.hist(data['b']['TrueStartFrequenciesCut']-24.5e9, bins=bins-24.5e9, label='channel b: {} counts'.format(len(data['b']['TrueStartFrequenciesCut'])))
        plt.hist(data['c']['TrueStartFrequenciesCut']-24.5e9, bins=bins-24.5e9, label='channel c: {} counts'.format(len(data['c']['TrueStartFrequenciesCut'])))
        plt.xlabel('Start frequencies')
        plt.ylabel('N')
        plt.legend()

        plt.tight_layout()
        plt.savefig('multi_channel_reader_test.png', dpi=200)

if __name__ == '__main__':
    unittest.main()