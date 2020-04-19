'''
This scripts aims at testing IO processors by reading files.
Author: C. Claessens
Date: April 08 2020
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

import matplotlib.pyplot as plt

class IOTests(unittest.TestCase):

    def test_MutliChannelReader(self):
        from mermithid.processors.IO import MultiChannelCicadaReader
        reader_config = {
            "action": "read",
            "channel_ids": ["a", "b", "c"],
            "filename": ["rid000069955_merged.root",
                         "rid000066843_merged.root",
                         "rid000069341_merged.root"],
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
        n, bins, p = plt.hist(data['F'], bins=100)
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