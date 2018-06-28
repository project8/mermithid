'''
This scripts aims at testing IO processors by reading files.
Author: M. Guigue
Date: Feb 27 2018
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

class IOTests(unittest.TestCase):
    
    def test_Cicada(self):
        from mermithid.processors.IO import IOCicadaProcessor
        reader_config = {
            "action": "read",
            "filename": "events_rid000001097_000000000_to_000000014.root",
            "treename": "multiTrackEvents",
            "objectname": "Event",
            "variables": ['StartTimeInAcq','StartFrequency']
        }
        b = IOCicadaProcessor("reader")
        b.Configure(reader_config)
        b.Run()
        data = b.results
        logger.info("Data extracted = {}".format(data.keys()))
        for key in data.keys():
            logger.info("{} -> size = {}".format(key,len(data[key])))
            self.assertEqual(len(data[key]),22)

if __name__ == '__main__':
    unittest.main()