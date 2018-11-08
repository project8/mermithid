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
            "filename": "events_000007031_katydid_v2.13.0_concat.root",
            "object_type": "TMultiTrackEventData",
            "object_name": "multiTrackEvents:Event",
            "use_katydid": False,
            "variables": ['StartTimeInAcq','StartFrequency']
        }
        b = IOCicadaProcessor("reader")
        b.Configure(reader_config)
        b.Run()
        data = b.data
        logger.info("Data extracted = {}".format(data.keys()))
        for key in data.keys():
            logger.info("{} -> size = {}".format(key,len(data[key])))
            self.assertEqual(len(data[key]),320)

if __name__ == '__main__':
    unittest.main()