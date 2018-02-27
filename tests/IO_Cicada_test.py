'''
This scripts aims at testing the IO Cicada processor by reading a file.
Author: M. Guigue
Date: Feb 27 2018
'''

from mermithid.processors.IO import IOCicadaProcessor
from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

reader_config = {
    "action": "read",
    "filename": "events_000001097_katydid_v2.7.0_concat.root",
    "variables": ['StartTimeInAcq','StartFrequency']
}

b = IOCicadaProcessor("reader")
b.Configure(reader_config)
data = b.Run()
logger.info("Data extracted = {}".format(data.keys()))
for key in data.keys():
    logger.info("{} -> size = {}".format(key,len(data[key])))