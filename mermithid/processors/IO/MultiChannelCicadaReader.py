'''
Processor that can read (not write) Katydid ROOT files using the Cicada library
Author: C. Claessens
Date: April 08 2020
'''

from morpho.processors.IO import IOProcessor
from morpho.utilities import reader, morphologging
logger = morphologging.getLogger(__name__)

try:
    from ROOT import TFile, TTreeReader, TTreeReaderValue
except ImportError:
    pass

import numpy as np

class MultiChannelCicadaReader(IOProcessor):
    '''
    Processor that can read (not write) Katydid ROOT files using the Cicada library
    '''

    def InternalConfigure(self,params):
        '''
        Args:
            object_type: class of the object to read in the file
            object_name: name of the tree followed by the name of the object
            use_katydid: retro-compatibility to Katydid namespace
        '''
        super().InternalConfigure(params)
        self.object_type = reader.read_param(params,"object_type","TMultiTrackEventData")
        self.object_name = reader.read_param(params,"object_name","multiTrackEvents:Event")
        self.use_katydid = reader.read_param(params,"use_katydid",False)

        self.N_channels = reader.read_param(params, "N_channels", 3)
        self.channel_ids = reader.read_param(params, "channel_ids", ['a', 'b', 'c'])
        self.rf_roi_min_freqs = reader.read_param(params, "rf_roi_min_freqs", [0, 0, 0])
        self.transition_freqs = reader.read_param(params, "channel_transition_freqs", [0, 0])
        self.frequency_variable_name = reader.read_param(params, "merged_frequency_variable", "F")
        return True

    def Reader(self):
        '''
        Reader method
        '''
        print('\nReading data from {} channels'.format(self.N_channels))
        print('Min frequencies: {}'.format(self.rf_roi_min_freqs))
        print('Transition freqs: {}'.format(self.transition_freqs))

        self.data = {}
        for i in range(self.N_channels):
            logger.debug("Reading {}".format(self.file_name[i]))
            try:
                from ReadKTOutputFile import ReadKTOutputFile
            except ImportError:
                logger.warn("Cannot import ReadKTOutputFile")
            self.data[self.channel_ids[i]] = ReadKTOutputFile(self.file_name[i],self.variables,katydid=self.use_katydid,objectType=self.object_type,name=self.object_name)



        if 'StartFrequency' in self.variables:

            if len(self.variables) == 1:
                for k in self.data.keys():
                    self.data[k] = {'StartFrequency': self.data[k]}

            all_frequencies = []

            for i in range(self.N_channels):

                true_frequencies = np.array(self.data[self.channel_ids[i]]['StartFrequency'])+self.rf_roi_min_freqs[i]

                index = np.where((true_frequencies>=self.transition_freqs[i][0]) &(true_frequencies<self.transition_freqs[i][1]))
                all_frequencies.extend(list(true_frequencies[index]))

                self.data[self.channel_ids[i]]['TrueStartFrequenciesCut'] = true_frequencies[index]

            self.data[self.frequency_variable_name] = all_frequencies
        return True

    def Writer(self):
        '''
        End-user analysis should not produce Katydid output objects...
        '''
        logger.error("End user analysis: cannot write reconstruction algorithm output")
        raise
