'''
Processor that can read (not write) multiple Katydid ROOT files
associated with multiple (roach) channels  using the Cicada library
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
            channel_ids: key list that will be used for storing the file content, length determines how many files are read
            rf_roi_min_freqs: list of frequencies that is added to all frequencies read from ROOT file, has to match length of channel ids
            channel_transition_ranges: list of frequency ranges for each channel
            merged_frequency_variable: reutrn key for merged start frequencies
        '''
        super().InternalConfigure(params)
        self.object_type = reader.read_param(params,"object_type","TMultiTrackEventData")
        self.object_name = reader.read_param(params,"object_name","multiTrackEvents:Event")
        self.use_katydid = reader.read_param(params,"use_katydid",False)

        self.channel_ids = reader.read_param(params, "channel_ids", ['a', 'b', 'c'])
        self.rf_roi_min_freqs = reader.read_param(params, "rf_roi_min_freqs", [0, 0, 0])
        self.transition_freqs = reader.read_param(params, "channel_transition_freqs", [0, 0])
        self.frequency_variable_name = reader.read_param(params, "merged_frequency_variable", "F")

        if len(self.channel_ids) > len(self.rf_roi_min_freqs):
            raise ValueError('More channel ids than min frequencies')
        if len(self.channel_ids) > len(self.transition_freqs):
            raise ValueError('More channel ids than frequency ranges')
        if len(self.channel_ids) > len(self.file_name):
            raise ValueError('More channel ids than root files')

        self.N_channels = len(self.channel_ids)
        return True

    def Reader(self):
        '''
        Reader method
        '''
        logger.info('\nReading data from {} channels'.format(self.N_channels))
        logger.info('Min frequencies: {}'.format(self.rf_roi_min_freqs))
        logger.info('Transition freqs: {}'.format(self.transition_freqs))

        self.data = {}
        for i in range(self.N_channels):
            logger.debug("Reading {}".format(self.file_name[i]))
            try:
                from ReadKTOutputFile import ReadKTOutputFile
            except ImportError:
                logger.warn("Cannot import ReadKTOutputFile")

            self.data[self.channel_ids[i]] = ReadKTOutputFile(self.file_name[i],self.variables,katydid=self.use_katydid,objectType=self.object_type,name=self.object_name)


            if len(self.variables) == 1:
                self.data[self.channel_ids[i]] = {self.variables[0]: self.data[self.channel_ids[i]]}

            self.data[self.channel_ids[i]]['TotalLifetime'] = self.get_total_live_time_from_root_rile(self.file_name[i])


        if 'StartFrequency' in self.variables:

            all_frequencies = []

            for i in range(self.N_channels):

                try:
                    true_frequencies = np.array(self.data[self.channel_ids[i]]['StartFrequency'])+self.rf_roi_min_freqs[i]
                except KeyError as e:
                    logger.warning('Key error: Possibly no events in channel {}?'.format(self.channel_ids[i]))
                    true_frequencies = np.array([])
                except TypeError as e:
                    logger.warning('TypeError: Possibly no events in channel {}?'.format(self.channel_ids[i]))
                    true_frequencies = np.array([])

                index = np.where((true_frequencies>=self.transition_freqs[i][0]) &(true_frequencies<self.transition_freqs[i][1]))
                all_frequencies.extend(list(true_frequencies[index]))

                self.data[self.channel_ids[i]]['TrueStartFrequenciesCut'] = true_frequencies[index]
                self.data[self.channel_ids[i]]['FrequencyRange'] = [self.transition_freqs[i][0], self.transition_freqs[i][1]]

            self.data[self.frequency_variable_name] = all_frequencies
        return True

    def Writer(self):
        '''
        End-user analysis should not produce Katydid output objects...
        '''
        logger.error("End user analysis: cannot write reconstruction algorithm output")
        raise

    def get_total_live_time_from_root_rile(self, path_to_root_file):
        f = TFile.Open(path_to_root_file, 'read')
        list_of_keys = f.GetListOfKeys()

        number_of_slices_of_livetime = 0
        for i in range(len(list_of_keys)):
            if 'livetime' in f.GetListOfKeys()[i].GetName():
                number_of_slices_of_livetime += 1

        total_livetime = 0
        for i in range(number_of_slices_of_livetime):
            name_of_livetime_slice = 'livetime;{}'.format(i+1)
            try:
                total_livetime += f.Get(name_of_livetime_slice)[0]
            except Exception as e:
                logger.error(e)
                logger.error('livetime not readable in slice {} of root file {}'.format(name_of_livetime_slice, path_to_root_file))
                continue
        logger.info('Total livetime in file: {}'.format(total_livetime))
        return total_livetime
