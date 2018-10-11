'''
Processor that can read (not write) Katydid ROOT files using the Cicada library
Author: M. Guigue
Date: Mar 30 2018
'''

from morpho.processors.IO import IOProcessor
from morpho.utilities import reader, morphologging
logger = morphologging.getLogger(__name__)

try:
    from ROOT import TFile, TTreeReader, TTreeReaderValue
except ImportError:
    pass

class IOCicadaProcessor(IOProcessor):
    '''
    Processor that can read (not write) Katydid ROOT files using the Cicada library
    '''

    def InternalConfigure(self,params):
        super().InternalConfigure(params)
        self.object_type = reader.read_param(params,"object_type","TMultiTrackEventData")
        self.object_name = reader.read_param(params,"object_name","multiTrackEvents:Event")
        self.use_katydid = reader.read_param(params,"use_katydid",False)
        return True

    def Reader(self):
        '''
        '''
        logger.debug("Reading {}".format(self.file_name))
        try:
            from ReadKTOutputFile import ReadKTOutputFile
        except ImportError:
            logger.warn("Cannot import ReadKTOutputFile")
        self.data = ReadKTOutputFile(self.file_name,self.variables,katydid=self.use_katydid,objectType=self.object_type,name=self.object_name)
        return True
        

    def Writer(self):
        '''
        End-user analysis should not produce Katydid output objects...
        '''
        logger.error("End user analysis: cannot write reconstruction algorithm output")
        raise
