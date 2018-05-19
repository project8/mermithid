'''
'''

import CicadaPy
CicadaPy.loadLibraries(True)
from ROOT import Katydid as KT

from morpho.processors.IO import IOProcessor
from morpho.utilities import reader, morphologging
logger = morphologging.getLogger(__name__)

from ROOT import TFile, TTreeReader, TTreeReaderValue


class IOCicadaProcessor(IOProcessor):

    def InternalConfigure(self,params):
        super().InternalConfigure(params)
        self.tree_name = reader.read_param(params,"tree_name","multiTrackEvents")
        self.object_name = reader.read_param(params,"object_name","Event")

    def Reader(self):
        '''
        '''
        logger.debug("Reading {}".format(self.file_name))
        file = TFile.Open(self.file_name)
        if not file:
            raise FileNotFoundError("File {} does not exist".format(self.file_name))

        # Extract tree from file
        tree = file.Get(self.tree_name)
        # Create TTreeReader
        treeReader = TTreeReader(tree)
        # Create object TMultiTrackEventData to "point" to the object "Event" in the tree
        multiTrackEvents = TTreeReaderValue(KT.TMultiTrackEventData)(treeReader, "Event")

        logger.debug("Extracting {} from {}".format(self.variables,self.object_name))
        theData = {}
        for var in self.variables:
            theData.update({str(var): []})
        # Go through the events
        while treeReader.Next():
            for var in self.variables:
                try:
                    function = getattr(multiTrackEvents,"Get{}".format(var))
                except:
                    logger.error("Variable {} does not exist in TMultiTrackEventData".format(var))
                    raise
                theData[var].append(function())
        return theData
        

    def Writer(self):
        '''
        End-user analysis should not produce Katydid output objects...
        '''
        logger.error("End user analysis: cannot write reconstruction algorithm output")
        raise
