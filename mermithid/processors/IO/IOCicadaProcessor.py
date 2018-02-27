'''
'''

import CicadaPy
CicadaPy.loadLibraries()
from ROOT import Katydid as KT

from morpho.processors.IO import IOProcessor

from ROOT import TFile, TTreeReader, TTreeReaderValue


class IOCicadaProcessor(IOProcessor):

    def Configure(self,params):
        super().Configure(params)

    def Reader(self):
        '''
        '''
        file = TFile.Open(self.file_name)
        if not file:
            raise FileNotFoundError("File {} does not exist".format(self.file_name))

        # Extract tree from file
        tree = file.Get("multiTrackEvents")
        # Create TTreeReader
        treeReader = TTreeReader(tree)
        # Create object TMultiTrackEventData to "point" to the object "Event" in the tree
        multiTrackEvents = TTreeReaderValue(KT.TMultiTrackEventData)(treeReader, "Event")

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
