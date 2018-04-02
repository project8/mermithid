'''                                                                                                                                     
Generate a Kurie plot from energy data
'''

from __future__ import absolute_import

import json
import os


from morpho.utilities import morphologging, reader, plots
from morpho.processors import BaseProcessor
from morpho.processors.plots import RootCanvas
logger=morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)

class KuriePlotGeneratorProcessor(BaseProcessor):
    '''                                                                                                                                
    Describe.
    '''

    def Configure(self, param_dict):
        '''
        Configure
        '''
        logger.info("Configure with {}".format(param_dict))
        # Initialize Canvas
        self.rootcanvas = RootCanvas.RootCanvas(param_dict,optStat=0)

        # Read other parameters
        self.nbins_x = int(reader.read_param(param_dict,'n_bins_x',100))
        self.namedata = reader.read_param(param_dict,'data',"required")
        # self.draw_opt_2d = reader.read_param(param_dict,'root_plot_option',"contz")

    def Run(self):
        logger.info("Run...")
        self.rootcanvas.Draw()

        import ROOT

        for value in self.data[self.namedata]:
            hist = ROOT.TH1F("%s%s" % (name, "_2sig"),name, self.n_bins_x, xmin, xmax)
