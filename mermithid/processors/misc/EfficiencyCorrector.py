'''
Plot a corrected histogram by dividing the histogram by a defined efficiency
function.
Author: M. Guigue, A. Ziegler
Date:8/1/2019
'''

from __future__ import absolute_import

from ROOT import TF1
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from morpho.processors.plots import RootCanvas, RootHistogram

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class EfficiencyCorrector(BaseProcessor):

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        # Initialize Canvas
        # try:
        #     self.rootcanvas = RootCanvas.RootCanvas(params, optStat=0)
        # except:
        self.rootcanvas = RootCanvas(params, optStat=0)


        # Read other parameters
        self.namedata = reader.read_param(params, 'variables', "required")
        self.bin_centers = reader.read_param(params, 'bin_centers', 'bin_centers') # Assumes binned data is an input dict with a list that has the bin centers.
        self.counts = reader.read_param(params, 'counts', 'counts') # Assumes binned data also contains a list that identifies the occupancy of each bin.
        self.eff_eqn = reader.read_param(params, 'efficiency', '1')
        self.mode = reader.read_param(params, 'mode', 'unbinned')
        self.n_bins_x = reader.read_param(params, 'n_bins_x', 100)
        self.histo = RootHistogram(params, optStat=0)
        self.range = reader.read_param(params, 'range', [0., -1.])
        self.eff_func = TF1("eff_func",self.eff_eqn,self.range[0],self.range[1])
        self.asInteger = reader.read_param(params, 'asInteger', False)
        self.total_counts = 0
        self.total_weighted_counts = 0
        self.eff_norm = 0
        self.corrected_data = {}
        self.corrected_data.update({self.namedata: [], 'efficiency': []})
        return True

    def InternalRun(self):
        self.rootcanvas.cd()
        if self.mode == 'unbinned':
            self.corrected_data[self.namedata] = self.data.get(self.namedata)
            for i in range(len(self.data.get(self.namedata))):
                self.eff_norm += self.eff_func.Eval(self.data.get(self.namedata)[i])
            for i in range(len(self.data.get(self.namedata))):
                self.corrected_data['efficiency'].append(self.eff_func.Eval(self.data.get(self.namedata)[i])/self.eff_norm)

        if self.mode == 'binned':
            self.corrected_data[self.namedata] = self.data.get(self.namedata)
            self.corrected_data.update({self.bin_centers: self.data.get(self.bin_centers), self.counts: self.data.get(self.counts)})
            for i in range(self.n_bins_x):
                # Assuming there is a list in the input dictionary that labels each data point with the bin it belongs to.
                self.total_counts += self.data.get(self.counts)[i]
                self.total_weighted_counts += self.data.get(self.counts)[i]/self.eff_func.Eval(self.data.get(self.bin_centers)[i])

            self.eff_norm = self.total_counts/self.total_weighted_counts
            for i in range(self.n_bins_x):
                self.corrected_data['efficiency'].append(self.eff_func.Eval(self.data.get(self.bin_centers)[i])/self.eff_norm)
                self.corrected_data['counts'][i] /= self.corrected_data['efficiency'][i]
                if self.asInteger:
                    temp_int = int(self.efficiency_data['counts'][i])
                    self.corrected_data['counts'][i] = temp_int

            self.histo.Fill(self.corrected_data.get(self.namedata))
            self.histo.SetBinsContent(self.corrected_data['counts'])
            #self.histo.Divide(self.eff_func)
            self.histo.Draw("hist")
        self.rootcanvas.Save()
        return True
