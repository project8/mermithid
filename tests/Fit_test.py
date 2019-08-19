'''
This scripts aims at testing Tritium specific processors.
Author: M. Guigue
Date: Apr 1 2018
'''


from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

    def test_rooFit(self):
        from mermithid.processors.TritiumSpectrum import TritiumSpectrumProcessor
        #from morpho.processors.plots import Histogram
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint

        energy_resolution = 36

        specGen_config = {
            "mode": "generate",
            "paramRange": {
                "KE": [18000, 19000]
            },

            "volume": 7e-6*1e-2, # [m3]
            "density": 3e17, # [1/m3]
            "duration": 1.*seconds_per_year()/12., # [s]
            "neutrino_mass" :0, # [eV]
            "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
            # "energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
            "background": 1e-6, # [counts/eV/s]
            # *****
            # "neutrino_mass" :0, # [eV]
            # "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
            # "energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
            # "background": 1e-6, # [counts/eV/s]
            # "n_events": 100,
            # "n_bkgd": 1022,
            # *****
            "energy_resolution": energy_resolution, # [eV],
            "fixedParams": {
                "m_nu": 0
            },
            "varName": "KE",
            "iter": 10000,
            "interestParams": ['KE'],
        }
        specFit_config = {
            "mode": "fit",
            "paramRange": {
                "KE": [18000, 19000]
            },
            "volume": 7e-6*1e-2, # [m3]
            "density": 3e17, # [1/m3]
            "duration": 1.*seconds_per_year()/12., # [s]
            "neutrino_mass" :0, # [eV]
            "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
            "background": 1e-6, # [counts/eV/s]
            # *****
            # "neutrino_mass" :0, # [eV]
            # "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
            # "energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
            # "background": 1e-6, # [counts/eV/s]
            # "n_events": 100,
            # "n_bkgd": 1022,
            # *****
            "energy_resolution": energy_resolution, # [eV],
            "fixedParams": {
                "m_nu": 0,
                "widthSmearing": energy_resolution
            },
            "varName": "KE",
            "interestParams": ['endpoint'],
            "make_fit_plot": False,
            "binned": False
        }
        specGen = TritiumSpectrumProcessor("specGen")
        specFit = TritiumSpectrumProcessor("specFit")
        #histo = Histogram("histo")

        specGen.Configure(specGen_config)
        #histo.Configure(histo_plot)
        specFit.Configure(specFit_config)
        specGen.Run()
        result = specGen.data
        #histo.data = result
        specFit.data = result
        #histo.Run()
        specFit.Run()
        print(specFit.result)


if __name__ == '__main__':
    unittest.main()
