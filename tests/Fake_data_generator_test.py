'''
This scripts aims at testing Tritium specific processors.
Author: C. Claessens
Date: Apr 6 2020
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

import matplotlib.pyplot as plt

class FakeDataGenerationTest(unittest.TestCase):

    def test_data_generation(self):
        from mermithid.processors.TritiumSpectrum.FakeDataGenerator import FakeDataGenerator
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint

        specGen_config = {
            "apply_efficiency": False,
            "efficiency_path": "/host/input_data/combined_energy_corrected_eff_at_quad_trap_frequencies.json",
            "simplified_lineshape_path": "/host/input_data/simplified_scattering_params.txt",
            "detailed_or_simplified_lineshape": "gaussian", #"simplified", "detailed"
            "use_lineshape": True, # if False only gaussian smearing is applied
            "return_frequency": True,
            "scattering_sigma": 18.6,
            "scattering_prob": 0.77,
            "B_field": 0.9578186017836624,
            "S": 4500, # number of tritium events
            "n_steps": 40000, # stepsize for pseudo continuous data is: (Kmax_eff-Kmin_eff)/nsteps
            "A_b": 1e-10 # background rate 1/eV/s
        }

        specGen = FakeDataGenerator("specGen")

        specGen.Configure(specGen_config)

        specGen.Run()
        results = specGen.results

        # plot histograms of generated data
        Kgen = results['K']
        Fgen = results['F']

        plt.figure(figsize=(7, 5))
        plt.subplot(121)
        plt.hist(Kgen, bins=50)
        plt.xlabel('K [eV]')
        plt.ylabel('N')

        plt.subplot(122)
        plt.hist(Fgen, bins=50)
        plt.xlabel('F [Hz]')
        plt.ylabel('N')

        plt.tight_layout()
        plt.savefig('GeneratedData.png', dpi=200)


if __name__ == '__main__':
    unittest.main()
