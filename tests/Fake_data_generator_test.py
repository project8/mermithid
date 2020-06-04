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

        specGen_config = {
            "apply_efficiency": False,
            "efficiency_path": "./combined_energy_corrected_eff_at_quad_trap_frequencies.json",
            "simplified_lineshape_path": None,
            "detailed_or_simplified_lineshape": "detailed", #"simplified" or "detailed"
            "use_lineshape": False, # if False only gaussian smearing is applied
            "return_frequency": True,
            "scattering_sigma": 18.6, # only used if use_lineshape = True
            "scattering_prob": 0.77, # only used if use_lineshape = True
            "B_field": 0.9578186017836624,
            "S": 4500, # number of tritium events
            "n_steps": 1000, # stepsize for pseudo continuous data is: (Kmax_eff-Kmin_eff)/nsteps
            "A_b": 1e-10, # background rate 1/eV/s
            "poisson_stats": True
        }

        specGen = FakeDataGenerator("specGen")

        specGen.Configure(specGen_config)

        specGen.Run()
        results = specGen.results

        # plot histograms of generated data
        Kgen = results['K']
        Fgen = results['F']

        plt.figure(figsize=(7, 7))
        plt.subplot(211)
        n, b, _ = plt.hist(Kgen, bins=50, label='Fake data')
        plt.plot(specGen.Koptions, specGen.probs/(specGen.Koptions[1]-specGen.Koptions[0])*(b[1]-b[0])*len(Kgen), label='Model')
        plt.xlabel('K [eV]')
        plt.ylabel('N')
        plt.legend()

        plt.subplot(212)
        n, b, p = plt.hist(Fgen, bins=50, label='Fake data')
        plt.xlabel('F [Hz]')
        plt.ylabel('N')
        plt.legend()

        plt.tight_layout()
        plt.savefig('GeneratedData.png', dpi=100)


if __name__ == '__main__':
    unittest.main()
