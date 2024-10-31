"""
Script to test the miscalleneous processors
Author: J. Johnston
Date: April 24, 2018

Converts frequencies corresponding to kinetic energy
"""

import unittest

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)


class MiscTest(unittest.TestCase):

    def test_FreqConversionTest(self):
        from mermithid.processors.misc import FrequencyEnergyConversionProcessor

        freq_data = [27.9925*10**9, 27.0094*10**9,
                     26.4195*10**9, 26.4169*10**9,
                     26.3460*10**9, 26.3457*10**9]
        logger.info("Will convert the following frequencies: %s"%freq_data)
        logger.debug("At 1 T, these correspond to kinetic energies (in keV) of " +
              "[0, 18.6, 30.424, 30.477, 31.934, 31.942]")

        freq_energy_dict = {
            "B": 1
        }

        freq_proc = FrequencyEnergyConversionProcessor("freq_energy_processor")
        freq_proc.Configure(freq_energy_dict)
        freq_proc.frequencies = freq_data
        freq_proc.Run()

        logger.info("Resulting energies: %s"%freq_proc.energies)



if __name__ == '__main__':

    args = parser.parse_args(False)


    logger = morphologging.getLogger('morpho',
                                     level=args.verbosity,
                                     stderr_lb=args.stderr_verbosity,
                                     propagate=False)
    logger = morphologging.getLogger(__name__,
                                     level=args.verbosity,
                                     stderr_lb=args.stderr_verbosity,
                                     propagate=False)

    unittest.main()
