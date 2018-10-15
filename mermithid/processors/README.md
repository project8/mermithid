# processors

This folder contains processors specific to Project 8 separated into sub-folders according to the type of operations made.
Note that the structure follows closely the one adopted in [morpho](https://github.com/morphoorg/morpho/tree/master/morpho/processors).
* **IO**: contains IO processors for Project 8 specific files
    * **IOCicadaProcessor**: processor that can read (not write) Katydid output files (uses Cicada library as a comon backend)
* **TritiumSpectrum**: contains processors used for Tritium analyses purposes
    * **TritiumSpectrumGenerator**: generates fake Tritium spectra using RooFit (Phylloxera) with smearing
    * **TritiumSpectrumLikelihoodSampler**: analyze Tritium spectrum using the Likelihood sampling from RooFit (Metropolis-Hastings algorithm)
* **misc**: contains miscalleneous processors used for Project 8 specific-purposes
    * **FrequencyEnergyConversionProcessor**: converts a set of cyclotron frequencies into energies given a magnetic field
* **plots**: contains plotting Project 8 specific processors
    * **KuriePlotGeneratorProcessor**: generates Kurie plots from unbinned data
