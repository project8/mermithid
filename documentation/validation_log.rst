Validation Log
==============

Log
---

Version: v1.2.3
~~~~~~~~~~~~~~~~

Release Date: Tues July 20 2021
''''''''''''''''''''''''''''''

Fixes:
'''''''''''''

* Updated mermithid and morpho submodule to handle PyROOT updates
* Changed "python" to "python3" for Dockerfile and tests


Version: v1.2.2
~~~~~~~~~~~~~~~~

Release Date: Fri June 8 2020
''''''''''''''''''''''''''''''

Fixes:
'''''''''''''

* Morpho install is done as a release install instead of with the `-e` flag
* Installed files are copied to a fresh copy of the container


Version: v1.2.1
~~~~~~~~~~~~~~~~

Release Date: Fri June 5 2020
''''''''''''''''''''''''''''''

Fixes:
'''''''''''''

* Removed the hard-coded path of the efficiency file in the FakeDataGenerator.py
* Made it optional to read the livetime from the root files in MultiChannelCicadaReader.py
* Added features that prevent the frequency window for analysis from going beyond the maximum FSS frequency in TritiumAndEfficiencyBinner.py
* Fixed function callings in KrComplexLineShape.py


Version: v1.2.0
~~~~~~~~~~~~~~~~

Release Date: Thur June 4 2020
''''''''''''''''''''''''''''''

New features:
'''''''''''''

* This release provides a complete framework to perform a preliminary analysis of the Tritium beta-decay energy spectrum with the aim of extracting the Tritium endpoint energy.
* Implemented the Mermithid processors required in the analysis chain
* Added test scripts for the processors
* Implemented Morpho as a git submodule
* Updated Morpho version to v2.7.0


Version: v1.1.12
~~~~~~~~~~~~~~~~

Release Date: Fri Jan 17 2020
'''''''''''''''''''''''''''''

Fixes:
''''''

* Fixing starting command in docker-compose

Version: v1.1.11
~~~~~~~~~~~~~~~~

Release Date: Mon Aug 26 2019
'''''''''''''''''''''''''''''

New features:
'''''''''''''

* Upgrade to morpho v2.5.0
* Rework of the RealTritiumSpectrum RooFit model
* P8 Compute Dependencies image update to v0.9.0

Version: v1.1.10
~~~~~~~~~~~~~~~~

Release Date: Sat Aug 3 2019
'''''''''''''''''''''''''''''

New features:
'''''''''''''

* Documentation update
* ReadTheDocs repair

Version: v1.1.9
~~~~~~~~~~~~~~~

Release Date: Mon Jul 22 2019
'''''''''''''''''''''''''''''

New features:
'''''''''''''

* P8 Compute Dependencies image update to v0.7.0

Version: v1.1.8
~~~~~~~~~~~~~~~

Release Date: Thur Apr 18 2019
'''''''''''''''''''''''''''''

New features:
'''''''''''''

* Morpho update to v2.3.2

Version: v1.1.7
~~~~~~~~~~~~~~~

Release Date: Thur Apr 4 2019
'''''''''''''''''''''''''''''

New features:
'''''''''''''

* Morpho update to v2.3.1
* Cicada update to v1.3.3

Version: v1.1.6
~~~~~~~~~~~~~~~

Release Date: Mon Feb 11 2019
'''''''''''''''''''''''''''''

New features:
'''''''''''''

* Update docker-compose.yaml

Version: v1.1.5
~~~~~~~~~~~~~~~

Release Date: Wed Dec 21 2018
'''''''''''''''''''''''''''''

New features:
'''''''''''''

* Update Dockerfile

Version: v1.1.4
~~~~~~~~~~~~~~~

Release Date: Wed Dec 6 2018
''''''''''''''''''''''''''''

New features:
'''''''''''''

* Update Dockerfile

Version: v1.1.3
~~~~~~~~~~~~~~~

Release Date: Wed Dec 5 2018
''''''''''''''''''''''''''''

New features:
'''''''''''''

* Update to Phylloxera v1.2.4

Version: v1.1.2
~~~~~~~~~~~~~~~

Release Date: Wed Dec 5 2018
''''''''''''''''''''''''''''

New features:
'''''''''''''

* Update to Phylloxera v1.2.3

Version: v1.1.1
~~~~~~~~~~~~~~~

Release Date: Wed Dec 5 2018
''''''''''''''''''''''''''''

Fixes:
'''''''''''''

* Changing base processor for TritiumLikelihoodSampler
* Upgrade of docker image build

Version: v1.1.0
~~~~~~~~~~~~~~~

Release Date: Mon Nov 19 2018
'''''''''''''''''''''''''''''

New Features:
'''''''''''''

* Documentation update (RTD and source code)
* morpho update to v2.3.0
* Kurie plot generator and fitter have been merged


Fixes:
'''''''''''''

* Various comments from users
