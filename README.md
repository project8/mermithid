# mermithid

[![DOI](https://zenodo.org/badge/122257399.svg)](https://zenodo.org/badge/latestdoi/122257399) [![Documentation Status](https://readthedocs.org/projects/mermithid/badge/?version=latest)](https://mermithid.readthedocs.io/en/latest/?badge=latest)

Mermithid is an extension of [morpho](https://github.com/morphoorg/morpho) that contains processors specific to Project 8 analysis, spectrum fitting and plotting.

## Requirements

If you are not using a container with pre-installed dependencies, you will need to install via a package manager (such as apt-get):

- python (3.x; 2.7.x support not guaranteed)
- python-pip
- git
- ROOT (Cern) with pyROOT

## Installation

These are two possible ways of installing and working with mermithid.

### Virtual environment installation

Before installing, clone subdirecories recursively: ``git submodule update --init --recursive``.
<br>Then install mermithid in your environment:

1. Cicada and Phylloxera need to be installed in a sub directory:

  ```bash
  mkdir build
  cd build
  cmake ..
  make -j3
  make -j3 install
  ```

2. These libraries need to be added to your PYTHONPATH:

  ```bash
  echo "export PYTHONPATH=${PWD}/build:$PYTHONPATH" >> ~/.bash_profile
  ```

3. Install mermithid:

  ```bash
  pip install . -e ./morpho
  ```

   (The `-e ./morpho` is here to install morpho as an egg.)

### Docker installation

Docker provides a uniform test bed for development and bug testing. Please use this environment to testing/resolving bugs.

1. Install Docker (Desktop version): <https://docs.docker.com/engine/installation/>
3. Clone and pull the latest main version of mermithid or the feature branch you want to work with
4. Go to the cloned directory: ``cd mermithid``
5. Pull the submodules: ``git submodule update --init --recursive``
6. Build docker image: ``docker build --no-cache -t mermithid:<tag> .``
7. To start the container and mount a directory for data sharing with your host into the container do: 
<br>```docker run --rm -it -v ~/mermithid_share:/host mermithid:<tag> /bin/bash```

An alternative to steps 6 and 7 is to use docker-compose by executing: ``docker-compose run mermithid``.

<br>In both cases files saved in ```/host``` are shared with the host in ```~/mermithid_share```.


### Running mermithid

For running mermithid you need to set the paths right for using these software. For example in the docker container:

```bash
source $MERMITHID_BUILD_PREFIX/setup.sh
source $MERMITHID_BUILD_PREFIX/bin/this_cicada.sh
source $MERMITHID_BUILD_PREFIX/bin/this_phylloxera.sh
```

## Quick start and examples

Mermithid works a-la morpho, where the operations on data are defined using processors. Each processor should be defined with a name, then should have its attributes configured using a dictionary before being run. Examples of how to use mermithid can be found in the "tests" and the "test_analysis" folders.


## Easy development

To develop mermithid without having to rebuild the container, share the repository on the host with the container by starting it with: ```docker run --rm -it -v ~/mermithid_share:/host -v ~/repos/mermithid:/mermithid mermithid:<tag> /bin/bash```.

Then, after sourcing the setup scripts modify the PYTHONPATH ```export PYTHONPATH=/mermithid:$PYTHONPATH```. Now changes made on the host will directly be used by the container.
