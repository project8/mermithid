# mermithid

Mermithid is an extension of morpho that contains processors specific to Project 8 analysis, spectrum fitting and plotting.


## Requirements

You will need to install via a package manager (such as apt-get):
  * python (2.7.x; 3.x)
  * python-pip
  * git
  * ROOT (Cern)

## Installation

  These are two possible ways of installing and working with mermithid.

### Virtual environment installation

  1. Cicada and Phylloxera need to be installed in a sub directory:
  ```bash
    mkdir build
    cd build
    cmake ..
    make -j3
    make -j3 install
  ```
  1. These libraries need to be added to your PYTHONPATH:
  ```bash
    echo "export PYTHONPATH=${PWD}/build:$PYTHONPATH" >> ~/.bash_profile
  ```
  1. Install mermithid:
  ```bash
    pip install . --process-dependency-links
  ```
  (The `--process-dependency-links` is here to install morpho.)

### Docker installation

  Docker provides a uniform test bed for development and bug testing.
  Please use this environment to testing/resolving bugs.

  1. Install Docker (Desktop version): https://docs.docker.com/engine/installation/
  1. Clone and pull the latest master version of morpho
  1. Inside the morpho folder, execute ```docker-compose run mermithid```.
  The container prompter should appear at the end of the installation.
  A directory (```mermithid_share```) should be created in your home and mounted under the ```/host``` folder: you can modify this by editing the docker-compose file.
  1. When reinstalling, you can remove the image using ```docker rmi mermithid_mermithid```

## Quick start and examples

  Mermithid works a-la morpho, where the operations on data are defined using processors.
  Each processor should be defined with a name, then should have its attributes configured using a dictionary before being run.
  Examples of how to use mermithid can be found in the "tests" folder.

