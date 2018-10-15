------------------
Installation
------------------

These are two possible ways of installing and working with mermithid.

Virtual environment installation
----------------------------------

One can install mermithid on one's machine using a virtual environment: this allows to keep the system relatively clean.
However, mermithid uses C++ librairies (Cicada and Phylloxera) that need to be built beforehand.

Cicada and Phylloxera need to be installed in a sub directory: ::

	mkdir build
	cd build
	cmake ..
	make -j3
	make -j3 install

These libraries need to be added to your PYTHONPATH: ::

	echo "export PYTHONPATH=${PWD}/build:$PYTHONPATH" >> ~/.bash_profile

Inside your virtual environement, install mermithid: ::

	source ~/path/to/the/virtual/environment/bin/activate # activate the virtual environement
	echo $PYTHONPATH # make sure the build folder above is in this path
	pip install . --process-dependency-links

(The `--process-dependency-links` is here to install the right morpho version from github.)

Docker installation
--------------------

Docker provides a uniform test bed for development and bug testing.
Please use this environment to testing/resolving bugs.

1. Install Docker (Desktop version): https://docs.docker.com/engine/installation/
2. Clone and pull the latest master version of morpho
3. Inside the morpho folder, execute ```docker-compose run mermithid```. The container prompter should appear at the end of the installation. A directory (```mermithid_share```) should be created in your home and mounted under the ```/host``` folder: you can modify this by editing the docker-compose file. Once inside the container, run `source /setup.sh` to be able to access morpho and mermithid libraries.
4. When reinstalling, you can remove the image using ```docker rmi mermithid_mermithid```
