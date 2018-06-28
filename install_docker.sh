#!/bin/bash

source /setup.sh

export BUILDLOCATION=/build

pip install pip pkgconfig --upgrade
pip install colorlog # need a first installation before being updated by mermithid

echo "Installing Dependencies"
mkdir -p /mermithid/my_build
cd /mermithid/my_build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=${BUILDLOCATION} -DCMAKE_SKIP_INSTALL_RPATH:BOOL=True ..
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=${BUILDLOCATION} -DCMAKE_SKIP_INSTALL_RPATH:BOOL=True ..
make -j3
make install

echo "Installing Mermithid"
cd /mermithid
pip install . --upgrade  --process-dependency-links
