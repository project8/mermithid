cmake_minimum_required(VERSION 3.1)

cmake_policy( SET CMP0048 NEW ) # version in project()
project( Mermithid VERSION 0.0.4 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/Phylloxera/Scarab/cmake )
# list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/Cicada/Scarab/cmake )
include( PackageBuilder )

pbuilder_prepare_project()

set( Phylloxera_ENABLE_EXECUTABLES FALSE)

pbuilder_add_submodule( Cicada Cicada )
pbuilder_add_submodule( Phylloxera Phylloxera )
