cmake_minimum_required(VERSION 3.1)

cmake_policy( SET CMP0048 NEW ) # version in project()
project( Mermithid VERSION 1.1.9 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/Phylloxera/Scarab/cmake )
# list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/Cicada/Scarab/cmake )
include( PackageBuilder )

pbuilder_prepare_project()

pbuilder_add_submodule( Cicada Cicada )
pbuilder_add_submodule( Phylloxera Phylloxera )

set( Scarab_BUILD_CODEC_JSON OFF CACHE BOOL "No coded" FORCE)
set( Scarab_BUILD_CODEC_YAML OFF CACHE BOOL "No codec" FORCE)
set( Scarab_BUILD_AUTHENTICATION OFF CACHE BOOL "No auth" FORCE)
set( Scarab_BUILD_PARAM OFF CACHE BOOL "No param" FORCE)

set( Cicada_ENABLE_KATYDID_NAMESPACE OFF CACHE BOOL "Use Cicada namespace" FORCE)

