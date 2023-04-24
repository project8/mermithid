cmake_minimum_required(VERSION 3.1)

cmake_policy( SET CMP0048 NEW ) # version in project()
project( Mermithid VERSION 1.2.2 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/Phylloxera/Scarab/cmake )
# list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/Cicada/Scarab/cmake )
include( PackageBuilder )

pbuilder_prepare_project()

set_option( Scarab_BUILD_CODEC_JSON OFF )
set_option( Scarab_BUILD_CODEC_YAML OFF )
set_option( Scarab_BUILD_AUTHENTICATION OFF )
set_option( Scarab_BUILD_CLI OFF )
set_option( Scarab_BUILD_PARAM OFF )
set_option( Scarab_ENABLE_EXECUTABLES OFF )
set_option( Cicada_ENABLE_KATYDID_NAMESPACE OFF )

pbuilder_add_submodule( Cicada Cicada )
pbuilder_add_submodule( Phylloxera Phylloxera )
