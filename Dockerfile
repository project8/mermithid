FROM project8/p8compute_dependencies:v0.2.0 as mermithid_common

ARG build_type=Release
ENV MERMITHOD_BUILD_TYPE=$build_type

ENV MERMITHID_TAG=feature/docker
ENV MERMITHID_BUILD_PREFIX=/usr/local/p8/mermithid/$MERMITHID_TAG

RUN mkdir -p $MERMITHID_BUILD_PREFIX &&\
    cd $MERMITHID_BUILD_PREFIX &&\
    echo "source ${COMMON_BUILD_PREFIX}/setup.sh" > setup.sh &&\
    echo "export MERMITHID_TAG=${MERMITHID_TAG}" >> setup.sh &&\
    echo "export MERMITHID_BUILD_PREFIX=${MERMITHID_BUILD_PREFIX}" >> setup.sh &&\
    echo 'ln -sf $MERMITHID_BUILD_PREFIX $MERMITHID_BUILD_PREFIX/../current' >> setup.sh &&\
    echo 'export PATH=$MERMITHID_BUILD_PREFIX/bin:$PATH' >> setup.sh &&\
    echo 'export LD_LIBRARY_PATH=$MERMITHID_BUILD_PREFIX/lib:$LD_LIBRARY_PATH' >> setup.sh &&\
    /bin/true

########################
FROM mermithid_common as mermithid_done

# repeat the cmake command to get the change of install prefix to set correctly (a package_builder known issue)
RUN source $MERMITHID_BUILD_PREFIX/setup.sh &&\
    mkdir /tmp_install &&\
    cd /tmp_install &&\
    git clone https://github.com/project8/mermithid &&\
    cd mermithid &&\
    git fetch && git fetch --tags &&\
    git checkout $MERMITHID_TAG &&\
    git submodule update --init --recursive &&\
    mkdir build &&\
    cd build &&\
    cmake -D CMAKE_BUILD_TYPE=$MERMITHID_BUILD_TYPE \
    -D CMAKE_INSTALL_PREFIX:PATH=$MERMITHID_BUILD_PREFIX \
    -D CMAKE_SKIP_INSTALL_RPATH:BOOL=True .. &&\
    cmake -D CMAKE_BUILD_TYPE=$MERMITHID_BUILD_TYPE \
    -D CMAKE_INSTALL_PREFIX:PATH=$MERMITHID_BUILD_PREFIX \
    -D CMAKE_SKIP_INSTALL_RPATH:BOOL=True .. &&\
    make -j3 install &&\
    /bin/true

########################
FROM mermithid_common

COPY --from=mermithid_done $MERMITHID_BUILD_PREFIX $MERMITHID_BUILD_PREFIX

# RUN mkdir /tmp_install &&\
#     cd /tmp_install &&\
#     git clone https://github.com/project8/mermithid &&\
#     cd mermithid &&\
#     git fetch && git fetch --tags &&\
#     git checkout $MERMITHID_TAG &&\
#     git submodule update --init --recursive &&\
#     source $MERMITHID_BUILD_PREFIX/setup.sh &&\
#     pip3 install . --process-dependency-links &&\
#     cd / &&\
#     rm /tmp_install &&\
#     morpho --help

RUN source $MERMITHID_BUILD_PREFIX/setup.sh &&\
    pip3 install git+git://github.com/project8/mermithid.git --process-dependency-links
