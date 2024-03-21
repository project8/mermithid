ARG IMG_USER=project8
ARG IMG_REPO=p8compute_dependencies
ARG IMG_TAG=v1.0.0

FROM ${IMG_USER}/${IMG_REPO}:${IMG_TAG} as mermithid_common

ARG build_type=Release
ENV MERMITHID_BUILD_TYPE=$build_type

ENV MERMITHID_TAG=v1.2.4
ENV MERMITHID_BUILD_PREFIX=/usr/local/p8/mermithid/$MERMITHID_TAG

RUN mkdir -p $MERMITHID_BUILD_PREFIX &&\
    chmod -R 777 $MERMITHID_BUILD_PREFIX/.. &&\
    cd $MERMITHID_BUILD_PREFIX &&\
    echo "source ${COMMON_BUILD_PREFIX}/setup.sh" > setup.sh &&\
    echo "export MERMITHID_TAG=${MERMITHID_TAG}" >> setup.sh &&\
    echo "export MERMITHID_BUILD_PREFIX=${MERMITHID_BUILD_PREFIX}" >> setup.sh &&\
    echo 'ln -sfT $MERMITHID_BUILD_PREFIX $MERMITHID_BUILD_PREFIX/../current' >> setup.sh &&\
    echo 'export PATH=$MERMITHID_BUILD_PREFIX/bin:$PATH' >> setup.sh &&\
    echo 'export LD_LIBRARY_PATH=$MERMITHID_BUILD_PREFIX/lib:$LD_LIBRARY_PATH' >> setup.sh &&\
    echo 'export PYTHONPATH=$MERMITHID_BUILD_PREFIX/$(python3 -m site --user-site | sed "s%$(python3 -m site --user-base)%%"):$PYTHONPATH' >> setup.sh &&\
    /bin/true

RUN source $COMMON_BUILD_PREFIX/setup.sh &&\
    pip install iminuit &&\
    pip install numericalunits &&\
    /bin/true

########################
FROM mermithid_common as mermithid_done

# Commented out for now: should be changed to whatever is needed later
# COPY analysis /tmp_source/analysis
COPY Cicada /tmp_source/Cicada
COPY documentation /tmp_source/documentation
COPY morpho /tmp_source/morpho
COPY mermithid /tmp_source/mermithid
COPY Phylloxera /tmp_source/Phylloxera
COPY tests /tmp_source/tests
COPY CMakeLists.txt /tmp_source/CMakeLists.txt
COPY setup.py /tmp_source/setup.py
COPY .git /tmp_source/.git

COPY tests $MERMITHID_BUILD_PREFIX/tests

# repeat the cmake command to get the change of install prefix to set correctly (a package_builder known issue)
RUN source $MERMITHID_BUILD_PREFIX/setup.sh &&\
    cd /tmp_source &&\
    mkdir -p build &&\
    cd build &&\
    cmake -D CMAKE_BUILD_TYPE=$MERMITHID_BUILD_TYPE \
        -D CMAKE_INSTALL_PREFIX:PATH=$MERMITHID_BUILD_PREFIX \
        -D CMAKE_SKIP_INSTALL_RPATH:BOOL=True .. &&\
    cmake -D CMAKE_BUILD_TYPE=$MERMITHID_BUILD_TYPE \
        -D CMAKE_INSTALL_PREFIX:PATH=$MERMITHID_BUILD_PREFIX \
        -D CMAKE_SKIP_INSTALL_RPATH:BOOL=True .. &&\
    make -j3 install &&\
    cd /tmp_source &&\
#    ls -altrh morpho &&\
    pip3 install . ./morpho --prefix $MERMITHID_BUILD_PREFIX &&\
    /bin/true

########################
FROM mermithid_common

COPY --from=mermithid_done $MERMITHID_BUILD_PREFIX $MERMITHID_BUILD_PREFIX
