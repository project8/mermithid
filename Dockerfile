FROM project8/p8compute_dependencies:v0.4.1 as mermithid_common

ARG build_type=Release
ENV MERMITHOD_BUILD_TYPE=$build_type

ENV MERMITHID_TAG=v1.1.7
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
    echo 'export PYTHONPATH=$MERMITHID_BUILD_PREFIX/$(python -m site --user-site | sed "s%$(python -m site --user-base)%%"):$PYTHONPATH' >> setup.sh &&\
    /bin/true

########################
FROM mermithid_common as mermithid_done

COPY Cicada /tmp_source/Cicada
COPY mermithid /tmp_source/mermithid
COPY Phylloxera /tmp_source/Phylloxera
COPY CMakeLists.txt /tmp_source/CMakeLists.txt
COPY setup.py /tmp_source/setup.py
COPY .git /tmp_source/.git

COPY tests $MERMITHID_BUILD_PREFIX/tests

# repeat the cmake command to get the change of install prefix to set correctly (a package_builder known issue)
RUN source $MERMITHID_BUILD_PREFIX/setup.sh &&\
    cd /tmp_source &&\
    mkdir build &&\
    cd build &&\
    cmake -D CMAKE_BUILD_TYPE=$MERMITHID_BUILD_TYPE \
        -D CMAKE_INSTALL_PREFIX:PATH=$MERMITHID_BUILD_PREFIX \
        -D CMAKE_SKIP_INSTALL_RPATH:BOOL=True .. &&\
    cmake -D CMAKE_BUILD_TYPE=$MERMITHID_BUILD_TYPE \
        -D CMAKE_INSTALL_PREFIX:PATH=$MERMITHID_BUILD_PREFIX \
        -D CMAKE_SKIP_INSTALL_RPATH:BOOL=True .. &&\
    make -j3 install &&\
    cd /tmp_source &&\
    pip3 install . --process-dependency-links --prefix $MERMITHID_BUILD_PREFIX &&\
    /bin/true

########################
FROM mermithid_common

COPY --from=mermithid_done $MERMITHID_BUILD_PREFIX $MERMITHID_BUILD_PREFIX
