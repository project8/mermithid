FROM guiguem/root-docker:python3

MAINTAINER Mathieu Guigue "Mathieu.Guigue@pnnl.gov"

COPY . /mermithid

RUN chmod +x /mermithid/install_docker.sh &&\
    sleep 1 &&\
    /mermithid/install_docker.sh

CMD ['source /setup.sh']
