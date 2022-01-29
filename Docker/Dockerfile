# HOW TO BUILD CONTAINER IMAGE : docker build -t isischameleon/usyd-pytorch:jidt2 .
# HOW TO RUN CONTAINER FROM THAT IMAGE:
# docker run -d -p 8896:8888 --shm-size=1G -e JUPYTER_ENABLE_LAB=yes --name tensyj2 -v data:/home/jovyan/data -v notebooks:/home/jovyan/notebooks isischameleon/usyd-pytorch:jidt2


#FROM jupyter/datascience-notebook:latest
FROM jupyter/scipy-notebook:latest

USER root

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    default-jdk \
    ant

USER $NB_UID

RUN mkdir "/home/$NB_USER/data" && \
    mkdir "/home/$NB_USER/notebooks" && \
    fix-permissions "/home/$NB_USER"

# Install Python 3 packages
## node-js: Installing nodejs for frontend debugging
## xeus-python: Installing a kernel supporting Jupyter debugging protocol
RUN conda install -c conda-forge --quiet --yes \
    'python-graphviz' \
    'jupyterlab-git' \
    'nodejs' \
    'jpype1' \
    'xeus-python' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN jupyter labextension install @jupyterlab/debugger @jupyterlab/git

RUN conda install -c pytorch --quiet --yes \
    'pytorch' \
    'torchvision' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN conda install -c conda-forge --quiet --yes \
    'pytorch-lightning' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

WORKDIR $HOME/notebooks