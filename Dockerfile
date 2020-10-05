FROM ubuntu:18.04
RUN apt-get update && apt-get upgrade -y 
RUN apt-get install -y \
    python3 \
    python3-pip 
RUN apt-get install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    git 
COPY ./ /code/
RUN git clone https://github.com/CobayaSampler/cobaya.git && \
    pip3 install -e cobaya --upgrade && \
    cobaya-install cosmo -p ./cosmo 

RUN pip3 install -r code/requirements.txt && \
    pip3 install -e code &&
    python3 -c "import spherelikes"

RUN cd code 
    #pytest scripts/
    #cobaya-run sample_planck.yaml -f
    #python3 scripts/run_cobaya.py -f
