# docker run -it -m inf -v $PWD:/code ubuntu:18.04 /bin/bash 
# docker run -it -m inf -v $PWD:/code spagnuolocarmine/docker-mpi:latest /bin/bash 

FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    gfortran \
    openmpi-bin \
    openmpi-common \
    liblapack-dev \
    libopenblas-dev \
    libopenmpi-dev 
    
COPY ./ /code/

RUN git clone https://github.com/CobayaSampler/cobaya.git \
&&  pip3 install -e cobaya --upgrade \
&&  cobaya-install camb -p ./cosmo 

RUN pip3 install -r code/requirements.txt \
&&  pip3 install -e code \
&&  python3 -c "import spherelikes" 

WORKDIR '/code'
RUN pytest ./tests/ -v -m short 

CMD ["python3", "scripts/run_chains.py", "./inputs/chains_pars/ps_base.yaml", "1", "-f", "-d", "-run_in_python"]
