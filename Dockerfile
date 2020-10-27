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

# From outside the container, you can run chains:
# docker run --rm chenheinrich/spherex:0.0.1
# This requires you have the chains prepared, i.e. inverse covariance matrix on disk.
# This step won't be needed once this becomes an installable likelihood, 
# so you can download the inverse covariance matrix from anywhere.

# For now, you can just separately run the following:
# 1) prepare chains (make inverse covariance matrix, etc.)
# docker run --rm chenheinrich/spherex:0.0.1 python3 scripts/prep_chains.py ./inputs/chains_pars/ps_base.yaml
# 2) run chains with default yaml files (or custom made if needed):
# docker run --rm chenheinrich/spherex:0.0.1
# or with a different yaml file
# docker run --rm chenheinrich/spherex:0.0.1 python3 scripts/run_chains.py ./inputs/chains_pars/ps_base.yaml

# Next steps: use containers w/ MPI
# mpirun -np 4 cobaya-run ./inputs/chains_pars/ps_base.yaml
    
