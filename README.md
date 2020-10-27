# SPHEREx Simulated Likelihood Analysis

This is a python package for the [SPHEREx](https://spherex.caltech.edu/) simulated likelihood anaysis.
It requires the MCMC sampler [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html) to run.

## Before you start

1. Clone the repository:

    `git clone https://github.com/chenheinrich/SphereLikes.git`
    
2. It is recommended you create a [virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) before installing the dependencies:

    `conda create -n <yourenvname> python=3.7 anaconda`

    `source activate <yourenvname>`

   or

    `pip3 install pipenv [--user]`

    `virtualenv venv`

    `source venv/bin/activate`

## Install Requirements

You may skip to step 3 if you already have Cobaya and its cosmological packages including camb and planck likelihoods.

1. Install Cobaya.

    `git clone https://github.com/CobayaSampler/cobaya.git`

    `pip3 install -e cobaya --upgrade`

   To test the installation: `python3 -c "import cobaya"`. If you have trouble, follow instructions here to install cobaya manually: https://cobaya.readthedocs.io/en/latest/installation.html#making-sure-that-cobaya-is-installed

2. Install cosmological packages in Cobaya. But before you proceed, make sure you have gfortran or ifort compiler installed (test with `<gfortran_or_ifort> --version`). Also, MPI installation is optional but highly recommended (follow instructions [here](https://cobaya.readthedocs.io/en/latest/installation.html)).

   Install cosmological packages in Cobaya, replacing `<path_to_packages>` with the path of your choice, e.g. `./cosmo`. This means you will have `cobaya`, `cosmo` and `SphereLikes` on the same level. 

    `cobaya-install cosmo -p <path_to_packages>`
    
    You may also decide to only install camb (the only one needed for now) and skip the rest:
    
    `cobaya-install camb -p <path_to_packages>`

   If Planck likelihood installation fails, follow instructions [here](cosmo/code/planck/code).

3. Install other requirements (add `--user` if you're on a cluster):

    `cd SphereLikes`

    `pip3 install -r requirements.txt [--user]`

## Pip install `spherelikes` package 

Install in editable mode for now:

`pip3 install -e . [--user]`

if you have venv activated and do not have administrative permission, give explicit path for pip in your environment, e.g.:

`venv/bin/pip3.7 install -e .`

Test with `python3 -c "import spherelikes"`

## Run a sample cobaya run (must be from the root of this directory):

You should be able to run:

   `python3 scripts/prep_chains.py ./inputs/chains_pars/ps_base_v27.yaml`

to generate data needed for running MCMC chains: covariance matrix, reference cosmology and simulated data.

Then you can run chains (in debug mode add `-d` and to force removing existing chains add `-f`):

`python3 scripts/run_chains.py ./inputs/chains_pars/ps_base_v27.yaml 1 -d -f -run_in_python`

## Alternative install (under development): Docker

From outside the container, you can run chains using:
`docker run --rm chenheinrich/spherex:0.0.1`

This requires you have the chains prepared, i.e. have a inverse covariance matrix on disk, etc. This step won't be needed once this becomes an installable likelihood, so the inverse covariance matrix will be automatically downloaded during installation. For now, run:

`docker run --rm chenheinrich/spherex:0.0.1 python3 scripts/prep_chains.py ./inputs/chains_pars/ps_base.yaml`

To run chains possibly with different commands:
`docker run --rm chenheinrich/spherex:0.0.1 python3 scripts/run_chains.py ./inputs/chains_pars/ps_base.yaml 1 -d -f -run_in_python`