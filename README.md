# SPHEREx Simulated Likelihood Analysis

This is a python package for the [SPHEREx](https://spherex.caltech.edu/) simulated likelihood anaysis.
It requires the MCMC sampler [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html) to run.

## Before you start

1. Clone the repository:

    `git clone https://github.com/chenheinrich/spherex_cobaya.git`
    
2. It is recommended you create a [virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) before installing the dependencies:

    `conda create -n <yourenvname> python=3.7`

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

   Install cosmological packages in Cobaya, replacing `<path_to_packages>` with the path of your choice, e.g. `./cosmo`. This means you will have `cobaya`, `cosmo` and `spherex_cobaya` on the same level. 

    `cobaya-install cosmo -p <path_to_packages>`
    
    You may also decide to only install camb (the only one needed for now) and skip the rest:
    
    `cobaya-install camb -p <path_to_packages>`

   If Planck likelihood installation fails, follow instructions [here](cosmo/code/planck/code).

3. Install other requirements (add `--user` if you're on a cluster):

    `cd spherex_cobaya`

    `python3 -m pip install -r requirements.txt [--user]`

## Pip install `spherex_cobaya` package 

Install in editable mode for now:

`python3 -m pip install -e . [--user]`

if you have venv activated and do not have administrative permission, give explicit path for pip in your environment, e.g.:

`venv/bin/pip3.7 install -e .`

Test with `python3 -c "import spherex_cobaya"`

`python3 -m pip install -e ./lss_theory [--user]`

Test with `python3 -c "import lss_theory"`

python3 tests/dev_test_theory.py

## Run a sample cobaya run (must be from the root of this directory):

You should be able to run:

   `python3 scripts/prep_chains.py ./inputs/chains_pars/ps_base_v27.yaml`

to generate data needed for running MCMC chains: covariance matrix, reference cosmology and simulated data.

Then you can run chains (in debug mode add `-d` and to force removing existing chains add `-f`):

`python3 scripts/run_chains.py ./inputs/chains_pars/ps_base_v27.yaml 1 -d -f -run_in_python`

## Alternatives: Docker (under development)

### Basic run

After installing Docker, you can pull the image

`docker pull chenheinrich/spherex:0.0.1`

and run the default chains for SPHEREx (simulated):
`docker run --rm chenheinrich/spherex:0.0.1`

### Custom chains

The following is subject to change.

To run a different chain than the basic one, use
`docker run --rm chenheinrich/spherex:0.0.1 python3 scripts/run_chains.py ./inputs/chains_pars/<run_name>.yaml 1 -d -f -run_in_python`
replacing <run_name> by the name of your new yaml file. Note that you will need to prepare the elements needed by the chains yourself in this case, by running
`docker run --rm chenheinrich/spherex:0.0.1 python3 scripts/prep_chains.py ./inputs/chains_pars/<run_name>.yaml`. 

This will:
1) run the reference cosmology and products needed for computing Alcock-Pazcynski effect (H(z) and D_A(z)).
2) create a covariance matrix with shot noise specified by survey parameters and invert it (which could take a few minutes).
3) create simulated data vector according to the cosmology specified.


