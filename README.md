# SPHEREx Simulated Likelihood Analysis

This is a python package for the [SPHEREx](https://spherex.caltech.edu/) simulated likelihood anaysis.
It requires the MCMC sampler [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html) to run.

## Before you start

1. Clone the repository:

    `git clone https://github.com/chenheinrich/SphereLikes.git`
    `cd SphereLikes`
    
2. It is recommended you create a [virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) before installing the dependencies:

    `conda create -n sphere python=3.7`

    `conda activate <yourenvname>`

   or

    `pip3 install pipenv [--user]`

    `virtualenv venv`

    `source venv/bin/activate`

## Install Requirements

1. Install requirements (add `--user` if you're on a cluster):

    `python3 -m pip install -r requirements.txt [--user]`

2. Install packages (spherex_cobaya and lss_theory) in this repository (add `--user` if you're on a cluster; and `-e` for editable mode if you are actively developing these packages):

    `python3 -m pip install [-e] . [--user] `
    `python3 -m pip install [-e] ./lss_theory [--user]`

If you have `venv` activated and do not have administrative permission, give explicit path for pip in your environment, e.g.:

`venv/bin/pip3.7 install -e .`

3. Test that the packages are properly installed:

`python3 -c "import spherex_cobaya; import lss_theory"`
`python3 -c "import lss_theory"`

## Run tests

Current working tests are (we are still under construction):

1. `python3 tests/dev_test_theory.py`
2. `python3 -m pytest tests/test_theory.py -m short`

Use `python3 -m pytest` instead of `pytest` to ensure that you are using the 
same `pytest` you installed earlier with `requirements.txt`.

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


