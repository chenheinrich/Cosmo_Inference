# SPHEREx Cosmo_Inference

The [SPHEREx](https://spherex.caltech.edu/) Cosmo_Inference is the Inference section of the SPHEREx L4 Cosmology pipeline. It is used for performing simulated likelihood anaysis.

It comprises of the following python packages:
- lss_theory
- spherex_cobaya

We make use of the public code [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html) as a MCMC sampler. 

## Before you start

1. Clone the repository:

    `git clone https://github.com/chenheinrich/SphereLikes.git`
    `cd Cosmo_Inference`
    
2. It is recommended you create a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) before installing the dependencies (Note that Python 2 is decaprecated, we are using Python 3; if you are on a cluster, do `module avail python` to see what's available and use `module load python<X>` to select the version that corresponds to Python 3.7):

    `python -m pip install pipenv [--user]`

    `virtualenv venv`

    `source venv/bin/activate`

or 
    `conda create -n spherex python=3.7`

    `conda activate spherex`

We recommend using `virtualenv` instead of `conda`, since installing with pip inside a conda environment could cause problems sometimes.

## Install Requirements

2. From inside Cosmo_Inference, install packages (`spherex_cobaya` and `lss_theory`) in this repository (add `--user` if you're on a cluster; and `-e` for installing in editable mode if you are actively developing these packages):

    `python -m pip install [-e] ./src/lss_theory [--user]`
    `python -m pip install [-e] ./src/spherex_cobaya [--user] `

3. Test that the packages are properly installed:

    `python -c "import lss_theory; import spherex_cobaya"`

## Run tests

[PLEASE SKIP THIS SECTION, IT IS UNDER CONSTRUCTION]

Current working tests are:

For lss_theory:
1. `python -m pytest src/lss_theory/tests`

For spherex_cobaya
1. `python tests/dev_test_theory.py`
2. `python -m pytest tests/test_theory.py -m short`

Note: Use `python -m pytest` instead of `pytest` to ensure that you are using the 
same `pytest` you installed earlier with `requirements.txt` if there are various
python versions.

## Run scripts from individual packages (current directory must be Cosmo_Inference):

Run lss_theory sample scripts:

    `python -m lss_theory.scripts.get_ps ./src/lss_theory/sample_inputs/get_ps.yaml`
    `python -m lss_theory.scripts.get_b3d_rsd ./src/lss_theory/sample_inputs/get_b3d_rsd.yaml`

[Other scripts are still under construction (covariance, Fisher matrix, bispectrum multipole signal).]

Run spherex_cobaya sample scripts:

    `cobaya-run ./src/spherex_cobaya/sample_inputs/cobaya_pars/b3d_rsd_vary_all_but_w0_wa_mnu.yaml -d -f`

To get the data needed for running MCMC chains (covariance matrix and simulated data), use:

    [Coming soon]

[Other yaml files are still being upgraded.]

## Run sample pipeline (current directory must be Cosmo_Inference):

There is a draft pipeline under construction:

`./pipeline/scripts/execute.sh`

## Alternatives: Docker (under development)

[PLEASE SKIP THIS SECTION, THE FOLLOWING NEEDS TO BE UPDATED]

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


