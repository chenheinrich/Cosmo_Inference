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

1. Install Cobaya. You may skip to step 3 if you already have Cobaya and its cosmological packages including camb and planck likelihoods.

   `git clone https://github.com/CobayaSampler/cobaya.git`
   
   `pip3 install -e cobaya --upgrade`

   To test the installation: `python3 -c "import cobaya"`. If you have trouble, follow instructions here to install cobaya manually: https://cobaya.readthedocs.io/en/latest/installation.html#making-sure-that-cobaya-is-installed

2. Install cosmological packages in Cobaya. But before you proceed, make sure you have gfortran or ifort compiler installed (test with `<gfortran_or_ifort> --version`). Also, MPI installation is optional but highly recommended (follow instructions [here](https://cobaya.readthedocs.io/en/latest/installation.html)).

   Install cosmological packages in Cobaya, replacing <path_to_packages> with the path of your choice, e.g. `./cosmo`. This means you will have cobaya, cosmo and SphereLikes on the same level.

   `cobaya-install cosmo -p <path_to_packages>`

3. Install other requirements (add `--user` if you're on a cluster):

   `pip3 install -r SphereLikes/requirements.txt [--user]`
   
   
## Pip install `spherelikes` package in editable mode

    `pip3 install -e SphereLikes [--user]`

if you have venv activated and do not have administrative permission, give explicit path for pip in your environment, e.g.:

    `venv/bin/pip3.7 install -e SphereLikes`

Test with `python -c "import spherelikes"`

## Run a sample cobaya run (must be from the root of this directory):

You should be able to run

    `python3 scripts/run_cobaya.py -f`

You may add `-f` to force delete any existing sample chains you created with this command when running a second time and `-d` to run in debug mode.

You may need to run `generate_covariance.py` first to generate the covariance matrix. 
