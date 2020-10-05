# SPHEREx Simulated Likelihood Analysis

This is a python package for the [SPHEREx](https://spherex.caltech.edu/) simulated likelihood anaysis.
It requires the MCMC sampler [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html) to run.

## Install Requirements

1. It is recommended you create a [virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/):

   `conda create -n <yourenvname> python=3.7 anaconda`

   `source activate <yourenvname>`

   or

   `pip3 install pipenv [--user]`
   `cd SphereLikes`
   `virtualenv venv`
   `source venv/bin/activate`

2. Install requirements within this environment (add `--user` on a cluster):

   `pip3 install -r requirements.txt [--user]`

3. Test that cobaya is properly installed:

   `python3 -c "import cobaya"`

   If you have trouble, follow instructions here to install cobaya manually: https://cobaya.readthedocs.io/en/latest/installation.html#making-sure-that-cobaya-is-installed

4. Install cosmology packages that comes with cobaya including camb and the planck likelihoods:

   cobaya-install cosmo -p ./cosmo

(For more information on cosmology packages for Cobaya, see
https://cobaya.readthedocs.io/en/latest/installation_cosmo.html)

## Pip install `spherelikes` package in editable mode (This will be included automatically in requirements.txt once published.)

    `pip3 install -e . [--user]`

if you have venv activated and do not have administrative permission, use for example:

    `venv/bin/pip3.7 install -e .`

Test with `python -c "import spherelikes"`

## Run a sample cobaya run (must be from the root of this directory):

You should be able to run

    `python scripts/run_cobaya.py`

Use `-f` to force delete existing sample chains when running a second time; use `-d` to run in debug mode:

    `python scripts/run_cobaya.py -f -d`
