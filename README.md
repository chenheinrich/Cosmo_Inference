# Spherex Simulated Likelihood Analysis

This is a python package for the [SPHEREx](https://spherex.caltech.edu/) simulated likelihood anaysis. 
It requires the MCMC sampler [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html) to run.

## Install Requirements

1. It is recommended you create a [virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/):

    `conda create -n <yourenvname> python=3.7 anaconda`

    `source activate <yourenvname>`

2. Install requirements (add `--user` on a cluster): 

    `pip install -r requirements.txt`

## Running spherelikes with Cobaya

1. Add `spherelikes` package to your python path by manually executing:

    `GIT_ROOT=$(git rev-parse --show-toplevel)

    LIKE_PATH=$GIT_ROOT/spherelikes/

    export PYTHONPATH=$PYTHONPATH:$LIKE_PATH`

    or make use of the setup.sh file:

    `bash setup.sh`
    `source ~/.bashrc`

2. Run a sample cobaya run (must be from the root of this directory):

    `python scripts/run_cobaya.py`

Use `-f` to force delete existing sample chains when running a second time and `-d` to run in debug mode:

    `python run.py -f -d` 


