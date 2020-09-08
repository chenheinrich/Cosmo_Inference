# SphereLikes

## Install Requirements

1. It is recommended you create a virtual environment:

`conda create -n <yourenvname> python=3.7 anaconda`

`source activate <yourenvname>`

2. Install requirements (add `--user` for a cluster): 

`pip install -r requirements.txt`

## Running spherelikes with Cobaya

1. Add `spherelikes` package to your python path by manually executing:

`GIT_ROOT=$(git rev-parse --show-toplevel)

LIKE_PATH=$GIT_ROOT/spherelikes/

export PYTHONPATH=$PYTHONPATH:$LIKE_PATH
`
or make use of the setup.sh file:

`bash setup.sh`
`source ~/.bashrc`

2. Run a sample cobaya run (must be from the root of this directory):

`python run.py`

Use `-f` to force delete existing sample chains when running a second time:

`python run.py -f` 
