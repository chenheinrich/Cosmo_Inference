# SphereLikes

## Install Requirements

1. It is recommended you create a virtual environment:
`conda create -n <yourenvname> python=3.7 anaconda`
`source activate <yourenvname>`

2. Install requirements (add `--user` for a cluster): 
`pip install -r requirements.txt`

## Running spherelikes with Cobaya

1. Add `spherelikes` package to your python path
`bash setup.sh`
`source ~/.bashrc`
or manually execute:
`GIT_ROOT=$(git rev-parse --show-toplevel)
LIKE_PATH=$GIT_ROOT/spherelikes/
export PYTHONPATH=$PYTHONPATH:$LIKE_PATH
`

2. Run a sample cobaya run
`python run.py`
Use `-f` to force delete existing sample chains.
`python run.py -f` 