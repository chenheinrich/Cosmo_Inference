# Put the spherelikes packages on your python path
GIT_ROOT=$(git rev-parse --show-toplevel)
LIKE_PATH=$GIT_ROOT/spherelikes/
printf "\n\n# SPHEREx_forecasts likelihoods \nexport PYTHONPATH=$PYTHONPATH:$LIKE_PATH" >> ~/.bashrc

