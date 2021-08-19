#Under construction!!

#!/bin/bash

echo "Pipeline started"

# Controls about what to execute goes here
python -m lss_theory.scripts.get_ps ./src/lss_theory/sample_inputs/get_ps.yaml

echo "Pipeline ended."