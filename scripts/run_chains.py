import sys
import argparse

from cobaya.run import run
from cobaya.yaml import yaml_load_file
from spherelikes.params import CobayaPar
from spherelikes.params_generator import CobayaParGenerator

if __name__ == '__main__':

    """
    Example usage:
        python3 scripts/run_chains.py ./inputs/chains_pars/ps_base.yaml -d -f
    """

    args = yaml_load_file(sys.argv[1])

    # Add default and custom settings to it
    cobaya_par_gen = CobayaParGenerator(
        args['cobaya_par_file'], 
        args['cobaya_par_file_gen_specs']
        )

    info = cobaya_par_gen.get_updated_info()

    for k, v in {"-f": "force", "-r": "resume", "-d": "debug"}.items():
        if k in sys.argv:
            info[v] = True

    updated_info, sampler = run(info)
