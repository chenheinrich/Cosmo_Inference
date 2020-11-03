import sys
import argparse
import subprocess

from cobaya.run import run
from cobaya.yaml import yaml_load_file, yaml_dump_file
from spherelikes.params import CobayaPar
from spherelikes.params_generator import CobayaParGenerator

if __name__ == '__main__':

    """
    Example usage:
    1) in python
        python3 scripts/run_chains.py ./inputs/chains_pars/ps_base.yaml 1 -d -run_in_python
    2) using mpi with 16 chains on TACC
        python3 scripts/run_chains.py ./inputs/chains_pars/ps_base.yaml 16 -d
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("chain_par", type=str, default=None,
        help="path to config file.")
    parser.add_argument("n", default=4,
        help="number of chains to run if using mpi.")
    parser.add_argument("-d", action='store_true',
        help="debug mode.")
    parser.add_argument("-f", action='store_true',
        help="force delete previous chains.")
    parser.add_argument("-r", action='store_true',
        help="resume chains.")
    parser.add_argument("--run_in_python", nargs='?', const=True, default=False,
        help="running inside of python (no mpi) inside of on commandline.")

    command_line_args = parser.parse_args()
    nproc = command_line_args.n
    run_in_python = command_line_args.run_in_python
    print('run_in_python', run_in_python, 'nproc', nproc)

    chain_yaml_file = sys.argv[1]
    args = yaml_load_file(chain_yaml_file)

    # Add default and custom settings to it
    cobaya_par_gen = CobayaParGenerator(
        args['cobaya_par_file'], 
        args['cobaya_par_file_gen_specs']
        )

    info = cobaya_par_gen.get_updated_info()

    if run_in_python is True:
        for k, v in {"-f": "force", "-r": "resume", "-d": "debug"}.items():
            if k in sys.argv:
                info[v] = True
        updated_info, sampler = run(info)
    else:
        file_name = args['cobaya_par_file'].replace('.yaml', '.gen.yaml')
        yaml_dump_file(file_name, info, error_if_exists=False)
        flags = []
        for k in ['-d', '-f', '-r']:
            if k in sys.argv:
                flags.append(k)
        #TODO ibrun -n is TACC specific (mpirun -np would work elsewhere)
        cmd = ["ibrun", "-n", str(nproc), "cobaya-run", file_name] 
        cmd.extend(flags)
        print('Starting command: {}'.format(cmd))
        list_files = subprocess.run(cmd)

    print('run_chains.py: Done')
