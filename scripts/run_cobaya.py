from cobaya.run import run
from cobaya.yaml import yaml_load_file
import sys

if __name__ == '__main__':

    info = yaml_load_file("./inputs/cobaya_pars/ps_base_minimal.yaml")

    for k, v in {"-f": "force", "-r": "resume", "-d": "debug"}.items():
        if k in sys.argv:
            info[v] = True

    updated_info, sampler = run(info)
