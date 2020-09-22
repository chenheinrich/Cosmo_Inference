from cobaya.run import run
from cobaya.yaml import yaml_load_file
import sys

info = yaml_load_file("./inputs/sample.yaml")

for k, v in {"-f": "force", "-r": "resume", "-d": "debug"}.items():
    if k in sys.argv:
        info[v] = True

updated_info, sampler = run(info)
