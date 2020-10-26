import re
import os 
import sys
import yaml 
import argparse

def find_key(rules, line):
    for key in rules.keys():
        if line.startswith(key):  # TODO key must be unique, no errors raised yet
            return key

def find_values(key, values):
    print(key, values)
    print(re.findall(r"[-+]?\d*\.\d+|\d+", key))
    num_numbers_in_key_name = len(re.findall(r"[-+]?\d*\.\d+|\d+", key))
    return values[num_numbers_in_key_name:]

def get_dict_from_path_with_rules(path, rules):
    output_dict = {}
    with open(path) as fp: 
        for line in fp: 
            for key in rules.keys():
                if line.startswith(key):
                    num_start = len(key) + len(' = ')
                    values = line[num_start:]
                    if len(values)>0:
                        values = values.split()
                        values = [float(v) for v in values]
                    else:
                        values = float(values)
                    rename = rules[key]['rename'] or key
                    output_dict[rename] = values

    output_dict['zbin_lo'] = output_dict['zPk'][:-1]
    output_dict['zbin_hi'] = output_dict['zPk'][1:]
    del output_dict['zPk']

    return output_dict

def get_output_path(path, rule_type):
    filename, extension = os.path.splitext(path)
    if rule_type == 'roland':
        output_path = filename.replace('params.', 'survey_pars_') + '.yaml'
    else:
        raise NotImplementedError
    return output_path

def write_yaml(output_path, output_dict):
    with open(output_path, 'w') as f:
        yaml.dump(output_dict, f) 
    print('Written file: {}'.format(output_path))

def main(path, rule_type):
    import importlib
    import conversion_rules
    rules = getattr(conversion_rules, rule_type)
    #from conversion_rules import roland as rules
    output_dict = get_dict_from_path_with_rules(path, rules)
    print('output_dict', output_dict)

    output_path = get_output_path(path, rule_type)
    write_yaml(output_path, output_dict)
    

if __name__ == '__main__':
    """Sample Usage: 
        python convert_to_yaml ./params.v27_base_cbe_Mar21.txt roland
    """

    parser = argparse.ArgumentParser(description='Convert survey parameter file to yaml')
    parser.add_argument('path', help='path to survey parameter file')
    parser.add_argument('--type', default='spherex_public', help="name of conversion rules: \
        currently supporting 'roland' and 'spherex-public' (default). ")
    args = parser.parse_args()

    main(args.path, args.type)


