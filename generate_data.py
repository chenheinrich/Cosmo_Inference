from spherelikes.model import ModelCalculator


def main():
    args = {
        'model_name': 'sim_data',
        'model_yaml_file': './inputs/sample_fid_model.yaml',
        'cobaya_yaml_file': './inputs/sample.yaml',
        'output_dir': './data/sim_data/',
    }

    calc = ModelCalculator(args)
    results = calc.get_and_save_results()
    calc.load_results()


if __name__ == '__main__':
    main()
