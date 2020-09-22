from spherelikes.model import FidModelCalculator


def main():
    args = {
        'model_name': 'covariance_debug',
        'model_yaml_file': './inputs/sample_fid_model.yaml',
        'cobaya_yaml_file': './inputs/sample.yaml',
        'output_dir': './data/covariance/',
    }

    fid_calculator = FidModelCalculator(args)
    fid_calculator.get_and_save_results()
    fid_calculator.check_load_results()


if __name__ == '__main__':
    main()
