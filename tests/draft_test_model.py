import pytest
import numpy as np

from spherelikes.model import ModelCalculator


@pytest.fixture
def model_calc():
    """Returns a ModelCalculator instance with sample arguments above."""
    args = {
        'model_name': 'test_model',
        'model_yaml_file': './inputs/sample_fid_model.yaml',
        'cobaya_yaml_file': './inputs/sample.yaml',
        'output_dir': './data/test_model/',
    }
    model_calc = ModelCalculator(args)
    return model_calc


def test_model_calc_get_save_and_load_results():
    args = {
        'model_name': 'test_model',
        'model_yaml_file': './inputs/sample_fid_model.yaml',
        'cobaya_yaml_file': './inputs/sample.yaml',
        'output_dir': './data/test_model/',
    }
    model_calc = ModelCalculator(args)
    model_calc.get_and_save_results()
    results = model_calc.load_results()
    assert check_equal_dictionary(results, model_calc.results)


def check_equal_dictionary(d1, d2):  # TODO put into utils.
    for key in d1.keys():
        is_dict_1 = isinstance(d1[key], dict)
        is_dict_2 = isinstance(d2[key], dict)
        if (is_dict_1 and is_dict_2):
            is_equal = check_equal_dictionary(d1[key], d2[key])
            if is_equal is False:
                return False
        elif (not is_dict_1) and (not is_dict_2):
            if np.all(d1[key] != d2[key]):
                return False
        else:
            return False
    return True
