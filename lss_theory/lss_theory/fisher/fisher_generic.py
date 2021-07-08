import os
import numpy as np
import json
import copy
import scipy.linalg as linalg

from lss_theory.fisher.derivative_generic import AllDerivativesConvergence
from lss_theory.utils.file_tools import mkdir_p
from lss_theory.math_utils import matrix

class Fisher():

    def __init__(self, info, inverse_atol=1e-6):
        self._info = copy.deepcopy(info)
        self._inverse_atol = inverse_atol
        self._dir = self._info['fisher']['data_dir']
        mkdir_p(self._dir)
        self._setup_paths()

        self._params_list = self._info['AllDerivatives']['params']
        self._setup_module_and_class_names()

        self._invcov = self._load_invcov()
        
        self._derivatives = self._get_derivatives()
        self._setup_dims()

        self._fisher = self._get_fisher()
        self._setup_metadata()
        self._save()

        self._errors = self._get_errors()
        self._print_fisher_results()

    @property
    def data(self):
        return self._fisher 

    @property
    def metadata(self):
        return self._metadata

    @property
    def errors(self):
        return self._errors

    def _print_fisher_results(self):
        print('Fisher results:')
        print('    params = {}'.format(self._params_list))
        print('    errors = {}'.format(self._errors))

    def _get_errors(self): 
        print('fisher = {}'.format(self._fisher))
        inv_fisher = linalg.inv(self._fisher)
        print('inv_fisher = {}'.format(inv_fisher))

        is_symmetric = matrix.check_matrix_symmetric(inv_fisher)
        print('Passed test inverse fisher is symmetric? - {}'.format(is_symmetric))

        is_inverse_passed = matrix.check_matrix_inverse(self._fisher, inv_fisher, atol=self._inverse_atol, feedback_level=0)
        print('Passed inverse test (atol={})? - {}'.format(self._inverse_atol, is_inverse_passed))
        
        errors = np.sqrt(np.diag(inv_fisher))
        return errors

    def _setup_module_and_class_names(self):
        raise NotImplementedError

    def _load_invcov(self):
        raise NotImplementedError

    def _get_derivatives(self):
        module_name = self._module_name
        class_name = self._class_name
        parent_dir = self._derivative_dir #TODO can put in info as well in derivative class
        info = copy.deepcopy(self._info)
        self._der_conv = AllDerivativesConvergence(info, module_name, class_name, \
            ignore_cache=False, do_save=True,\
            parent_dir = parent_dir)
        return self._der_conv.data 
        
    def _get_fisher(self):
        fisher = np.zeros((self._nparam, self._nparam))

        for iparam in range(self._nparam):
            for jparam in range(self._nparam):
                print('iparam = {}, jparam = {}'.format(iparam, jparam))
                fisher[iparam, jparam] = self._get_fisher_matrix_element(iparam, jparam)

        print('fisher', fisher) 
        
        is_symmetric = matrix.check_matrix_symmetric(fisher)
        print('Passed test fisher is symmetric? - {}'.format(is_symmetric))

        return fisher
        
    def _get_fisher_matrix_element(self, iparam, jparam):
        raise NotImplementedError

    def _setup_dims(self):
        raise NotImplementedError

    def _setup_paths(self):
        self._fisher_path = os.path.join(self._dir, 'fisher.npy') 
        self._metadata_path = os.path.join(self._dir, 'metadata.json') 
        # TODO more sophiscated structure like in derivatives?

    def _setup_metadata(self):
        self._metadata = copy.deepcopy(self._info)
        der_conv_metadata = self._der_conv.metadata
        self._metadata['AllDerivatives']['h_frac'] = der_conv_metadata['AllDerivatives']['h_frac']

    def _save(self):
        self._save_data()
        self._save_metadata()

    def _save_data(self):
        np.save(self._fisher_path, self._fisher)
        print('Fisher data saved at {}'.format(self._fisher_path))

    def _save_metadata(self):
        with open(self._metadata_path, 'w') as json_file:
            json.dump(self._metadata, json_file, sort_keys=True, indent=4)
        print('Fisher metadata saved at {}'.format(self._metadata_path))
