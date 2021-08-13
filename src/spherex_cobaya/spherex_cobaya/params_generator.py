import numpy as np 
import copy 
import importlib
from cobaya.yaml import yaml_load_file

from spherex_cobaya.params import CobayaPar

class CobayaParGenerator():

    def __init__(self, cobaya_par_file, gen_specs):
        self._cobaya_par_file = cobaya_par_file
        self._gen_specs = gen_specs 
        self._info = yaml_load_file(self._cobaya_par_file)

    def get_updated_info(self):
        params = self._info['params']
        new_params = self._get_new_params(params)
        self._info['params'].update(new_params)
        return self._info
    
    def _get_new_params(self, params):
        new_params = copy.deepcopy(params)
        for theory_name in self._gen_specs.keys():
            params_to_update = self._get_params_to_update_for_theory_name(theory_name)
            new_params.update(params_to_update)
        return new_params

    def _get_params_to_update_for_theory_name(self, theory_name):

        module = importlib.import_module(theory_name)
        par_gen = getattr(module, 'ParGenerator')()

        cobaya_par = CobayaPar(self._cobaya_par_file)
        survey_par = cobaya_par.get_survey_par()

        params = par_gen.get_params(survey_par, self._gen_specs[theory_name])

        return params

class TheoryParGenerator():

    def __init__(self):
        pass

    def get_params(self, **kwargs):
        """Returns a dictionary of parameters to update for the particular thoery"""
        pass
