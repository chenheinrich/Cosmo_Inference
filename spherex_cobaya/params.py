import numpy as np

from cobaya.yaml import yaml_load_file
from spherex_cobaya.utils.log import class_logger

class SurveyParFileError(Exception):
    pass

class CobayaPar():

    """Interface for the cobaya parameter file, delivering quantities such as
    ...."""
    
    def __init__(self, cobaya_par_file):

        self._logger = class_logger(self)
        self._cobaya_par_file = cobaya_par_file
        self._info = yaml_load_file(self._cobaya_par_file)
        self._run_checks()
        
    def _run_checks(self):
        pass

    def get_theory_list(self):
        return self._info['theory'].keys()
    
    def get_likelihood_list(self):
        return self._info['likelihood'].keys()

    def get_spherex_theory_list(self):
        theories = self.get_theory_list()
        spherex_theories = [name for name in theories if name.startswith('sphere')]
        print('spherex_theories', spherex_theories)
        return spherex_theories

    def get_spherex_theory(self, use_grs_ingredients=True):
        if use_grs_ingredients is True:
            return 'spherex_cobaya.theory.GRSIngredients'
        else:
            first_theory_name = list(self._info['theory'].keys())[1]
            return first_theory_name

    def get_survey_par(self):
        try:
            theory_name = self.get_spherex_theory(use_grs_ingredients=True)
            survey_par_file = self._info['theory'][theory_name]['survey_par_file']
        except KeyError as e:
            theory_name = self.get_spherex_theory(use_grs_ingredients=False)
            self._logger.debug('theory_name = {}'.format(theory_name))
            survey_par_file = self._info['theory'][theory_name]['survey_par_file']
        
        survey_par = SurveyPar(survey_par_file)
        return survey_par

    @property
    def filename(self):
        return self._cobaya_par_file

    def get_params(self):
        return self._info['params']

#TODO need to refactor with galaxy_3d/theory/params.py:SurveyPar
class SurveyPar():

    """Interface for the survey parameter file, delivering quantities such as
    number densities, galaxy biases, nz, nsample, redshifts and errors."""
    
    def __init__(self, survey_par_file):
        self._survey_par_file = survey_par_file
        self._info = yaml_load_file(self._survey_par_file)
        
        self._nz = self.get_nz()
        self._nsample = self.get_nsample()
        self._run_checks()


    @property
    def filename(self):
        return self._survey_par_file

    def get_nz(self):
        return int(self._info['nz'])

    def get_nsample(self):
        return int(self._info['nsample'])
        
    def _run_checks(self):
        self._check_arrays_are_length_nz()
        self._check_arrays_are_length_nsample()
    
    def _check_arrays_are_length_nz(self):
        for key in self._info.keys():
            if key not in ['nz', 'nsample', 'sigz_over_one_plus_z']:
                nz = len(self._info[key])
                if nz != self._nz:
                    msg = "lenghth of array for {} is {}, not consistent \
                        with nz = {} in file".format(key, nz, self._nz)
                    raise SurveyParFileError(msg)
    
    def _check_arrays_are_length_nsample(self):
        key = 'sigz_over_one_plus_z'
        nsample = len(self._info[key])
        if nsample != self._nsample:
            msg = "lenghth of array for {} is {}, not consistent \
                with nz = {} in file".format(key, nsample, self._nsample)
            raise SurveyParFileError(msg)
        
    def get_number_density_array(self):
        """Returns 2d numpy array of shape (nsample, nz) 
        for number density in h/Mpc"""
        data = np.empty((self._nsample, self._nz))
        for i in range(self._nsample):
            data[i,:] = self._info['number_density_in_hinvMpc_%s'%(i+1)]
        return data

    def get_galaxy_bias_array(self):
        """Returns 2d numpy array of shape (nsample, nz) 
        for the Gaussian galaxy bias"""
        data = np.empty((self._nsample, self._nz))
        for i in range(self._nsample):
            data[i,:] = self._info['galaxy_bias_%s'%(i+1)]
        return data
    
    def get_zlo_array(self):
        return np.array(self._info['zbin_lo'])

    def get_zhi_array(self):
        return np.array(self._info['zbin_hi'])
    
    def get_zmid_array(self):
        return 0.5 * (self.get_zlo_array() + self.get_zhi_array())
    
    def get_sigz_array_over_one_plus_z(self):
        return np.array(self._info['sigz_over_one_plus_z'])

    def get_sigz_array(self):
        """Returns 2d numpy array of shape (nsample, nz)
        for the sigma_z the redshift error."""
        zmid = self.get_zmid_array()
        sigz_over_one_plus_z = self.get_sigz_array_over_one_plus_z()
        return sigz_over_one_plus_z[:, np.newaxis] * (1.0 + zmid[np.newaxis, :])
