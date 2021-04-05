import numpy as np

from cobaya.yaml import yaml_load_file

class SurveyParFileError(Exception):
    pass

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
