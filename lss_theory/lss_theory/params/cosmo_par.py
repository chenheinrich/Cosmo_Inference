import numpy as np
import yaml

class CosmoPar(object):
    def __init__(self, cosmo_par_file, overwrite_dict=None):
        self.cosmo_par_file = cosmo_par_file
        self._set_par(overwrite_dict)

        self._params_list = list(self.__dict__.keys()) # HACK Test if this works!

    @property
    def params_list(self):
        return self._params_list

    def _set_par(self, overwrite_dict):

        #TODO add error handling
        with open(self.cosmo_par_file) as f:
            info = yaml.load(f, Loader=yaml.FullLoader)

        for k, v in info.items():
            setattr(self, k, v)

        if overwrite_dict is not None:
            for k, v in overwrite_dict.items():
                setattr(self, k, v)

        self._add_aux_par()
        
        #TODO turn into unit test
        # print('self.fnl', self.fnl)

    def _add_aux_par(self):

        if (hasattr(self, 'logA') is True) and (hasattr(self, 'As') is False):
            self.As = 1e-10*np.exp(self.logA)

        if (hasattr(self, 'theta_MC_100') is True) and (hasattr(self, 'cosmomc_theta') is False):
            self.cosmomc_theta = self.theta_MC_100/100.0

    
