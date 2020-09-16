from cobaya.likelihood import Likelihood
from cobaya.likelihood import Likelihood
import numpy as np
import os


class LikelihoodBase(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """

        self.data = np.loadtxt(self.cl_file)

    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed
        """
        return {'galaxy_ps': None, 'derived_param': None}

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        #H0_theory = self.provider.get_param("H0")
        galaxy_ps = self.provider.get_galaxy_ps()
        derived_param = self.provider.get_param('derived_param')

        print('==>galaxy_ps from my_like_class', galaxy_ps)
        print('==>derived_param from my_like_class', derived_param)

        my_foreground_amp = params_values['my_foreground_amp']

        chi2 = 1.0
        return -chi2 / 2
