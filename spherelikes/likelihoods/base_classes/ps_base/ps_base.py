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
        return {'A': None, 'tt_sum': None}

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        #H0_theory = self.provider.get_param("H0")
        cls = self.provider.get_A()
        tt_sum = self.provider.get_param('tt_sum')

        print('====> cls from my_like_class', cls)
        print('====> tt_sum from my_like_class', tt_sum)

        my_foreground_amp = params_values['my_foreground_amp']

        chi2 = 1.0
        return -chi2 / 2
