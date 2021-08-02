from cobaya.likelihood import Likelihood
import numpy as np
import os

class TestLikelihood(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """

        #self.data = np.loadtxt(self.cl_file)
        pass

    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed

         e.g. here we need C_L^{tt} to lmax=2500 and the H0 value
        """
        return {'Cl': {'tt': 2500}, 
                'H0': None,
                'CAMBdata': None,
                }

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        H0_theory = self.provider.get_param("H0")
        cls = self.provider.get_Cl(ell_factor=True)
        my_foreground_amp = params_values['my_foreground_amp']

        chi2 = 0.0
        return -chi2 / 2