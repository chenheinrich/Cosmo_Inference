# decompyle3 version 3.3.2
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.9 (default, Aug 31 2020, 07:22:35)
# [Clang 10.0.0 ]
# Embedded file name: /Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/SphereLikes/spherelikes/likelihoods/base_classes/ps_base/ps_base.py
# Compiled at: 2020-09-21 17:58:58
# Size of source mod 2**32: 3032 bytes
import numpy as np
import os
import pickle
from cobaya.likelihood import Likelihood


class LikelihoodBase(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        print('Setting up likelihood')
        self.setup()

    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed
        """
        return {'galaxy_ps': None,
                'derived_param': None}

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        galaxy_ps = self.provider.get_galaxy_ps()
        derived_param = self.provider.get_param('derived_param')
        my_foreground_amp = params_values['my_foreground_amp']
        if self.is_fiducial_model is False:
            delta = self.data - self.get_sample()
            # TODO turn into official error handling
            print('delta.shape = {}'.format(delta.shape))
            print('self.invcov.shape = {}'.format(self.invcov.shape))
            print('    expecting ({},{})'.format(
                np.prod(delta.shape), np.prod(delta.shape)))
            tmp = np.matmul(self.invcov, delta.ravel())
            print('tmp.shape = {}'.format(tmp.shape))
            chi2 = np.matmul(delta.ravel(), tmp)
            print('chi2 = {}'.format(chi2))
        else:
            print('is_fiducial_model = True == > return chi2=0.')
            chi2 = 0.0
        return -chi2 / 2

    def setup(self):
        if self.is_fiducial_model is False:
            self.invcov = self.load_invcov()
            self.data = self.load_data()
        else:
            print('is_fiducial_model = True ==> Not loading fiducial inverse covariance and simulated data vector.')
        print('Done setting up')

    def load_invcov(self):
        print('Loading invcov ...')
        invcov = np.load((self.invcov_path), allow_pickle=True)
        #n = np.prod(self.provider.get_galaxy_ps().shape)
        #assert invcov.shape == (n, n)
        print('Done loading invcov.')
        return invcov

    def load_data(self):
        results = pickle.load(open(self.sim_data_path, 'rb'))
        data = results['galaxy_ps']
        # msg = 'data.shape = {}, but need shape = {}'.format(
        #    data.shape, self.provider.get_galaxy_ps().shape)
        #assert data.shape == self.provider.get_galaxy_ps().shape, msg
        return data

    def get_sample(self):
        galaxy_ps = self.provider.get_galaxy_ps()
        nuisance_model = self.get_nuisance_model()
        sample = galaxy_ps + nuisance_model
        return sample

    def get_nuisance_model(self):
        return 0.0
# okay decompiling ps_base.cpython-37.pyc
