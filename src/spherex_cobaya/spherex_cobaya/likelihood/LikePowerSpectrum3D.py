import numpy as np
import os
import pickle
import sys

from cobaya.likelihood import Likelihood

from spherex_cobaya.utils.log import LoggedError, class_logger
from lss_theory.utils.profiler import profiler


class LikePowerSpectrum3D(Likelihood):

    def initialize(self):
        """
        Prepares any computation, importing any necessary code, files, etc.
        """
        self.logger = class_logger(self)
        self.setup()

    def get_requirements(self):
        """
        Returns dictionary specifying quantities calculated by a theory code are needed
        """
        return {'galaxy_ps': None}

    #@profiler
    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """

        # TODO Placeholder for future derived parameter and foreground calculations.
        #derived_param = self.provider.get_param('derived_param')
        my_foreground_amp = params_values['my_foreground_amp']

        if self.is_reference_likelihood is True:
            print('is_reference_likelihood = True == > return chi2=0.')
            chi2 = 0.0
        else:
            delta = self.simulated_data - self.get_sampled_data()
            
            chi2 = 0.0
            for iz in range(self.nz):
                for ik in range(self.nk):
                    for imu in range(self.nmu):

                        tmp = np.matmul(self.invcov[:,:,iz,ik,imu], delta[:,iz,ik,imu])
                        chi2 = chi2 + np.matmul(delta[:,iz,ik,imu], tmp)

            self.logger.debug('chi2 = {}'.format(chi2))

        return -0.5 * chi2

    #@profiler
    def logp_old_format(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """

        # TODO Placeholder for future derived parameter and foreground calculations.
        #derived_param = self.provider.get_param('derived_param')
        my_foreground_amp = params_values['my_foreground_amp']

        if self.is_reference_likelihood is True:
            print('is_reference_likelihood = True == > return chi2=0.')
            chi2 = 0.0
        else:
            delta = self.simulated_data - self.get_sampled_data()

            # TODO turn into official error handling
            print('delta.shape = {}'.format(delta.shape))
            print('self.invcov.shape = {}'.format(self.invcov.shape))
            print('    expecting ({},{})'.format(
                np.prod(delta.shape), np.prod(delta.shape)))

            tmp = np.matmul(self.invcov, delta.ravel())
            chi2 = np.matmul(delta.ravel(), tmp)

            print('tmp.shape = {}'.format(tmp.shape))
            self.logger.debug('chi2 = {}'.format(chi2))

        return -chi2 / 2

    def setup(self):
        self.logger.info('Setting up likelihood ...')

        if self.is_reference_likelihood is False:
            self.simulated_data = self.load_simulated_data()
            self.invcov = self.load_invcov()
        else:
            print('==> Not loading inverse covariance and simulated data vector.')

        self.logger.info('... Done.')

    def load_invcov(self):

        self.logger.info('Loading invcov ...')

        (self.nps, self.nz, self.nk, self.nmu) = self.simulated_data.shape
        expected_shape = (self.nps, self.nps, self.nz, self.nk, self.nmu)

        try:
            self.logger.debug('self.invcov_path = {}'.format(self.invcov_path))
            
            invcov = np.load((self.invcov_path), allow_pickle=True)
            print('Done loading invcov.')
            if invcov.shape != expected_shape:
                msg = 'Inverse covariance at %s does not match data vector dimensions. \n' % self.invcov_path \
                    + 'Loaded invcov has shape %s, but expecting %s (from simulated_data with shape %s).' %\
                    (invcov.shape, expected_shape, self.simulated_data.shape)
                raise LoggedError(self.logger, msg)
            return invcov
        except FileNotFoundError as e:
            msg = '%s \n' % e \
                + 'Inverse covariance matrix does not exist. Run python scripts/generate_covariance.py first.'
            raise LoggedError(self.logger, msg)

    def load_simulated_data(self):
        try:
            #HACK needs proper transition
            self.logger.debug('self.sim_data_path = {}'.format(self.sim_data_path))
            results = pickle.load(open(self.sim_data_path, 'rb'))
            return results['galaxy_ps']
        except FileNotFoundError as e:
            msg = '%s' % e + '\n Simulated data vector does not exist.' \
                + '\n Run python scripts/generate_data.py first.'
            raise LoggedError(self.logger, msg)

    def get_sampled_data(self):
        galaxy_ps = self.provider.get_galaxy_ps()
        nuisance_model = self.get_nuisance_model()
        sample = galaxy_ps + nuisance_model
        return sample

    def get_nuisance_model(self):
        return 0.0
