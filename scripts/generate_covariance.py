import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os

from cobaya.yaml import yaml_load_file

from spherelikes.model import ModelCalculator


class CovCalculator():

    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.galaxy_ps = self.data['galaxy_ps']
        self.nsample = self.data['aux']['nsample']
        self.make_dict()
        self.load_number_density()
        # TODO turn into unit tests
        assert self.dict_ips_from_sample_pair['0,1'] == 1
        assert self.dict_ips_from_sample_pair['1,1'] == self.nsample
        assert self.dict_sample_pair_from_ips['%s' % self.nsample] == (1, 1)
        assert self.dict_sample_pair_from_ips['1'] == (0, 1)

    def get_cov(self):
        """Returns the covariance matrix for all power spectra between different 
        galaxy samples, assuming it is diagonal in z, k and mu. 

        We are also assuming that the covariance is symmetric under j1j2 <--> j1'j2':
            Cov[P_{j1, j2} P{j1', j2'}] = Cov[P_{j1', j2'} P{j1, j2}] 
        where j1, j2, j1', j2' are indices of the galaxy samples forming the 
        power spectra, and the z, k, mu dependence in P_{j1, j2}(z, k, mu) have been
        suppressed here for reasons mentioned above. 
        """
        # Careful: Do not change elements in self.data_1d, it would modify self.data
        # from outside of this class since it is created w/ np.ravel().
        self.get_data_1d()
        self.shape = self.galaxy_ps.shape
        (self.nps, self.nz, self.nk, self.nmu) = self.shape
        ntot = np.prod(self.shape)

        self.cov = np.zeros((ntot, ntot))
        for ips1 in range(self.nps):
            for ips2 in range(self.nps):
                print('(ips1, ips2) = ({} {})'.format(ips1, ips2))
                block_size = np.prod(self.shape[1:])
                ind1 = ips1 * block_size
                ind2 = ips2 * block_size
                block = self.get_block(ips1, ips2)
                self.cov[ind1:(ind1 + block_size), ind2:(ind2 + block_size)]\
                    = block

        self.block_size = block_size

        return self.cov

    def test_cov_is_symmetric_exchanging_ips1_and_ips2(self):
        # TODO turn into unit test that covariance is symmetric under ips1<-->ips2
        # Note: if covariance expression is no longer symmetric in ips1<-->ips2, remove test
        ips1 = np.random.randint()
        ips2 = np.random.randint()
        self.get_cov()

        block_size = np.prod(self.shape[1:])
        ind1 = ips1 * block_size
        ind2 = ips2 * block_size

        block12 = self.cov[ind1:(ind1 + block_size), ind2:(ind2 + block_size)]
        block21 = self.cov[ind2:(ind2 + block_size), ind1:(ind1 + block_size)]
        assert np.allclose(block12 == block12)

    def get_and_save_invcov(self):
        self.get_cov()
        self.get_invcov()
        self.do_inv_test(self.cov, self.invcov)
        self.save()

    def save(self, fn=None):
        fn = os.path.join(self.args['output_dir'], 'fid.covmat')
        fn_invcov = os.path.join(self.args['output_dir'], 'fid.invcov')
        np.save(fn, self.cov)
        np.save(fn_invcov, self.invcov)
        print('Saved covariance matrix: {}'.format(fn))
        print('Saved inverse covariance matrix: {}'.format(fn_invcov))

    def do_test_cov_block_is_invertible(self, nps):
        # TODO turn into unit test
        """Plot a block of the covariance matrix, invert it
        and check precision."""
        cov0 = self.cov[:nps * self.block_size, :nps * self.block_size]
        cov0 = cov0 * 1e-9
        self.plot_mat(
            cov0, plot_name='plot_cov_block_%i_ps_x_%i_ps.png' % (nps, nps))

        print('determinant = {}'.format(np.linalg.det(cov0)))
        inv = np.linalg.inv(cov0)
        self.do_inv_test(cov0, inv)

    def do_inv_test(self, mat, invmat, threshold=1e-3):
        id0 = np.diag(np.ones(mat.shape[0]))
        id1 = np.matmul(mat, invmat)
        id2 = np.matmul(invmat, mat)
        ind_big1 = np.where(id1 - id0 > threshold)
        ind_big2 = np.where(id2 - id0 > threshold)
        print(
            'inv test mat * invmat did not pass for indices {} at tolerance = {}'.format(ind_big1, threshold))
        print(
            'inv test invmat * mat did not pass for indices {} at tolerance = {}'.format(ind_big2, threshold))
        for i in range(ind_big1[0].size):
            x = ind_big1[0][i]
            y = ind_big1[1][i]
            print('(id1 - id0)[{},{}] = {}'
                  .format(x, y, id1[x, y] - id0[x, y]))
        for i in range(ind_big2[0].size):
            x = ind_big2[0][i]
            y = ind_big2[1][i]
            print('(id2 - id0)[{},{}] = {}'
                  .format(x, y, id2[x, y] - id0[x, y]))

    def get_invcov(self, rescale=1.0):
        try:
            self.invcov = np.linalg.inv(self.cov * rescale)
            print('self.invcov.shape = {}'.format(self.invcov.shape))
            return self.invcov
        except Exception as e:
            print('Error: {}'.format(e))

    def get_noise(self, a, b):
        """Return shot noise = 1/n if the same galaxy sample, for all galaxy redshift.
        Args:
            a, b are integers denoting indices (starting at 0) for galaxy samples."""
        if a == b:
            noise = np.array([1. / self.number_density[a, iz]
                              * np.ones((self.nk, self.nmu))
                              for iz in range(self.nz)])
            assert noise.shape == (self.nz, self.nk, self.nmu)
        else:
            noise = np.zeros((self.nz, self.nk, self.nmu))
        return noise

    def test_get_noise(self):  # TODO use for unit test later
        assert self.get_noise(1, 1)[0, 0, 0] == 8.13008130e+01
        assert self.get_noise(1, 1)[-1, 0, 0] == 1.47275405e+06

    def load_number_density(self):
        """Loads number density into a 2-d numpy array of shape (nsample, nz) in units of 1/Mpc,
        expecting number density in input_survey_pars are given in h/Mpc."""
        h = self.data['H0'] / 100.0
        pars = yaml_load_file(self.args['input_survey_pars'])
        # TODO need to do this in a way that's independent of number of samples.
        self.number_density = h * np.array(
            [pars['numdens1'], pars['numdens2'], pars['numdens3'], pars['numdens4'], pars['numdens5']])

    def get_galaxy_ps(self, a, b):
        ips = self.dict_ips_from_sample_pair['%i,%i' % (a, b)]
        ps = self.galaxy_ps[ips, :, :, :]
        return ps

    def get_observed_ps(self, a, b):
        ps = self.get_galaxy_ps(a, b) + self.get_noise(a, b)
        return ps

    def get_block(self, ips1, ips2):
        """Returns a Cov[P_{1}, P_{2}] where ips1, ips2 are integer indices
        for the galaxy power spectra.
        """
        (a, b) = self.dict_sample_pair_from_ips['%i' % ips1]
        (c, d) = self.dict_sample_pair_from_ips['%i' % ips2]
        cov = self.get_observed_ps(a, c) * self.get_observed_ps(b, d) \
            + self.get_observed_ps(a, d) * self.get_observed_ps(b, c)
        array = np.ravel(cov)
        msg = 'array.size = {}, expect {}'.format(
            array.size, np.prod(self.shape[1:]))
        block = np.diag(array)
        assert array.size == np.prod(self.shape[1:]), (msg)
        print('block.shape = {}'.format(block.shape))
        return block

    # TODO to turn into a unit test later
    def test_cov_block_constructed_correctly(self):
        """Randomly draw a block Cov[P_{axb}, P_{cxd}] where a, b, c, d are galaxy sample indices."""
        # nsample = (-1 + np.sqrt(1 + 8 * self.nps))/2.0 # TODO use this and put in utils later
        nsample = 5
        a = np.random.randint(nsample)
        b = np.random.randint(nsample)
        c = np.random.randint(nsample)
        d = np.random.randint(nsample)
        ips1 = self.dict_ips_from_sample_pair['%i,%i' % (a, b)]
        ips2 = self.dict_ips_from_sample_pair['%i,%i' % (c, d)]
        block = self.get_block(ips1, ips2)
        # self.get_observed_ps(a,c)
        ips_ac = self.dict_ips_from_sample_pair['%i,%i' % (a, c)]
        ips_bd = self.dict_ips_from_sample_pair['%i,%i' % (b, d)]
        ips_ad = self.dict_ips_from_sample_pair['%i,%i' % (a, d)]
        ips_bc = self.dict_ips_from_sample_pair['%i,%i' % (b, c)]

        iz = np.random.randint(self.nz)
        ik = np.random.randint(self.nk)
        imu = np.random.randint(self.nmu)

        idx = iz * self.nk * self.nmu + ik * self.nmu + imu
        block_entry = block[idx, idx]
        expected = (self.galaxy_ps[ips_ac, iz, ik, imu] + 1. / self.number_density[a, iz] * (a == c)) \
            * (self.galaxy_ps[ips_bd, iz, ik, imu] + 1. / self.number_density[b, iz] * (b == d)) \
            + (self.galaxy_ps[ips_ad, iz, ik, imu] + 1. / self.number_density[a, iz] * (a == d)) \
            * (self.galaxy_ps[ips_bc, iz, ik, imu] + 1. / self.number_density[b, iz] * (b == c))
        msg = 'block[{}, {}]={}, \n  expected={} for PS({},{}) x PS({},{}), for (iz, ik, imu) = ({}, {}, {})'.format(
            iz, iz, block_entry, expected, a, b, c, d, iz, ik, imu)
        print(msg)
        assert block_entry == expected, (msg)

        self.plot_mat(
            block, plot_name='plot_block_%i_%i_x_%i_%i.png' % (a, b, c, d))

    def make_dict(self):
        """Creates two dictionaries to map between power spectrum index and galaxy sample pairs.

        Note: For example, (j1, j2) = (0,1) is mapped to ips = 0, (1,1) is mapped to ips = 5 
        if there are 5 different galaxy samples. 

        In general, the power spectrum index ips is mapped to (j1, j2) the galaxy sample indices,
        where j1 < j2, while both (j1, j2) and (j2, j1) are mapped to the same power spectrum index.
        """

        self.dict_ips_from_sample_pair = {}
        self.dict_sample_pair_from_ips = {}

        ips = 0
        for j1 in range(self.nsample):
            for j2 in range(j1, self.nsample):

                self.dict_sample_pair_from_ips['%i' % ips] = (j1, j2)
                self.dict_ips_from_sample_pair['%i,%i' % (j1, j2)] = ips
                self.dict_ips_from_sample_pair['%i,%i' % (j2, j1)] = ips

                print('(j1, j2) = ({}, {}), ips = {}'.format(j1, j2, ips))

                ips = ips + 1

    def get_data_1d(self):
        """Flatten the 4d numpy array self.galaxy_ps into 1d data vector, 
        to facilitate the computation of the covariance matrix."""

        self.data_1d = self.galaxy_ps.ravel()

        msg = 'self.data_1d.size = {}, expect {}'\
            .format(self.data_1d.size, np.prod(self.galaxy_ps.shape))
        assert self.data_1d.size == np.prod(self.galaxy_ps.shape), (msg)

        return self.data_1d

    def plot_mat(self, mat, plot_name=None):
        """Plot input 2d matrix mat and save in self.output_dir w/ input plot_name."""

        mat1 = np.ma.masked_where(mat <= 0, mat)

        fig, ax = plt.subplots()
        cs = ax.imshow(mat1, norm=LogNorm(), cmap='RdBu')
        cbar = fig.colorbar(cs)

        if plot_name is None:
            plot_name = os.path.join(self.output_dir, 'plot_mat.png')

        fig.savefig(plot_name)
        print('Saved plot: {}'.format(plot_name))


def main():

    CWD = os.getcwd()
    args = {
        'model_name': 'covariance_debug',
        'model_yaml_file': CWD + '/inputs/cosmo_pars/planck2018_fiducial.yaml',
        'cobaya_yaml_file': CWD + '/inputs/cobaya_pars/ps_base.yaml',
        'input_survey_pars': './inputs/survey_pars/survey_pars_v28_base_cbe.yaml',
        'output_dir': CWD + '/data/ps_base/',
    }

    args['is_reference_model'] = True
    args['is_reference_likelihood'] = True

    model_calc = ModelCalculator(args)
    results = model_calc.get_results()

    cov_calc = CovCalculator(results, args)
    cov_calc.get_and_save_invcov()


if __name__ == '__main__':

    """Computes and saves covariance matrix and its inverse, using an input
    fiducial cosmology (need to be the same as reference cosmollogy for AP).

    Usage: python scripts/generate_covariance.py

    Note: We set is_reference_model = True automatically in this script to 
    avoid calculating AP effects and bypass likelihood calculations. 

    Note: You can also disable the likelihood calculation to not load elements 
    yet to be calculated (e.g. inverse covariance and simulated data vectors) 
    by setting is_reference_likelihood = True.
    """

    main()
