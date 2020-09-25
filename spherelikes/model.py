import pickle
import os
import sys
import shutil
import pathlib
import yaml
import numpy as np
from collections import namedtuple

from cobaya.yaml import yaml_load_file
from cobaya.yaml import yaml_dump
from cobaya.model import get_model
from cobaya.tools import sort_cosmetic


class ModelCalculator():

    # TODO should add docstrings, so it's obvious calling it from an outside script what to do.

    def __init__(self, args):

        model_args = namedtuple('ModelArgs', sorted(args))
        self.args = model_args(**args)

        self.model_name = self.args.model_name
        self.model_yaml_file = self.args.model_yaml_file
        self.cobaya_yaml_file = self.args.cobaya_yaml_file
        self.output_dir = self.args.output_dir
        self.is_reference_model = self.args.is_reference_model

        self.setup_paths()
        self.setup_model()

    def setup_paths(self):
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.fname = os.path.join(
            self.output_dir, self.model_name + '.pickle')

    def setup_model(self):
        self._make_model()
        self._make_initial_pars()
        self._update_pars()

    def get_and_save_results(self):
        self.get_results()
        self.test_results()
        self.save_results()

    def get_results(self):
        """Returns a dictionary of cosmological results and auxiliary quantities.

        - results['aux'] is a dictionary with keys k, z, mu.
        - results['Hubble'] and results['angular_diameter_distance'] give H(z) and D_A(z) at a list of z.
        - results['galaxy_ps'] give a 4-d numpy array for the galaxy power spectra.
        - results['logposterior'] gives the log posterior.
        """

        # Cobaya: Cosmological observables requested this way always correspond
        # to the last set of parameters with which the likelihood was evaluated.
        self.logposterior = self.model.logposterior(self.point)

        aux = self._get_auxiliary_variables()
        redshifts = aux['z']

        theory1 = self.model.theory["camb"]
        theory1.must_provide(
            Hubble={"z": redshifts}, angular_diameter_distance={"z": redshifts})

        Hubble = theory1.get_Hubble(redshifts)
        angular_diameter_distance = theory1.get_angular_diameter_distance(
            redshifts)

        galaxy_ps = self.model.provider.get_galaxy_ps()

        H0 = theory1.get_param('H0')

        self.results = {
            'aux': aux,
            'Hubble': Hubble,
            'angular_diameter_distance': angular_diameter_distance,
            'galaxy_ps': galaxy_ps,
            'logposterior': self.logposterior,
            'H0': H0,
        }

        return self.results

    def save_results(self):
        pickle.dump(self.results, open(self.fname, "wb"))
        print('Results saved to %s.' % self.fname)
        self.copy_yaml_files()

    def test_results(self):
        # TODO how to not make this specific?
        theory = self.model.theory["theories.base_classes.ps_base.ps_base.PowerSpectrumBase"]
        ap = theory.get_AP_factor()
        assert np.all(ap == np.ones(theory.nz)), (ap, np.ones(theory.nz))

    def load_results(self):
        results = pickle.load(open(self.fname, "rb"))
        return results

    def copy_yaml_files(self):
        """Saves a copy of input model yaml file and cobaya sampler yaml file in the output directory."""
        yaml_path = os.path.join(
            self.output_dir, self.model_name + '.cosmo.yaml'
        )
        shutil.copy(self.model_yaml_file, yaml_path)
        print('Copied over input model yaml file to %s.' % yaml_path)

        yaml_path = os.path.join(
            self.output_dir, self.model_name + '.cobaya.yaml'
        )
        shutil.copy(self.cobaya_yaml_file, yaml_path)
        print('Saved upated cobaya yaml file to %s.' % yaml_path)

    def _make_model(self):
        info = yaml_load_file(self.cobaya_yaml_file)
        info = self._set_is_reference_model(info)
        self.model = get_model(info)

    def _make_initial_pars(self):
        self.point = dict(zip(
            self.model.parameterization.sampled_params(),
            self.model.prior.sample(ignore_external=True)[0]
        ))

    def _update_pars(self):
        fid_info = yaml_load_file(self.model_yaml_file)
        print('fid_info ', fid_info)
        self.point.update(fid_info)

    def _set_is_reference_model(self, info):
        key_to_set = 'is_reference_model'
        components = []
        for category in ['theory', 'likelihood']:
            for component in info[category].keys():
                if key_to_set in info[category][component].keys():
                    info[category][component]['is_reference_model'] = self.is_reference_model
                    components.append(category + ':, ' + component)
        print("Found components with key 'is_reference_model': {}".format(components))
        print('... Overwritten key is_reference_model = {} for these components.'.format(
            self.is_reference_model))
        return info

    # TODO need a more general interface here
    def _get_auxiliary_variables(self):
        """Return a dictionary aux with keys 'k', 'mu' and 'z' used for computing results.
        """
        # TODO (long term) how to not reference the particular theory?
        # maybe have an inheritance structure here? (not sure ...)
        theory = self.model.theory["theories.base_classes.ps_base.ps_base.PowerSpectrumBase"]
        theory.must_provide(galaxy_ps={}, ap={})
        k = theory.k
        mu = theory.mu
        z = theory.z
        nsample = theory.nsample
        nps = theory.nps
        aux = {'k': k, 'mu': mu, 'z': z, 'nsample': nsample, 'nps': nps}
        return aux


def main():

    args = {
        'model_name': 'model_debug',
        'model_yaml_file': './inputs/cosmo_pars/planck2018_fiducial.yaml',
        'cobaya_yaml_file': './inputs/cobaya_pars/ps_base.yaml',
        'output_dir': './data/debug',
    }

    fid_calculator = FidModelCalculator(args)
    fid_calculator.get_and_save_results()
    fid_calculator.load_results()


if __name__ == '__main__':
    main()
