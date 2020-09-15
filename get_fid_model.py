from cobaya.yaml import yaml_load_file
from cobaya.yaml import yaml_dump
from cobaya.model import get_model
from cobaya.tools import sort_cosmetic
import pickle
import os
import sys
import shutil
import pathlib
import yaml


# Package name (subject to change)
package_name = 'spherelikes'


class FidModelCalculator():

    def __init__(self, args):

        for key in args:
            setattr(self, key, args[key])

        self.make_model_dir()
        self.make_fname()
        self.make_model()
        self.make_point()

    def make_model_dir(self):
        self.model_dir = os.path.join(self.model_dir, self.model_name)
        pathlib.Path(
            self.model_dir).mkdir(parents=True, exist_ok=True)

    def make_fname(self):
        self.fname = os.path.join(
            self.model_dir, self.model_name + '.pickle')

    def make_model(self):
        info = yaml_load_file(self.cobaya_yaml_file)
        self.model = get_model(info)

    def make_point(self):
        self.make_initial_point()
        self.update_point()

    def make_initial_point(self):
        self.point = dict(zip(
            self.model.parameterization.sampled_params(),
            self.model.prior.sample(ignore_external=True)[0]
        ))

    def update_point(self):
        fid_info = yaml_load_file(self.model_yaml_file)
        print('fid_info ', fid_info)
        self.point.update(fid_info)
        # {'logA': 3.06, 'ns': 0.966, 'omch2': 0.1,
        # 'ombh2': 0.02, 'tau': 0.07, 'theta_MC_100': 1.0,
        # 'my_foreground_amp': 1.0,
        # 'fnl': 1.0,
        # 'gaussian_bias_1': 1.0,
        # 'gaussian_bias_2': 1.0,
        # })

    def get_results(self):

        # Cosmological observables requested this way always correspond to the last set of parameters with which the likelihood was evaluated.
        self.logposterior = self.model.logposterior(self.point)

        aux = self.get_auxiliary_variables()
        redshifts = aux['z']

        theory1 = self.model.theory["camb"]
        theory1.must_provide(
            Hubble={"z": redshifts}, angular_diameter_distance={"z": redshifts})

        Hubble = theory1.get_Hubble(redshifts)
        angular_diameter_distance = theory1.get_angular_diameter_distance(
            redshifts)
        galaxy_ps = self.model.provider.get_galaxy_ps()

        self.results = {
            'aux': aux,
            'Hubble': Hubble,
            'angular_diameter_distance': angular_diameter_distance,
            'galaxy_ps': galaxy_ps,
            'logposterior': self.logposterior
        }

    def get_auxiliary_variables(self):  # TODO this is too specific, maybe?
        theory = self.model.theory["theories.base_classes.ps_base.ps_base.PowerSpectrumBase"]
        theory.must_provide(galaxy_ps={})
        k = theory.k
        mu = theory.mu
        z = theory.z
        aux = {'k': k, 'mu': mu, 'z': z}
        return aux

    def save_results(self):
        pickle.dump(self.results, open(self.fname, "wb"))
        print('Results saved to %s.' % self.fname)

    def save_yaml_files(self):
        yaml_path = os.path.join(self.model_dir, 'model.yaml')
        shutil.copy(self.model_yaml_file, yaml_path)
        print('Copied over input model yaml file to %s.' % yaml_path)

        yaml_path = os.path.join(self.model_dir, 'cobaya.yaml')
        shutil.copy(self.cobaya_yaml_file, yaml_path)
        print('Saved upated cobaya yaml file to %s.' % yaml_path)

    def get_and_save_results(self):
        self.get_results()
        self.save_results()
        self.save_yaml_files()

    def check_load_results(self):
        results = pickle.load(open(self.fname, "rb"))
        print('results = ', results)


def main():
    args = {
        'model_name': 'model_test',
        'model_yaml_file': '%s/inputs/sample_fid_model.yaml' % package_name,
        'cobaya_yaml_file': '%s/inputs/sample.yaml' % package_name,
        'model_dir': '%s/data/models/' % (package_name),
    }

    fid_calculator = FidModelCalculator(args)
    fid_calculator.get_and_save_results()
    # check: # TODO turn into test?
    fid_calculator.check_load_results()


if __name__ == '__main__':
    main()
