import numpy as np
import yaml

def test_B3D_RSD(cosmo_par_file, fn_expected):

    with open("./theory/tests/inputs/get_bis_rsd.yaml", 'r') as f:
        info_rsd = yaml.safe_load(f)

    info_rsd['cosmo_par_file'] = cosmo_par_file

    from theory.scripts.get_bis_rsd import get_data_vec_bis
    data_vec_b3d_rsd = get_data_vec_bis(info_rsd)

    exp = np.load(fn_expected)
    galaxy_bis = data_vec_b3d_rsd.get('galaxy_bis')
    diff = (galaxy_bis - exp)/exp
    print('diff', diff)

    #assert exp.shape == galaxy_bis.shape, ('Shapes of calculated vs imported data vector are not same')
    #print(galaxy_bis - exp)
    assert np.allclose(exp, galaxy_bis)

def main():
    
    cosmo_par_file = './inputs/cosmo_pars/planck2018_fiducial.yaml'
    fn_expected = './theory/tests/data/bis_rsd_fnl_0.npy'
    #fn_expected = './plots/theory/bispectrum_oriented_theta1_phi12_2_4/fnl_fiducial_test_no_AP_optimized_new_fog_Z1_functions/bis_rsd.npy'
    test_B3D_RSD(cosmo_par_file, fn_expected)

if __name__ is '__main__':
    main()