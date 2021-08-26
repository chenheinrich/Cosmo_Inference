#Turn into a test later

import sys
import numpy as np
fn_test = sys.argv[1]
do_multitracer_unique = sys.argv[2]

fn_0 = './pipeline/sample_outputs/lss_theory/Bispectrum3DRSD/cosmo_planck2018_fnl_1p0/nk_11/do_folded_signal_True/theta_phi_2_4/b3d_rsd.npy'
bis0 = np.load(fn_0)

print('do_multitracer_unique = ', do_multitracer_unique )
if do_multitracer_unique:
    fn = './pipeline/sample_outputs/lss_theory/Bispectrum3DRSD/do_unique_multitracer_True/cosmo_planck2018_fnl_1p0/nk_11/do_folded_signal_True/theta_phi_2_4/b3d_rsd_first.npy'
else:
    fn = './pipeline/sample_outputs/lss_theory/Bispectrum3DRSD/cosmo_planck2018_fnl_1p0/nk_11/do_folded_signal_True/theta_phi_2_4/b3d_rsd.npy'

bis_expected = np.load(fn)
print('bis_expected.shape', bis_expected.shape)

bis_test = np.load(fn_test)
print('bis_test.shape', bis_test.shape)

print(bis_test - bis_expected)
print(np.allclose(bis_test, bis_expected))

if do_multitracer_unique:
    isamples_list_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    isamples_list_false = [0, 1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 18, 19, 24, 31]
    b0 = bis0[isamples_list_false, :, :, :]
    b1 = bis_test[isamples_list_true, :, :, :]
    print('partial:', np.allclose(b0, b1))




