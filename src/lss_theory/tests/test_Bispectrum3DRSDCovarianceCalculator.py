import pytest
import os
import copy
import numpy as np
import yaml 


@pytest.mark.debug
@pytest.mark.parametrize("yaml_file, iz, itri, expected", \
    [
        ("./galaxy_3d/tests/data/get_covariance_bis_rsd.yaml", 0, 0, 7.694486671443942e+28),
        ("./galaxy_3d/tests/data/get_covariance_bis_rsd.yaml", 0, 1, 3.0725297000383867e+28)
    ]
)
def test_Bispectrum3DRSDCovarianceCalculator(yaml_file, iz, itri, expected):
    
    from lss_theory.covariance import Bispectrum3DRSDCovarianceCalculator

    with open(yaml_file) as f:
        info = yaml.load(f, Loader=yaml.FullLoader)

    cov_calculator = Bispectrum3DRSDCovarianceCalculator(info)
    
    ntri = cov_calculator._b3d_rsd_spec.ntri
    iblock = iz * ntri + itri

    ib = 0
    ans = cov_calculator.get_cov_smallest_nondiagonal_block(iz, itri)[ib,ib]
    ans = ans * cov_calculator._cov_rescale[iblock]
    assert (ans == expected), (ans, expected)

    #TODO add invcov test too
    #invcov = cov_calculator.get_invcov()
        

