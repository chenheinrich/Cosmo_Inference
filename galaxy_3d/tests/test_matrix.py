import numpy as np
from theory.math_utils import matrix as m

def test_split_and_assemble():
    a = np.arange(16).reshape(4,4)
    b = m.split_matrix_into_blocks(a, 2, 2)
    c = m.assemble_matrix_from_blocks(b, 2)
    assert np.allclose(a, c)

"""Example usage: pytest ./galaxy_3d/tests/math_utils/"""