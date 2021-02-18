import os
import errno
import numpy as np

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_file_npy(fn, data):
    np.save(fn, data)
    print('Saved file: {}'.format(fn))