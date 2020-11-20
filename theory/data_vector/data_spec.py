

class DataSpec():

    def __init__(self, data_spec_dict):
        self._dict = data_spec_dict

    def get_k(self):
        k = [1e-4, 1e-3, 1e-2, 1e-1]
        return k

