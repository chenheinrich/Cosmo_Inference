

class OtherPar():
      
    def __init__(self, dict_in):
          
        self._dict_in = dict_in
        self._params_list = list(self._dict_in.keys())

        for key in dict_in:
            setattr(self, key, dict_in[key])
    
    @property
    def params_list(self):
        return self._params_list