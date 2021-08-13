
class NameNotAllowedError(Exception):

    def __init__(self, name, allowed_names):
        self.msg = '%s is not in the list of allowed names: %s'%(name, allowed_names)
