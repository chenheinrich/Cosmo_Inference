import numpy as np
import numbers

def evaluate_string_to_float(input):
    if isinstance(input, numbers.Number):
        return input
    elif type(input) is str:
        return float(eval(input))
    else:
        print('Wrong type: %s'%(type(input)))


def strp(num, fmt=None):
    if num is None:
        return 'None'
    else:
        if fmt == None:
            string = str(num)
        else:
            string = fmt % num
        return string.replace('.', 'p')