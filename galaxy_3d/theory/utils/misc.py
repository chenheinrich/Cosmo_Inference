import numpy as np
import numbers

def evaluate_string_to_float(input):
    if isinstance(input, numbers.Number):
        return input
    elif type(input) is str:
        return float(eval(input))
    else:
        print('Wrong type: %s'%(type(input)))
