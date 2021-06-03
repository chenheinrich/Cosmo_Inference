import cProfile
import io
import os
import pstats
from datetime import datetime
import errno

def profiler(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        cwd = os.getcwd()

        profiler_dir = os.path.join(cwd, 'profiler/')
        mkdir_p(profiler_dir)
        
        filename = get_filename(func.__name__)
        filepath = os.path.join(profiler_dir, filename)

        try:
            with open(filepath, 'w') as f:
                f.write(s.getvalue())
            print('Saved profiling results to file: {}'.format(filepath))
        except Exception as e:
            print(e)

        return retval

    return wrapper

def get_filename(func_name):
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    name = func_name + '_'  + time_string + '.profile'
    return name

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise