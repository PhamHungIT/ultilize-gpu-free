import inspect
import time
from functools import wraps

from common.utils import logger as default_logger

timeit_total = {}
timeit_hit = {}


def _is_method(func):
    spec = inspect.getargspec(func)
    return spec.args and spec.args[0] == 'self'


def timeit(*iargs, **ikwargs):
    def inner(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            t = te - ts

            trace_args = args
            if _is_method(func):
                try:
                    func_self = args[0].__class__.__qualname__
                    func_name = f"{func_self}.{func.__name__}"
                    trace_args = args[1:]
                except:
                    func_name = func.__qualname__
            else:
                func_name = func.__qualname__

            total = timeit_total.get(func_name, 0) + t
            timeit_total[func_name] = total

            hit = timeit_hit.get(func_name, 0) + 1
            timeit_hit[func_name] = hit

            average = total / hit * 1000
            t = t*1000

            if 'trace_args' in ikwargs and ikwargs['trace_args']:
                msg = f"--- {func_name}({trace_args}, {kwargs}) --- took: {t:.2f}ms, hit: {hit}, average: {average:.2f}ms"
            else:
                msg = f"--- {func_name}() --- took: {t:.2f}ms, hit: {hit}, average: {average:.2f}ms"

            try:
                if 'logger' in ikwargs and ikwargs['logger']:
                    if 'print' == ikwargs['logger']:
                        print(msg)
                    elif 'logger' == ikwargs['logger']:
                        default_logger.debug(msg)
                    else:
                        logger = ikwargs['logger']
                        logger.debug(msg)
                else:
                    default_logger.debug(msg)
            except Exception as ex:
                default_logger.error(f"--- Logger or Msg error: {ex}")
                default_logger.debug(f"Msg: {msg}")

            return result
        return decorated

    if len(iargs) == 1 and callable(iargs[0]):
        # No arguments, this is the decorator
        # Set default values for the arguments
        return inner(iargs[0])
    else:
        # This is just returning the decorator
        return inner
