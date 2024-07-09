import threading

"""

apply(func, ((arg1, arg2) for arg2 in seq)
"""

def apply(*args, **kwargs):
    return apply_single(*args, **kwargs)
    # return apply_threaded(*args, **kwargs)
    # return apply_multiprocessing(*args, **kwargs)

def apply_single(func, args, single=False, **kwargs):
    if single:
        return {key: func(args[key], **kwargs) for key in args}
    return {key: func(*args[key], **kwargs) for key in args}


def _apply_single(results, func, args, **kwargs):
    results.update({key: func(args[key], **kwargs) for key in args}) 

def _apply_multi(results, func, args, **kwargs):
    results.update({key: func(*args[key], **kwargs) for key in args}) 

def apply_threaded(func, args, nthread=2, single=False, **kwargs):
    if not args:
        return {}

    keys = list(args)
    chunk_size = len(keys) // nthread

    _apply = _apply_single if single else _apply_multi
    
    threads = []
    results = {}
    for i in range(nthread):
        start = i * chunk_size
        end = start + chunk_size if i < nthread - 1 else len(keys)
        subset = {key: args[key] for key in keys[start:end]}
        thread = threading.Thread(target=_apply, args=(results, func, subset), kwargs=kwargs)
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    return results