"""One source of truth for effective CUDA optimization flags."""


def cuda_optimization_flags(ir):
    compiler = ir.get('compiler', {}) or {}
    flags = []
    fast_math = compiler.get('use_fast_math', False)
    if type(fast_math) is not bool:
        raise ValueError('compiler.use_fast_math must be a boolean')
    if fast_math:
        flags.append('--use_fast_math')
    limit = compiler.get('max_register_count')
    if limit is not None:
        if type(limit) is not int or not 16 <= limit <= 255:
            raise ValueError('max_register_count must be an integer between 16 and 255')
        flags.append(f'--maxrregcount={limit}')
    return flags
