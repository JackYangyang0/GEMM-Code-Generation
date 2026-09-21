"""Compiler hints must not regenerate the GEMM algorithm."""
import re

HINTS = {'Compiler.RestrictPointer', 'Compiler.ForceInline.DeviceFunctions'}


def apply_compiler_hint(source, strategy_id):
    if strategy_id == 'Compiler.RestrictPointer':
        signature = re.compile(r'(__global__\s+(?:__launch_bounds__\([^\n]+?\)\s*)?void\s+gemm\s*\()([^{};]+)(\)\s*\{)')
        matches = list(signature.finditer(source))
        if len(matches) != 1:
            raise ValueError('RestrictPointer requires one recognized gemm kernel signature')
        match = matches[0]
        params = match[2]
        for name in ('A', 'B', 'C'):
            pattern = r'(\b(?:const\s+)?float\s*\*\s*)(?:__restrict__\s*)?(\b' + name + r'\b)'
            params, count = re.subn(pattern, r'\1__restrict__ \2', params)
            if count != 1:
                raise ValueError('Unrecognized GEMM pointer parameter: ' + name)
        return source[:match.start(2)] + params + source[match.end(2):]
    if strategy_id == 'Compiler.ForceInline.DeviceFunctions':
        # Only explicit device-only definitions. Never inline/rename a kernel.
        pattern = r'\b__device__\s+(?!(?:__forceinline__|__global__)\b)(?:inline\s+)?(?=[\w:*<> \t]+\s+\w+\s*\([^;{}]*\)\s*\{)'
        return re.sub(pattern, '__device__ __forceinline__ ', source)
    raise ValueError('Not a narrow compiler hint: ' + strategy_id)
