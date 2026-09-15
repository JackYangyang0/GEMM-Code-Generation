"""Read attributes of CUDA logical device zero, respecting CUDA_VISIBLE_DEVICES."""
from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path


# cudaDeviceAttr values from CUDA driver_types.h; unsupported attributes stay unknown.
ATTRIBUTES = {
    "max_threads_per_block": 1, "max_shared_memory_per_block_bytes": 8,
    "warp_size": 10, "registers_per_block": 12, "multiprocessor_count": 16,
    "l2_cache_bytes": 38, "max_threads_per_multiprocessor": 39,
    "compute_capability_major": 75, "compute_capability_minor": 76,
    "max_shared_memory_per_multiprocessor_bytes": 81,
    "registers_per_multiprocessor": 82, "max_shared_memory_per_block_optin_bytes": 97,
    "max_blocks_per_multiprocessor": 106,
    "max_persisting_l2_cache_bytes": 108, "max_access_policy_window_bytes": 109,
}


def query_cuda_attributes(runtime=None) -> dict:
    if runtime is None:
        candidates = [ctypes.util.find_library("cudart")]
        if os.name == "nt":
            roots = [Path(os.environ[key]) for key in ("CUDA_PATH", "CUDA_HOME") if os.environ.get(key)]
            roots += list(Path(os.environ.get("ProgramFiles", "C:/Program Files"),
                               "NVIDIA GPU Computing Toolkit/CUDA").glob("v*"))
            candidates += [str(path) for root in roots for path in (root / "bin").glob("cudart64_*.dll")]
        else:
            candidates += ["libcudart.so"]
            candidates += [str(path) for root in [Path("/usr/local/cuda"), *Path("/usr/local").glob("cuda-*")]
                           for path in (root / "lib64").glob("libcudart.so*")]
        for candidate in dict.fromkeys(candidates):
            if not candidate:
                continue
            try:
                loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
                runtime = loader(candidate)
                break
            except OSError:
                continue
    if runtime is None:
        return {}
    try:
        query = runtime.cudaDeviceGetAttribute
        query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        query.restype = ctypes.c_int
        result = {}
        for name, attribute in ATTRIBUTES.items():
            value = ctypes.c_int()
            if query(ctypes.byref(value), attribute, 0) == 0:
                result[name] = value.value
        if "compute_capability_major" in result:
            result["compute_capability"] = f"{result['compute_capability_major']}.{result['compute_capability_minor']}"
        if result:
            result["cuda_attribute_source"] = "cudaDeviceGetAttribute(logical_device=0)"
        return result
    except (AttributeError, OSError):
        return {}
