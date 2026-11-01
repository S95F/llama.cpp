#!/usr/bin/env python3
"""EXP-0005: does the CUDA driver accept a host-to-device copy that crosses the
edge of a cudaHostRegister'd range?

Registers one 4 KiB page in the middle of four pageable pages and copies
ranges that start and end inside, before and after it, synchronously and
asynchronously. Prints the cudaError_t of each copy (0 = success, 1 = invalid
value). This is the failure the expert-level pinning hit before uploads were
split at the pinned bounds.

    python cross_copy.py
"""
import ctypes

PAGE = 4096
rt = ctypes.CDLL("C:/Users/steve/cuda126/bin/cudart64_12.dll")
k32 = ctypes.WinDLL("kernel32")
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]

host = k32.VirtualAlloc(None, 4 * PAGE, 0x3000, 0x04)
ctypes.memset(host, 7, 4 * PAGE)
dev = ctypes.c_void_p()
assert rt.cudaMalloc(ctypes.byref(dev), ctypes.c_size_t(4 * PAGE)) == 0
stream = ctypes.c_void_p()
assert rt.cudaStreamCreate(ctypes.byref(stream)) == 0
# cudaHostRegisterPortable | cudaHostRegisterReadOnly, as ggml_backend_cuda_register_host_buffer
err = rt.cudaHostRegister(ctypes.c_void_p(host + PAGE), ctypes.c_size_t(PAGE), 0x01 | 0x08)
print(f"register page 1: {err}")

cases = [
    ("inside the registered page", PAGE, PAGE),
    ("starts in it, ends one byte past it", PAGE, PAGE + 1),
    ("starts in it, runs into the next page", PAGE, 2 * PAGE),
    ("starts 512 B in, runs into the next page", PAGE + 512, PAGE),
    ("starts before it, ends inside it", 0, 2 * PAGE),
    ("starts before it, runs past it", 0, 3 * PAGE),
    ("pageable only", 2 * PAGE, 2 * PAGE),
]
for name, off, n in cases:
    e_sync = rt.cudaMemcpy(dev, ctypes.c_void_p(host + off), ctypes.c_size_t(n), 1)
    rt.cudaGetLastError()
    e_async = rt.cudaMemcpyAsync(dev, ctypes.c_void_p(host + off), ctypes.c_size_t(n), 1, stream)
    rt.cudaGetLastError()
    rt.cudaStreamSynchronize(stream)
    rt.cudaGetLastError()
    print(f"{name:42s} offset {off:5d} size {n:5d}: cudaMemcpy {e_sync}, cudaMemcpyAsync {e_async}")
rt.cudaHostUnregister(ctypes.c_void_p(host + PAGE))
