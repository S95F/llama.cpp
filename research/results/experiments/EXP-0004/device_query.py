#!/usr/bin/env python3
"""EXP-0004: device attributes that decide what copy/compute overlap is possible at all."""
import ctypes
import json
import sys

rt = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\steve\cuda126\bin\cudart64_12.dll")

# cudaDeviceAttr values from the CUDA 12 runtime headers
ATTRS = {
    "asyncEngineCount": 40,
    "concurrentKernels": 31,
    "tccDriver": 35,
    "pciBusId": 33,
    "pageableMemoryAccess": 88,
    "canUseHostPointerForRegisteredMem": 91,
    "hostRegisterSupported": 99,
    "hostRegisterReadOnlySupported": 113,
    "computeCapabilityMajor": 75,
    "computeCapabilityMinor": 76,
    "multiProcessorCount": 16,
    "memoryClockRate": 36,
    "globalMemoryBusWidth": 37,
}

out = {}
v = ctypes.c_int()
for name, attr in ATTRS.items():
    err = rt.cudaDeviceGetAttribute(ctypes.byref(v), attr, 0)
    out[name] = v.value if err == 0 else f"error {err}"
ver = ctypes.c_int()
rt.cudaRuntimeGetVersion(ctypes.byref(ver))
out["runtimeVersion"] = ver.value
rt.cudaDriverGetVersion(ctypes.byref(ver))
out["driverVersion"] = ver.value
print(json.dumps(out, indent=1))
