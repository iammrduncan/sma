"""The GPU's cumulative energy counter, read from the NVIDIA driver's NVML library.

ctypes and the driver's own libnvidia-ml.so.1, so no extra package. Zeus, AIPerf, and the
ML.ENERGY benchmark all read this counter instead of integrating sampled power. On a DGX Spark
(GB10) it updates more often than 4 times a second and reports GPU energy only, not the whole SoC.
"""

from __future__ import annotations

import ctypes


def energy_counter_mj(gpu_index: int) -> int | None:
    """Energy the GPU has used since the driver loaded, in millijoules. None when NVML cannot say."""
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError:
        return None
    if lib.nvmlInit_v2() != 0:
        return None
    try:
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(gpu_index, ctypes.byref(handle)) != 0:
            return None
        value = ctypes.c_ulonglong()
        if lib.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(value)) != 0:
            return None
        return int(value.value)
    finally:
        lib.nvmlShutdown()
