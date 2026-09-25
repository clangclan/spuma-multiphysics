#!/usr/bin/env python3
"""High-rate NVML polling; polling rate is NOT the sensor's update rate."""
import ctypes as C
import json
import threading
import time

import benchmark as common


class PowerSampler:
    def __init__(self, case, interval=0.02):
        if interval <= 0:
            raise ValueError('Power polling interval must be positive')
        self.case, self.interval = case, interval
        self.stop = threading.Event()
        self.errors = {}
        self.samples = 0
        self.thread = threading.Thread(target=self.collect, name='nvml-power', daemon=True)
        self.thread.start()

    def collect(self):
        lib = None
        initialized = False
        try:
            lib = C.CDLL('libnvidia-ml.so.1')
            if lib.nvmlInit_v2() != 0:
                raise RuntimeError('NVML initialization failed')
            initialized = True
            device = C.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(0, C.byref(device)) != 0:
                raise RuntimeError('NVML device lookup failed')
            with (self.case/'gpu-power-20ms.jsonl').open('x') as stream:
                deadline = time.monotonic()
                while not self.stop.is_set():
                    row = dict(unixNs=time.time_ns(), monotonicNs=time.monotonic_ns())
                    for field, name, kind in (
                        ('powerMilliwatts', 'nvmlDeviceGetPowerUsage', C.c_uint),
                        ('totalEnergyMillijoules', 'nvmlDeviceGetTotalEnergyConsumption', C.c_ulonglong),
                    ):
                        value = kind()
                        try:
                            rc = getattr(lib, name)(device, C.byref(value))
                            row[field] = value.value if rc == 0 else None
                            if rc:
                                self.errors[name] = rc
                        except AttributeError:
                            row[field] = None
                            self.errors[name] = 'symbol unavailable'
                    row['queryEndMonotonicNs'] = time.monotonic_ns()
                    stream.write(json.dumps(row, separators=(',', ':'))+'\n')
                    self.samples += 1
                    if self.samples % 50 == 0:
                        stream.flush()
                    deadline += self.interval
                    now = time.monotonic()
                    if deadline < now:
                        deadline = now
                    self.stop.wait(max(0, deadline-now))
        except Exception as exc:
            self.errors['collector'] = repr(exc)
        finally:
            if initialized:
                lib.nvmlShutdown()

    def close(self):
        self.stop.set()
        self.thread.join()
        common.atomic_json(self.case/'gpu-power-metadata.json', dict(
            samples=self.samples, requestedPollSeconds=self.interval, errors=self.errors,
            scope='Device-wide board power, including any other GPU processes.',
            powerMeaning='nvmlDeviceGetPowerUsage reports a 1-second average on this RTX 5080; 20 ms polling is not 20 ms sensor resolution.',
            energyMeaning='Raw NVML cumulative energy is retained separately; do not assume agreement with integrated board power.',
            cpuEnergy='Not collected: RAPL energy_uj is not readable by this user.',
            timestampMeaning='Monotonic/unix timestamps immediately before each NVML query; query end is also recorded.'))
