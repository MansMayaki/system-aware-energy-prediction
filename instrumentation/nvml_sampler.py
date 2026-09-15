# energy_bert/energy/nvml_sampler.py

import time
from typing import Optional

try:
    import pynvml

    _NVML_OK = True
except Exception:
    _NVML_OK = False


class NVMLPowerSampler:
    def __init__(self, interval_sec: float = 0.1):
        self.interval = interval_sec
        self._running = False
        self._samples = []
        self._handles = []
        self._device_count = 0
        self._thr = None

    def start(self):
        if not _NVML_OK:
            return
        import threading

        try:
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self._device_count)
            ]
            self._running = True

            def _loop():
                while self._running:
                    total = 0.0
                    for h in self._handles:
                        try:
                            total += pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                        except Exception:
                            pass
                    self._samples.append((time.time(), total))
                    time.sleep(self.interval)

            self._thr = threading.Thread(target=_loop, daemon=True)
            self._thr.start()
        except Exception:
            pass

    def stop(self):
        if not _NVML_OK:
            return
        self._running = False
        try:
            if self._thr is not None:
                self._thr.join(timeout=2.0)
        except Exception:
            pass
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def energy_kwh(self) -> Optional[float]:
        if len(self._samples) < 2:
            return None
        e_ws = 0.0
        for i in range(1, len(self._samples)):
            t0, p0 = self._samples[i - 1]
            t1, p1 = self._samples[i]
            e_ws += 0.5 * (p0 + p1) * (t1 - t0)
        return (e_ws / 3600.0) / 1000.0
