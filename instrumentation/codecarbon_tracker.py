# energy_bert/energy/codecarbon_tracker.py

from typing import Optional
from codecarbon import EmissionsTracker


class EpochCodeCarbon:
    """
    Simple per-epoch CodeCarbon wrapper.
    Each EpochCodeCarbon instance measures from start() to stop() only.
    """

    def __init__(self, **tracker_kwargs):
        self._tracker_kwargs = tracker_kwargs
        self._tracker: Optional[EmissionsTracker] = None

    def start(self):
        self._tracker = EmissionsTracker(**self._tracker_kwargs)
        self._tracker.start()

    def stop(self) -> Optional[float]:
        if self._tracker is None:
            return None
        try:
            self._tracker.flush()
            self._tracker.stop()
            return getattr(self._tracker._total_energy, "kWh", None)
        except Exception:
            return None
        finally:
            self._tracker = None
