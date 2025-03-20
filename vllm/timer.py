import time

from torch import nan

class Timer:
    _timers = {}

    @classmethod
    def start(cls, name):
        if name not in cls._timers:
            cls._timers[name] = {'start_time': None, 'total': 0.0}
        cls._timers[name]['start_time'] = time.perf_counter()

    @classmethod
    def end(cls, name):
        if name not in cls._timers or cls._timers[name]['start_time'] is None:
            raise RuntimeError(f"Timer '{name}' is not running.")
        end_time = time.perf_counter()
        cls._timers[name]['total'] += end_time - cls._timers[name]['start_time']
        cls._timers[name]['start_time'] = None

    @classmethod
    def get_time(cls, name):
        return cls._timers.get(name, {'total': nan})['total']