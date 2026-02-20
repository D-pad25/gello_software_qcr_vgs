#!/usr/bin/env python3

# Created By:   Alec Tutin
# Date:         05 June 2025

import time

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

class DataStatCalculator(ABC):
    def __init__(self, max_updates: int = 100, max_sample_timedelta_multiplier: float = 100.0):
        self.__max_updates: int = max_updates
        self.__max_sample_timedelta_multilpier: float = max_sample_timedelta_multiplier

        self.__data: List[float] = []
        self.__time_last: Optional[float] = None
        self.__time_of_next_expected_sample: Optional[float] = None

        self.__data_mean: float = 0.0
        self.__data_min: float = 0.0
        self.__data_max: float = 0.0
    
    @property
    def data_mean(self) -> float:
        return self.__data_mean
    
    @property
    def data_min(self) -> float:
        return self.__data_min
    
    @property
    def data_max(self) -> float:
        return self.__data_max
    
    @property
    def sample_count(self) -> int:
        return len(self.__data)
    
    @property
    def is_input_stopped(self) -> bool:
        if self.__time_of_next_expected_sample is None:
            return False
        
        return time.monotonic() > self.__time_of_next_expected_sample
    
    def reset(self) -> None:
        self.__data.clear()
        self.__time_last = None
        self.__time_of_next_expected_sample = None

    def append_sample(self, measured_time: Optional[float] = None, / , **data_kwargs) -> None:
        time_this: float = measured_time if measured_time is not None else time.monotonic()

        if self.__time_last is None:
            self.__time_last = time_this

            return
        
        time_delta: float = time_this - self.__time_last

        self.__time_of_next_expected_sample = time_this + time_delta * self.__max_sample_timedelta_multilpier
        self.__time_last = time_this

        self.__data.append(self._calculate_datapoint(time_delta, **data_kwargs))
        
        while len(self.__data) > self.__max_updates:
            self.__data.pop(0)

        data_sum: float = 0.0

        self.__data_min = self.__data[0]
        self.__data_max = self.__data[0]

        for value in self.__data:
            self.__data_min = min(value, self.__data_min)
            self.__data_max = max(value, self.__data_max)
            data_sum += value
        
        self.__data_mean = data_sum / len(self.__data)

        self._on_post_update()
        
    @abstractmethod
    def _calculate_datapoint(self, time_delta: float, data_kwargs: Dict[str, any]) -> float:
        pass

    @abstractmethod
    def _on_post_update(self) -> None:
        pass

    def __str__(self) -> str:
        output: str = f'Min {self.__data_min:.3f} | Max {self.__data_max:.3f} | Mean {self.__data_mean:.3f} | Last {self.sample_count} Samples'

        if self.is_input_stopped:
            output += ' | WARN: Input Stopped!'
        
        return output

class BandwidthCalculator(DataStatCalculator):
    def append_sample(self, data_length: int, measured_time: Optional[float] = None):
        return super().append_sample(measured_time, data_length=data_length)
    
    def _calculate_datapoint(self, time_delta: float, data_length: int) -> float:        
        if time_delta <= 0.0:
            return 0.0
        
        return data_length / time_delta

    def _on_post_update(self) -> None:
        pass

    def __str__(self) -> str:
        bandwidth: float = self.data_mean
        unit: str = 'B/s'

        if bandwidth > 10 ** 9:
            bandwidth = bandwidth / 10 ** 9
            unit = 'GB/s'
        elif bandwidth > 10 ** 6:
            bandwidth = bandwidth / 10 ** 6
            unit = 'MB/s'
        elif bandwidth > 10 ** 3:
            bandwidth = bandwidth / 10 ** 3
            unit = 'KB/s'

        return f'{bandwidth:.3f} {unit}'

class RateCalculator(DataStatCalculator):
    def __init__(self, max_updates: int = 100, max_sample_timedelta_multiplier: float = 10.0):
        super().__init__(max_updates, max_sample_timedelta_multiplier)
        
        self.__rate_hz: float = 0.0

    @property
    def rate_hz(self) -> float:
        return self.__rate_hz
    
    def reset(self) -> None:
        super().reset()

        self.__rate_hz = 0.0
    
    def _calculate_datapoint(self, time_delta: float) -> float:
        return time_delta
    
    def _on_post_update(self) -> None:
        self.__rate_hz = 1.0 / self.data_mean if self.data_mean > 0.0 else 0.0

    def __str__(self) -> str:
        return f'{self.__rate_hz:.3f}Hz | {super().__str__()}'

class TimingBuilder:
    def __init__(self, name: str):
        self.__name: str = name
        self.__timings: List[Tuple[str, float]] = []
        self.__start: float = time.monotonic()

    def add(self, name: str) -> None:
        now: float = time.monotonic()
        self.__timings.append((name, now))

    def __str__(self) -> str:
        result: str = f'{self.__name} Timing:'
        time_last: float = self.__start

        for name, time in self.__timings:
            result += f'\n\t{name}: {int((time - time_last) * 1000.0)}ms'
            time_last = time

        result += f'\n\tTotal: {int((time_last - self.__start) * 1000.0)}ms'

        return result
    
class CumulativeTimer:
    def __init__(self):
        self.__total: float = 0.0
        self.__started: float = 0.0

    @property
    def total(self) -> float:
        return self.__total

    def start(self) -> None:
        self.__started = time.monotonic()

    def stop(self) -> None:
        now: float = time.monotonic()
        self.__total += now - self.__started
