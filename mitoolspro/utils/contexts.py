import logging
import random
import time
from contextlib import ContextDecorator
from functools import wraps
from typing import Any, Callable, Literal, Optional, Tuple, Type, Union

logger = logging.getLogger("mtp")


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    jitter: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.info(f"Attempt {attempt}/{max_attempts} for {func.__name__}")
                    result = func(*args, **kwargs)
                    logger.info(f"{func.__name__} succeeded")
                    return result
                except exceptions as e:
                    last_exception = e
                    logger.error(f"Error in {func.__name__} on attempt {attempt}: {e}")
                    if attempt < max_attempts:
                        delay = delay_seconds * (backoff_factor ** (attempt - 1))
                        if jitter:
                            delay *= random.uniform(0.9, 1.1)
                        time.sleep(delay)
                    else:
                        raise TimeoutError(
                            f"Failed to execute {func.__name__} after {max_attempts} attempts. Last error: {last_exception}"
                        ) from last_exception

        return wrapper

    return decorator


class Timing(ContextDecorator):
    TIME_UNITS = {
        "ns": 1e0,  # Nanoseconds (default internal unit)
        "ms": 1e-6,  # Milliseconds
        "s": 1e-9,  # Seconds
        "m": 1e-9 / 60,  # Minutes
    }

    def __init__(
        self,
        prefix: Optional[str] = "",
        on_exit: Optional[Callable[[float], str]] = None,
        enabled: bool = True,
        unit: Literal["ns", "ms", "s", "m"] = "ms",
    ):
        if unit not in self.TIME_UNITS:
            raise KeyError(
                f"Unit {unit} must be in self.TIME_UNITS.keys(), {list(self.TIME_UNITS.keys())}"
            )
        self.prefix = prefix
        self.on_exit = on_exit
        self.enabled = enabled
        self.unit = unit
        self.unit_conversion = self.TIME_UNITS[unit]
        self.unit_label = unit

    def __enter__(self):
        self.start_time = time.perf_counter_ns()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time_ns = time.perf_counter_ns() - self.start_time
        elapsed_time_converted = elapsed_time_ns * self.unit_conversion
        if self.enabled:
            output = f"{self.prefix}{elapsed_time_converted:.2f} {self.unit_label}"
            if self.on_exit:
                output += self.on_exit(elapsed_time_ns)
            print(output)


def timeit(
    prefix: str = "",
    unit: Literal["ns", "ms", "s", "m"] = "ms",
    enabled: bool = True,
) -> Timing:
    return Timing(prefix=prefix, unit=unit, enabled=enabled)
