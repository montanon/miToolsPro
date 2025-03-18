import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Tuple, Type, Union

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
