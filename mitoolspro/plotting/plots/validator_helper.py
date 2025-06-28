import inspect
from typing import Any, Iterable, Tuple

from pydantic import ValidationError


def _call_validator(validator, value, **kwargs):
    """Call a validator class or callable with filtered kwargs."""
    try:
        sig = inspect.signature(validator)
    except (ValueError, TypeError):
        # For classes get __init__ signature
        sig = inspect.signature(getattr(validator, "__init__"))
    params = {"value": value}
    for name in ("sizes", "sub_sizes", "structured"):
        if name in sig.parameters and name in kwargs:
            params[name] = kwargs[name]
    for k, v in kwargs.get("extra", {}).items():
        if k in sig.parameters:
            params[k] = v
    result = validator(**params)
    return getattr(result, "value", result)


def apply_validators(
    value: Any,
    validators: Iterable,
    sizes: Any | None = None,
    sub_sizes: Any | None = None,
    structured: bool | None = True,
    **extra,
) -> Tuple[Any | None, list[str]]:
    """Iterate validators and return first successful value and collected errors."""
    errors: list[str] = []
    for validator in validators:
        try:
            validated = _call_validator(
                validator,
                value,
                sizes=sizes,
                sub_sizes=sub_sizes,
                structured=structured,
                extra=extra,
            )
            return validated, errors
        except ValidationError as e:
            errors.append(str(e))
        except Exception as e:  # pragma: no cover - unexpected errors
            errors.append(str(e))
    return None, errors
