import sys
from itertools import cycle
from os import PathLike
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import chardet
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

from mitoolspro.exceptions import ArgumentValueError
import logging

logger = logging.getLogger(__name__)

COLOR_CODES = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "reset": "\033[0m",  # Reset to default color
}


def iterable_chunks(
    iterable: Iterable, chunk_size: int
) -> Generator[Iterable, None, None]:
    if not isinstance(iterable, (str, list, tuple, bytes)):
        raise TypeError(
            f"Provided iterable of type {type(iterable).__name__} doesn't support slicing."
        )
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def dict_from_kwargs(**kwargs: Dict[str, Any]) -> Dict:
    return kwargs


def add_significance(row: Series) -> Series:
    p_value = float(row.split(" ")[1].replace("(", "").replace(")", ""))
    if p_value < 0.001:
        return row + "***"
    elif p_value < 0.01:
        return row + "**"
    elif p_value < 0.05:
        return row + "*"
    else:
        return row


def can_convert_to(items: Iterable, type: Type) -> bool:
    try:
        return all(isinstance(type(item), type) for item in items)
    except ValueError:
        return False


def invert_dict(dictionary: Dict) -> Dict:
    return {value: key for key, value in dictionary.items()}


def check_symmetrical_matrix(
    a: ndarray, rtol: Optional[float] = 1e-05, atol: Optional[float] = 1e-08
) -> bool:
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def unpack_list_of_lists(list_of_lists: List[List]) -> List:
    return [item for sub_list in list_of_lists for item in sub_list]


def display_env_variables(
    env_vars: List[Tuple[str, Any]], threshold_mb: float
) -> DataFrame:
    large_vars = []
    for name, value in env_vars:
        size_mb = sys.getsizeof(value) / (1024**2)
        if size_mb > threshold_mb:
            info = f"Type: {type(value).__name__}, ID: {id(value)}"
            if hasattr(value, "__doc__"):
                doc = str(value.__doc__).split("\n")[0]
                info += f", Doc: {doc[:50]}..."
            large_vars.append((name, size_mb, info))
    df = DataFrame(large_vars, columns=["Variable", "Size (MB)", "Info"])
    df.sort_values(by="Size (MB)", ascending=False, inplace=True)
    return df


def sort_dict_keys(
    input_dict: Dict, key: Callable = None, reverse: bool = False
) -> List:
    try:
        sorted_dict = dict(
            sorted(
                input_dict.items(),
                key=key if key else lambda item: item[0],
                reverse=reverse,
            )
        )
        return sorted_dict
    except Exception as e:
        raise ArgumentValueError(f"An error occured shile sorting the dict: {e}")


def get_file_encoding(file: PathLike, fallback: str = "utf-8") -> str:
    try:
        with open(file, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
        encoding = result.get("encoding")
        confidence = result.get("confidence", 0.0)
        if not encoding or confidence < 0.8:
            return fallback
        if encoding.lower() == "ascii":
            return "utf-8"
        return encoding
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file}' was not found.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the file '{file}': {e}")


def all_can_be_ints(items: Sequence) -> bool:
    try:
        return all(int(item) is not None for item in items)
    except (ValueError, TypeError):
        return False


def iprint(
    iterable: Union[Iterable, str], splitter: Optional[str] = "", c: Optional[str] = ""
):
    if not hasattr(iprint, "color_cycler"):
        iprint.color_cycler = cycle(COLOR_CODES.keys() - {"reset"})
    color_code = COLOR_CODES.get(
        c, ""
    )  # Get the ANSI escape code for the specified color
    if c == "cycler":
        color_code = COLOR_CODES[next(iprint.color_cycler)]
    else:
        color_code = COLOR_CODES.get(
            c, ""
        )  # Get the ANSI escape code for the specified color
    if isinstance(iterable, str):
        iterable = [iterable]
    elif not isinstance(iterable, Iterable):
        iterable = [repr(iterable)]
    for item in iterable:
        if splitter:
            logger.info(splitter * 40)
        if color_code:
            logger.info("%s%s%s", color_code, item, COLOR_CODES["reset"])
        else:
            logger.info("%s", item)
        if splitter:
            logger.info(splitter * 40)
