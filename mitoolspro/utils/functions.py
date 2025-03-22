import re
from typing import Generator, Iterable, Optional


def remove_characters_from_string(string: str, characters: Optional[str] = None) -> str:
    if characters is None:
        characters = r'[\\/*?%&:"<>|]'
    return re.sub(characters, "", string)


def remove_characters_from_strings(
    strings: Iterable[str], characters: Optional[str] = None
) -> Iterable[str]:
    if characters is None:
        characters = r'[\\/*?%&:"<>|]'
    return [re.sub(characters, "", string) for string in strings]


def iterable_chunks(
    iterable: Iterable, chunk_size: int
) -> Generator[Iterable, None, None]:
    if not isinstance(iterable, (str, list, tuple, bytes)):
        raise TypeError(
            f"Provided iterable of type {type(iterable).__name__} doesn't support slicing."
        )
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]
