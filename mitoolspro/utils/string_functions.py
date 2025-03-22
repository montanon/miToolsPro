import re
import string
from typing import Iterable, List, Optional, Pattern, Union

from thefuzz import fuzz

PUNCTUATION_REGEX = re.compile(
    "[" + "".join(re.escape(p) for p in string.punctuation) + "]"
)


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


def decode_string(value: Union[str, bytes], encoding: str = "utf-8") -> str:
    if isinstance(encoding, (str, bytes)):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    for e in encoding:
        try:
            return value.decode(*e)
        except Exception:
            pass
    return str(value)


def encode_string(value: Union[str, bytes], encoding: str = "utf-8") -> str:
    if isinstance(encoding, (str, bytes)):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    for e in encoding:
        try:
            return value.encode(*e)
        except Exception:
            pass
    return str(value)


def strip_punctuation(s: str, all: bool = False) -> str:
    if all:
        return PUNCTUATION_REGEX.sub("", s.strip())
    else:
        return s.strip().strip(string.punctuation)


def lowerstrip(s: str, all: bool = False) -> str:
    return strip_punctuation(s.lower().strip(), all=all)


def str_is_number(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


def get_numbers_from_str(string: str, n: Optional[int] = None) -> List:
    pattern = r"(-?\d*\.?\d*(?:[eE][-+]?\d+)?)"
    values = [s for s in re.findall(pattern, string.strip()) if s and s != "-"]
    numbers = [float(s) if s != "." else 0 for s in values]
    return numbers[n] if n else numbers


def remove_multiple_spaces(string: str) -> str:
    return re.sub(r"\s+", " ", string)


def find_str_line_number_in_text(text: str, substring: str) -> int:
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if substring in line:
            return idx


def replace_prefix(string: str, prefix: Pattern, replacement: str) -> str:
    return re.sub(r"^" + re.escape(prefix), replacement, string)


def split_strings(str_list: List[str]) -> List[str]:
    new_list = []
    for s in str_list:
        new_list += re.split("(?=[A-Z])", s)
    return [s for s in new_list if s]


def remove_chars(input_string: str, chars_to_remove: str) -> str:
    remove_set = set(chars_to_remove)
    return "".join(char for char in input_string if char not in remove_set)


def stretch_string(string: str, length: Optional[int] = 60) -> str:
    string = " ".join(string.split())
    if len(string) > length:
        index = length
        while index >= 0 and string[index] != " ":
            index -= 1
        if index >= 0:
            return string[:index] + "\n" + stretch_string(string[index + 1 :], length)
        else:
            return string[:length] + "\n" + stretch_string(string[length:], length)
    else:
        return string


def clean_str(string: str, pattern: Optional[str], sub_char: Optional[str] = "") -> str:
    return re.sub(rf"{pattern}", sub_char, string)


def lcs_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 in s2 or s2 in s1:
        return min(len(s1), len(s2)) / max(len(s1), len(s2))
    len_s1, len_s2 = len(s1), len(s2)
    if len_s1 < len_s2:
        s1, s2, len_s1, len_s2 = s2, s1, len_s2, len_s1
    curr_row = [0] * (len_s2 + 1)
    for i in range(len_s1):
        prev_val = curr_row[0]
        for j in range(len_s2):
            temp = curr_row[j + 1]
            if s1[i] == s2[j]:
                curr_row[j + 1] = prev_val + 1
            else:
                curr_row[j + 1] = max(curr_row[j], curr_row[j + 1])
            prev_val = temp
    lcs_length = curr_row[-1]
    return lcs_length / max(len_s1, len_s2)


def fuzz_string_in_string(
    src_string: str, dst_string: str, threshold: Optional[int] = 90
) -> bool:
    similarity_score = fuzz_ratio(src_string, dst_string)
    return similarity_score > threshold


def fuzz_ratio(src_string: str, dst_string: str) -> float:
    similarity_score = fuzz.partial_ratio(src_string, dst_string)
    return similarity_score
