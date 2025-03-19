import json
import pickle
from os import PathLike
from typing import Any, Dict


def read_text(text_path: PathLike) -> str:
    with open(text_path, "r") as f:
        return f.read()


def write_text(text: str, text_path: PathLike) -> None:
    with open(text_path, "w") as f:
        f.write(text)


def read_json(json_path: PathLike) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)


def write_json(
    data: Dict, json_path: PathLike, ensure_ascii: bool = True, encoding: str = "utf-8"
) -> None:
    with open(json_path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=4, ensure_ascii=ensure_ascii)


def store_pkl(obj, filename: PathLike) -> None:
    with open(filename, "wb") as output_file:
        pickle.dump(obj, output_file)


def load_pkl(filename: PathLike) -> Any:
    with open(filename, "rb") as input_file:
        obj = pickle.load(input_file)
        return obj
