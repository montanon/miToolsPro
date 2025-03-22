import json
import pickle
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Union


def read_text(text_path: Union[str, PathLike], encoding: str = "utf-8") -> str:
    text_path = Path(text_path)
    if not text_path.exists():
        raise FileNotFoundError(f"File not found: {text_path}")
    try:
        with open(text_path, "r", encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        raise ValueError(f"Failed to decode file with encoding {encoding}: {text_path}")


def write_text(
    text: str, text_path: Union[str, PathLike], encoding: str = "utf-8"
) -> None:
    text_path = Path(text_path)
    try:
        with open(text_path, "w", encoding=encoding) as f:
            f.write(text)
    except UnicodeEncodeError:
        raise ValueError(f"Failed to encode text with encoding {encoding}")


def read_json(json_path: Union[str, PathLike], encoding: str = "utf-8") -> Dict:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    try:
        with open(json_path, "r", encoding=encoding) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")


def write_json(
    data: Dict,
    json_path: Union[str, PathLike],
    ensure_ascii: bool = True,
    encoding: str = "utf-8",
    indent: Optional[int] = 4,
) -> None:
    json_path = Path(json_path)
    try:
        with open(json_path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    except TypeError as e:
        raise ValueError(f"Data cannot be serialized to JSON: {e}")


def store_pkl(obj: Any, filename: Union[str, PathLike]) -> None:
    filename = Path(filename)
    try:
        with open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)
    except (pickle.PicklingError, TypeError) as e:
        raise ValueError(f"Failed to pickle object: {e}")


def load_pkl(filename: Union[str, PathLike]) -> Any:
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    try:
        with open(filename, "rb") as input_file:
            return pickle.load(input_file)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Failed to unpickle file: {e}")


def read_html(html_path: Union[str, PathLike], encoding: str = "utf-8") -> str:
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"File not found: {html_path}")
    try:
        with open(html_path, "r", encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        raise ValueError(
            f"Failed to decode HTML file with encoding {encoding}: {html_path}"
        )


def write_html(
    html_content: str, html_path: Union[str, PathLike], encoding: str = "utf-8"
) -> None:
    html_path = Path(html_path)
    try:
        with open(html_path, "w", encoding=encoding) as f:
            f.write(html_content)
    except UnicodeEncodeError:
        raise ValueError(f"Failed to encode HTML content with encoding {encoding}")
