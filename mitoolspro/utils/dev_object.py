import json
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Dict, Optional


class Dev:
    _instance: ClassVar[Optional["Dev"]] = None
    _lock: ClassVar[Lock] = Lock()
    variables: Dict[str, Any]

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Dev, cls).__new__(cls)
                cls._instance.variables = {}
            return cls._instance

    def store_var(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        self.variables[key] = value

    def get_var(self, key: str) -> Optional[Any]:
        if key not in self.variables:
            raise KeyError(f"Key '{key}' not found in variables.")
        return self.variables[key]

    def delete_var(self, key: str) -> None:
        if key in self.variables:
            del self.variables[key]
        else:
            raise KeyError(f"Key '{key}' not found in variables.")

    def clear_vars(self) -> None:
        self.variables.clear()

    def save_variables(self, filepath: str) -> None:
        if not self.variables:
            raise ValueError("No variables to store!")
        with open(filepath, "w") as f:
            json.dump(self.variables, f)
        print(f"Variables saved to {filepath}")

    def load_variables(self, filepath: str) -> None:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File '{filepath}' does not exist.")
        with open(filepath, "r") as f:
            self.variables = json.load(f)
        print(f"Variables loaded from {filepath}")

    def __repr__(self) -> str:
        if not self.variables:
            return "Dev()"
        repr_str = "Dev("
        for key, value in self.variables.items():
            repr_str += f"\n    {key}: {value},"
        repr_str = repr_str.rstrip(",")  # Remove trailing comma
        repr_str += "\n)"
        return repr_str

    def __getitem__(self, key: str) -> Any:
        return self.get_var(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.store_var(key, value)

    def __delitem__(self, key: str) -> None:
        self.delete_var(key)

    def __iter__(self):
        return iter(self.variables)

    def __len__(self) -> int:
        return len(self.variables)

    def items(self):
        return self.variables.items()


# Creating a global instance in the module
DEV = Dev()


# Functions for external usage
def store_dev_var(key: str, value: Any) -> None:
    DEV.store_var(key, value)


def get_dev_var(key: str) -> Optional[Any]:
    return DEV.get_var(key)
