from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from pandas import DataFrame


@dataclass(frozen=True)
class BaseRegressionStrs(ABC):
    pass


class BaseRegressionSpecs(ABC):
    @abstractmethod
    def get_formula(self) -> str:
        pass

    @abstractmethod
    def get_id(self) -> str:
        pass


class BaseRegressionResult(ABC):
    @abstractmethod
    def get_coefficients(self) -> DataFrame:
        pass

    @abstractmethod
    def get_residuals(self) -> Dict[float, DataFrame]:
        pass

    @abstractmethod
    def get_summaries(self) -> Dict[float, str]:
        pass
