import logging
import random
import time
from typing import Dict, Literal, Optional, Tuple, Union

logger = logging.getLogger("mtp")


class SleepTimer:
    presets: Dict[str, Tuple[float, float]]
    multiplier: float

    def __init__(
        self,
        presets: Optional[Dict[str, Tuple[float, float]]] = None,
        multiplier: float = 1.0,
    ) -> None:
        self.presets = presets or {
            "very_short": (0.05, 0.15),
            "short": (0.15, 0.5),
            "medium": (0.75, 2.0),
            "long": (2.0, 2.5),
            "very_long": (2.5, 3.0),
            "extremely_long": (8.0, 10.0),
        }
        self.multiplier = multiplier

    def sleep(
        self,
        category: Union[
            Literal[
                "very_short", "short", "medium", "long", "very_long", "extremely_long"
            ],
            int,
        ],
    ) -> None:
        if isinstance(category, int):
            duration = category
        else:
            if category not in self.presets:
                raise ValueError(
                    f"Invalid category '{category}'. Available categories: {list(self.presets.keys())}"
                )
            duration = random.uniform(*self.presets[category])
        time.sleep(duration * self.multiplier)
