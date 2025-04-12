import unicodedata
from dataclasses import dataclass, field
from typing import List, Set, Tuple


def _strip_accents(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


@dataclass
class CategoryMatches:
    name: str
    matches: List[str]
    strip_accents: bool = True
    lowercase: bool = True
    single_matches: Set[str] = field(init=False)
    multi_matches: List[Tuple[str, ...]] = field(init=False)

    def __post_init__(self):
        self.single_matches = set()
        self.multi_matches = []

        for text in self.matches:
            if self.strip_accents:
                text = _strip_accents(text)
            if self.lowercase:
                text = text.lower()

            matches = tuple(text.split())
            if len(matches) == 1:
                self.single_matches.add(matches[0])
            else:
                self.multi_matches.append(matches)
