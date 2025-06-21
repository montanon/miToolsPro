import hashlib
from typing import List, Optional

from mitoolspro.utils.objects import StringMapper


def prettify_index_level(
    mapper: StringMapper,
    pattern: str,
    level: str,
    level_name: str,
    levels_to_remap: List[str],
) -> str:
    if level_name in levels_to_remap:
        return level.map(lambda x: prettify_with_pattern(x, mapper, pattern))
    return level


def prettify_with_pattern(string: str, mapper: StringMapper, pattern: str) -> str:
    base_string, pattern_str, _ = string.partition(pattern)
    remapped_base = mapper.prettify_str(base_string)
    return f"{remapped_base}{pattern}" if pattern_str else remapped_base


def create_regression_id(
    regression_type: str,
    regression_degree: str,
    regression_dependent_var: str,
    regression_indep_vars: List[str],
    control_variables: List[str],
    id_len: Optional[int] = 6,
) -> str:
    str_to_hash = " ".join(
        [
            regression_type,
            regression_degree if regression_degree else "linear",
        ]
    )
    id_hasher = hashlib.md5()
    id_hasher.update(rf"{str_to_hash}".encode("utf-8"))
    kind_id = id_hasher.hexdigest()[:id_len]

    id_hasher = hashlib.md5()
    id_hasher.update(rf"{regression_dependent_var}".encode("utf-8"))
    dep_id = id_hasher.hexdigest()[:id_len]

    str_to_hash = " ".join([v for v in regression_indep_vars if "_square" not in v])
    id_hasher = hashlib.md5()
    id_hasher.update(rf"{str_to_hash}".encode("utf-8"))
    indep_id = id_hasher.hexdigest()[:id_len]

    control_vars_str = " ".join([v for v in control_variables])
    id_hasher = hashlib.md5()
    id_hasher.update(rf"{control_vars_str}".encode("utf-8"))
    control_vars_id = id_hasher.hexdigest()[:id_len] if control_variables else "None"
    return f"{kind_id}-{dep_id}-{indep_id}-{control_vars_id}"


def regex_symbol_replacement(match):
    return rf"\{match.group(0)}"
