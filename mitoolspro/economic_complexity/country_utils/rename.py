import logging

import pandas as pd

from mitoolspro.economic_complexity.country_utils.country_converter import (
    CountryConverter,
)

coco_logger = logging.getLogger()
coco_logger.setLevel(logging.CRITICAL)

custom_data = pd.DataFrame.from_dict(
    {
        "name_short": ["Bonaire", "Netherlands Antilles", "Serbia"],
        "name_official": [
            "Bonaire, Saint Eustatius and Saba",
            "Netherlands Antilles",
            "Serbia",
        ],
        "regex": ["bonaire", "antilles", "serbia"],
        "ISO3": ["BES", "ANT", "SER"],
        "ISO2": ["a", "b", "c"],
        "continent": ["America", "America", "Europe"],
    }
)

name_converter = CountryConverter(additional_data=custom_data)
