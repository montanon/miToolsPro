import logging

import country_converter as coco
import pandas as pd

coco_logger = coco.logging.getLogger()
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

name_converter = coco.CountryConverter(additional_data=custom_data)
