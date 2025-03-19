import re


def remove_characters_from_string(string: str, characters: str = None) -> str:
    if characters is None:
        characters = r'[\\/*?%&:"<>|]'
    return re.sub(characters, "", string)
