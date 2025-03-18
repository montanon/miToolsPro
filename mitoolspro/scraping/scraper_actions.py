import logging
from typing import Callable, Dict, List, Optional

from selenium.webdriver.remote.webelement import WebElement
from seleniumbase import SB

from mitoolspro.utils.contexts import retry

logger = logging.getLogger("mtp")


class ScraperActions:
    def __init__(
        self,
        sb: SB,
        selectors_config: Dict[str, str],
        on_action: Optional[Callable] = None,
    ) -> None:
        self.sb = sb
        self.selectors = selectors_config
        self.on_action = on_action

    def _on_action(self) -> None:
        if self.on_action:
            self.on_action()

    @retry(max_attempts=3, delay_seconds=2)
    def click(self, key: str, formatting: Optional[str] = None) -> None:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Clicking on '{key}' with selector: {selector}")
        self.sb.click(selector)
        self._on_action()

    @retry(max_attempts=3, delay_seconds=2)
    def click_if_visible(self, key: str, formatting: Optional[str] = None) -> None:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Clicking if visible on '{key}' with selector: {selector}")
        self.sb.click_if_visible(selector)
        self._on_action()

    @retry(max_attempts=3, delay_seconds=2)
    def press_keys(self, key: str, keys: str, formatting: Optional[str] = None) -> None:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Pressing keys on '{key}' with selector: {selector}")
        self.sb.cdp.press_keys(selector, keys)
        self._on_action()

    @retry(max_attempts=3, delay_seconds=2)
    def find_elements_by_text(
        self, key: str, formatting: Optional[str] = None
    ) -> List[any]:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Finding elements by text: {selector}")
        elements = self.sb.cdp.find_elements_by_text(selector)
        self._on_action()
        return elements

    @retry(max_attempts=3, delay_seconds=2)
    def find_element_by_text(
        self, key: str, formatting: Optional[str] = None
    ) -> WebElement:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Finding elements by text: {selector}")
        element = self.sb.cdp.find_element_by_text(selector)
        self._on_action()
        return element

    @retry(max_attempts=3, delay_seconds=2)
    def find_element(
        self, key: str, best_match: bool = True, formatting: Optional[str] = None
    ) -> WebElement:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Finding element: {selector}")
        element = self.sb.cdp.find_element(selector, best_match=best_match)
        self._on_action()
        return element

    @retry(max_attempts=3, delay_seconds=2)
    def find_elements(
        self, key: str, formatting: Optional[str] = None
    ) -> List[WebElement]:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Finding elements: {selector}")
        elements = self.sb.cdp.find_elements(selector)
        self._on_action()
        return elements

    @retry(max_attempts=3, delay_seconds=2)
    def click_element(self, element: WebElement) -> None:
        logger.info(f"Clicking element: {element}")
        element.click()
        self._on_action()

    @retry(max_attempts=3, delay_seconds=2)
    def get_attribute(
        self, key: str, attribute: str, formatting: Optional[str] = None
    ) -> str:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Getting attribute: {selector}")
        element = self.sb.cdp.get_attribute(selector, attribute)
        self._on_action()
        return element

    @retry(max_attempts=3, delay_seconds=2)
    def select_option_by_text(
        self, key: str, option: str, formatting: Optional[str] = None
    ) -> None:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Selecting option by text: {selector}")
        self.sb.cdp.select_option_by_text(selector, option)
        self._on_action()

    @retry(max_attempts=3, delay_seconds=2)
    def type(
        self, key: str, text: str | int | float, formatting: Optional[str] = None
    ) -> None:
        text = str(text)
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Typing text: {selector}")
        self.sb.cdp.type(selector, text)
        self._on_action()

    def is_element_visible(self, key: str, formatting: Optional[str] = None) -> bool:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        logger.info(f"Checking if element is visible: {selector}")
        return self.sb.cdp.is_element_visible(selector)

    def wait_for_element_visible(self, key: str, timeout_seconds: int = 7) -> None:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        logger.info(f"Waiting for element to be visible: {selector}")
        self.sb.cdp.wait_for_element_visible(selector, timeout=timeout_seconds)
        self._on_action()

    def find_element_with_timeout(
        self, key: str, timeout_seconds: float = 3.5, formatting: Optional[str] = None
    ) -> WebElement:
        selector = self.selectors.get(key)
        if not selector:
            raise ValueError(f"No selector defined for key: {key}")
        if formatting:
            selector = selector.format(formatting)
        element = self.sb.cdp.find_element(selector, timeout=timeout_seconds)
        if element:
            logger.info(f"Found element {key}")
            self._on_action()
            return element

        raise TimeoutError(f"Element {key} not found after {timeout_seconds} seconds")
