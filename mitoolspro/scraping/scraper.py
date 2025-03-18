import json
import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, PriorityQueue
from typing import Any, Dict, Optional, Type

from seleniumbase import SB

from mitoolspro.scraping.scraper_actions import ScraperActions
from mitoolspro.utils.objects import SleepTimer

logger = logging.getLogger("mtp")


class Scraper(ABC):
    def __init__(
        self,
        url: str,
        selectors_file: Path,
        cdp_mode: bool = False,
        headless: bool = True,
        proxy: Optional[str] = None,
        timer_multiplier: float = 1.0,
        activity_timeout: timedelta = timedelta(minutes=2),
    ) -> None:
        self.url = url
        self.sb = None
        self.cdp_mode = cdp_mode
        self.timer_multiplier = timer_multiplier
        self.timer = SleepTimer(multiplier=self.timer_multiplier)
        self.selectors_file = selectors_file
        self.selectors = self._get_selectors(selectors_file)
        self.headless = headless
        self.proxy = proxy
        self.queue = PriorityQueue()
        self.last_activity = datetime.now()
        self.activity_timeout = activity_timeout
        self.current_search = None
        self.last_search = None
        self.running = True
        self.warm_up = True
        self._is_stuck = False
        self.restore = False

    def _get_selectors(self, selectors_file: Path) -> Dict[str, Dict[str, str]]:
        logger.info("Loading selectors")
        with open(selectors_file, "r") as f:
            return json.load(f)

    def _update_activity(self) -> None:
        self.last_activity = datetime.now()

    def is_stuck(self) -> bool:
        return self.queue.qsize() != 0 and (
            self._is_stuck
            or datetime.now() - self.last_activity > self.activity_timeout
        )

    def stop(self) -> None:
        logger.info("Stopping scraper...")
        self.running = False
        try:
            while True:
                self.queue.get_nowait()
                self.queue.task_done()
        except Empty:
            pass

    def add_scrape_request(self, scrape_input: Any) -> None:
        logger.info(f"Adding scrape request: {scrape_input}")
        self.queue.put((2, scrape_input))

    def add_priority_scrape_request(self, scrape_input: Any) -> None:
        logger.info(f"Adding priority scrape request: {scrape_input}")
        self.queue.put((1, scrape_input))

    def _initialize_scraper_actions(self) -> Dict[str, Type["ScraperActions"]]:
        if all(isinstance(selector, dict) for selector in self.selectors.values()):
            return {
                key: ScraperActions(
                    self.sb, selectors_config=selectors, on_action=self._update_activity
                )
                for key, selectors in self.selectors.items()
            }
        else:
            return {
                "actions": ScraperActions(
                    self.sb,
                    selectors_config=self.selectors,
                    on_action=self._update_activity,
                )
            }

    @abstractmethod
    def _handle_scrape(
        self, search_input: str, actions: Dict[str, ScraperActions]
    ) -> bool:
        pass

    @abstractmethod
    def _handle_scrape_error(self, e: Exception, priority: int) -> None:
        pass

    def scrape(self) -> None:
        logger.info("Initializing browser")
        try:
            with SB(
                uc=self.cdp_mode,
                test=False,
                locale="en",
                ad_block=True,
                incognito=True,
                headless=self.headless,
                proxy=self.proxy,
                browser="chrome",
            ) as sb:
                self.sb = sb
                actions = self._initialize_scraper_actions()
                while self.running:
                    try:
                        priority, search_input = self.queue.get(timeout=60)
                        logger.critical(
                            f"Processing search request: {search_input}, PriorityQueue has {self.queue.qsize()} requests left"
                        )
                        self.current_search = search_input

                        if self._handle_scrape(search_input, actions):
                            self.queue.task_done()
                            self.last_search = search_input

                    except Empty:
                        logger.info("No search requests in queue, waiting...")
                        continue
                    except Exception as e:
                        self._handle_scrape_error(e, priority)

        except Exception as e:
            logger.critical(f"Error Found while searching for {self.current_search}")
            logger.error(f"Error: {e}", exc_info=True)
            logger.critical(f"Traceback: {traceback.format_exc()}")
            self._is_stuck = True

    def _capture_debug_info(self) -> None:
        self.debug_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        screenshot_path = f"logs/debug/screenshot_{timestamp}_{self.debug_counter}.png"
        self.sb.save_screenshot(screenshot_path)
        logger.info(f"Saved screenshot to {screenshot_path}")

        html_path = f"logs/debug/page_source_{timestamp}_{self.debug_counter}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self.sb.get_page_source())
        logger.info(f"Saved page source to {html_path}")

        current_url = self.sb.get_current_url()
        logger.info(f"Current URL when stuck: {current_url}")


def main():
    scraper = Scraper(
        url="https://www.google.com",
        selectors_file=Path("selectors.json"),
        cdp_mode=True,
        headless=False,
        timer_multiplier=1.0,
    )

    scraper.scrape()
    scraper.timer.sleep("short")


if __name__ == "__main__":
    main()
