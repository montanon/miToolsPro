import logging
import random
import time
from queue import Empty, PriorityQueue
from threading import Lock, Thread
from typing import Any, Dict, List, Tuple

from mitoolspro.scraping.scraper import Scraper

logger = logging.getLogger("mtp")


class MultiScraper:
    def __init__(
        self,
        num_scrapers: int = 4,
        headless: bool = True,
        timer_multiplier: float = 1.0,
        scraper_class: Scraper = Scraper,
        scraper_kwargs: Dict[str, Any] = {},
    ) -> None:
        self.num_scrapers = num_scrapers
        self.scraper_class = scraper_class
        self.headless = headless
        self.timer_multiplier = timer_multiplier
        self.scraper_kwargs = scraper_kwargs
        # Override the headless and timer_multiplier kwargs
        self.scraper_kwargs["headless"] = self.headless
        self.scraper_kwargs["timer_multiplier"] = self.timer_multiplier
        self.queue = PriorityQueue()
        self.scrapers: List[Scraper] = []
        self.scraper_threads: List[Thread] = []
        self._next_scraper = 0
        self.running = True
        self.proxies_thread: Dict[Thread, str] = {}
        self.lock = Lock()
        self.health_check_thread = Thread(target=self._monitor_threads, daemon=True)
        self.health_check_thread.start()

    def _create_scraper_thread(self) -> Tuple[Scraper, Thread]:
        logger.info("Creating new scraper thread...")
        try:
            scraper = self.scraper_class(**self.scraper_kwargs)
            thread = Thread(target=scraper.scrape, daemon=True)
            logger.info("Scraper thread created successfully")
            return scraper, thread
        except Exception as e:
            logger.error(f"Error creating scraper thread: {str(e)}", exc_info=True)
            raise

    def initialize_scrapers(self) -> None:
        logger.info(f"Initializing {self.num_scrapers} scrapers...")
        for i in range(self.num_scrapers):
            try:
                logger.info(f"Creating scraper {i}...")
                scraper, thread = self._create_scraper_thread()
                self.scrapers.append(scraper)
                self.scraper_threads.append(thread)
                logger.info(f"Starting thread {i}...")
                thread.start()
                logger.info(f"Scraper {i} initialized successfully")
                time.sleep(random.uniform(0.5, 2.5))
            except Exception as e:
                logger.error(f"Error initializing scraper {i}: {str(e)}", exc_info=True)
                raise
        logger.info("All scrapers initialized successfully")

    def _get_next_scraper(self) -> Scraper:
        with self.lock:
            scraper = self.scrapers[self._next_scraper]
            self._next_scraper = (self._next_scraper + 1) % len(self.scrapers)
            return scraper

    def add_scrape_request(self, scrape_request: str) -> None:
        logger.info(f"Adding scrape request: {scrape_request}")
        try:
            scraper = self._get_next_scraper()
            scraper.add_scrape_request(scrape_request)
            logger.info("Scrape request added successfully")
        except Exception as e:
            logger.error(f"Error adding scrape request: {str(e)}", exc_info=True)
            raise

    def add_priority_scrape_request(self, scrape_request: str) -> None:
        logger.info(f"Adding priority scrape request: {scrape_request}")
        try:
            scraper = self._get_next_scraper()
            scraper.add_priority_scrape_request(scrape_request)
            logger.info("Priority scrape request added successfully")
        except Exception as e:
            logger.error(
                f"Error adding priority scrape request: {str(e)}", exc_info=True
            )
            raise

    def _monitor_threads(self) -> None:
        logger.info("Starting thread monitor...")
        while self.running:
            with self.lock:
                for i, (scraper, thread) in enumerate(
                    zip(self.scrapers, self.scraper_threads)
                ):
                    try:
                        if not thread.is_alive():
                            logger.error(f"Thread {i} is not alive!")
                        if scraper.is_stuck():
                            logger.warning(
                                f"Scraper {i} is stuck. Queue size: {scraper.queue.qsize()}"
                            )
                            if thread.is_alive():
                                logger.warning(f"Restarting stuck scraper {i}...")

                                remaining_items = []
                                try:
                                    while True:
                                        remaining_items.append(
                                            scraper.queue.get_nowait()
                                        )
                                except Empty:
                                    pass

                                logger.info(f"Stopping stuck scraper {i}...")
                                scraper.stop()
                                thread.join(timeout=5)
                                if thread.is_alive():
                                    logger.error(
                                        f"Failed to stop scraper {i} gracefully"
                                    )

                                logger.info(f"Creating new scraper to replace {i}...")
                                new_scraper, new_thread = self._create_scraper_thread()

                                for item in remaining_items:
                                    new_scraper.queue.put(item)

                                self.scrapers[i] = new_scraper
                                self.scraper_threads[i] = new_thread
                                new_thread.start()
                                logger.info(
                                    f"Restarted scraper {i} with {len(remaining_items)} remaining items"
                                )
                    except Exception as e:
                        logger.error(
                            f"Error in thread monitor for scraper {i}: {str(e)}",
                            exc_info=True,
                        )
            time.sleep(10)

    def wait_for_completion(self) -> None:
        for scraper in self.scrapers:
            scraper.queue.join()

    def stop(self) -> None:
        logger.info("Stopping multi scraper...")
        self.running = False
        if self.health_check_thread and self.health_check_thread.is_alive():
            try:
                self.health_check_thread.join(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping health check thread: {e}")
        for i, scraper in enumerate(self.scrapers):
            try:
                logger.info(f"Stopping scraper {i}...")
                scraper.stop()
            except Exception as e:
                logger.error(f"Error stopping scraper {i}: {e}")
        for i, thread in enumerate(self.scraper_threads):
            try:
                logger.info(f"Joining thread {i}...")
                thread.join(timeout=10)
                if thread.is_alive():
                    logger.warning(f"Thread {i} did not stop gracefully")
            except Exception as e:
                logger.error(f"Error joining thread {i}: {e}")
        self.scrapers.clear()
        self.scraper_threads.clear()
        logger.info("Multi scraper stopped")
