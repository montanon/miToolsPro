import logging
from queue import Queue
from threading import Event, Thread
from typing import Callable, Optional

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import Session, sessionmaker


class DBQueueWriter:
    def __init__(
        self,
        db_path: str,
        base: DeclarativeMeta,
        handlers: Optional[dict[str, Callable]] = None,
    ):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.queue = Queue()
        self._handlers = handlers or {}
        self._stop_event = Event()
        self._thread = None

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = Thread(target=self._process_queue, daemon=True)
            self._thread.start()

    def register_handler(self, op_name: str, handler: Callable):
        self._handlers[op_name] = handler

    def enqueue(self, op_name: str, *args):
        if not self._stop_event.is_set():
            self.queue.put((op_name, args))

    def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                operation = self.queue.get(timeout=0.1)
                if operation is None:
                    break

                op_name, args = operation
                session: Session = self.Session()
                try:
                    handler = self._handlers.get(op_name)
                    if not handler:
                        raise ValueError(f"Unsupported operation: {op_name}")
                    handler(session, *args)
                    session.commit()
                except Exception as e:
                    logging.error(f"Error in DBTaskWriter: {e}")
                    session.rollback()
                finally:
                    session.close()
                    self.queue.task_done()
            except TimeoutError:
                continue

    def close(self):
        self._stop_event.set()
        self.queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def wait_until_done(self):
        self.queue.join()
