"""Graceful shutdown coordination."""

import signal
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType
from typing import Callable, Iterator, Optional


class GracefulShutdown(Exception):
    """Raised to stop benchmark work while preserving completed results."""


@dataclass
class ShutdownState:
    """Signal-aware shutdown token shared across benchmark layers."""

    requested: bool = False
    signum: Optional[int] = None

    def request(self, signum: Optional[int] = None) -> None:
        """Mark shutdown as requested; a second request interrupts immediately."""
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        self.signum = signum
        raise GracefulShutdown

    def is_requested(self) -> bool:
        """Return whether graceful shutdown has been requested."""
        return self.requested


@contextmanager
def install_signal_handlers() -> Iterator[ShutdownState]:
    """Install SIGINT/SIGTERM handlers for graceful benchmark shutdown."""
    state = ShutdownState()
    handled_signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        handled_signals.append(signal.SIGTERM)

    previous_handlers: dict[int, Callable | int | None] = {}

    def _handler(signum: int, frame: Optional[FrameType]) -> None:
        print("\n⚠️  Shutdown requested. Finishing cleanup and saving partial results...")
        state.request(signum)

    for signum in handled_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handler)

    try:
        yield state
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
