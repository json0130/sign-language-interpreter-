"""Pepper transports. Speech and LEDs only: every call is checked against ALLOWED.

DryRunTransport records calls (tests, smoke runs). QiTransport talks to a real robot via the
optional `qi` bindings, imported lazily so the rest of the package never needs them.
"""
import threading
import time
from typing import Dict, List, Protocol, Set, Tuple

ALLOWED: Dict[str, Set[str]] = {
    "ALTextToSpeech": {"say", "setLanguage", "setVolume", "getVolume", "getLanguage",
                       "getAvailableLanguages", "getAvailableVoices", "stopAll"},
    "ALLeds": {"fadeRGB", "listGroups"},
    "ALSystem": {"systemVersion", "robotName"},
    "ALAutonomousLife": {"getState"},
}


def check_allowed(service: str, method: str) -> None:
    if method not in ALLOWED.get(service, ()):
        raise PermissionError(f"{service}.{method} is not allowed (speech and LEDs only)")


class Transport(Protocol):
    def call(self, service: str, method: str, *args, post: bool = False): ...


class DryRunTransport:
    """Records (service, method, args); returns canned values for the getters."""
    _CANNED = {
        ("ALTextToSpeech", "getAvailableLanguages"): ["English"],
        ("ALTextToSpeech", "getAvailableVoices"): ["naoenu"],
        ("ALTextToSpeech", "getLanguage"): "English",
        ("ALTextToSpeech", "getVolume"): 0.6,
        ("ALLeds", "listGroups"): ["FaceLeds"],
        ("ALSystem", "systemVersion"): "dry-run",
        ("ALSystem", "robotName"): "dry-run-pepper",
        ("ALAutonomousLife", "getState"): "disabled",
    }

    def __init__(self, say_delay: float = 0.0):
        self.say_delay = say_delay
        self.calls: List[Tuple[str, str, tuple]] = []
        self._lock = threading.Lock()

    def call(self, service: str, method: str, *args, post: bool = False):
        check_allowed(service, method)
        with self._lock:
            self.calls.append((service, method, args))
        if method == "say" and self.say_delay:
            time.sleep(self.say_delay)
        return self._CANNED.get((service, method))


class QiTransport:
    def __init__(self, host: str, port: int = 9559):
        try:
            import qi   # lazy: optional dependency
        except ImportError as e:
            raise RuntimeError(
                "The 'qi' module is not available. Either install the qi Python bindings "
                "(pip install qi, Python 3) or run under the Aldebaran Python 2.7 NAOqi SDK."
            ) from e
        self._session = qi.Session()
        self._session.connect(f"tcp://{host}:{port}")
        self._services: dict = {}

    def call(self, service: str, method: str, *args, post: bool = False):
        check_allowed(service, method)
        if service not in self._services:
            self._services[service] = self._session.service(service)
        fn = getattr(self._services[service], method)
        return fn(*args, _async=True) if post else fn(*args)
