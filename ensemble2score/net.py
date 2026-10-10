"""Network guards for model downloads.

Two failure modes on a restricted network, both seen for real:

* Hugging Face is unreachable.  Demucs tries the HF mirror first and retries each
  file five times, so a run stalls for minutes before falling back — unless
  ``HF_HUB_OFFLINE`` is set, which skips the attempt entirely.
* A download URL answers ``HTTP 200`` with an HTML error page.  A naive check
  treats that as success and hands a text file to a checkpoint loader.  The fix is
  to look at the first bytes and refuse anything that is not an archive/weight
  file.

Nothing here downloads anything by itself; it only prevents hangs and lies.
"""

from __future__ import annotations

import contextlib
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = [
    "hf_unreachable_guard",
    "set_offline",
    "is_offline",
    "checkpoint_fetchable",
    "probe_host",
]

_OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
_MAGIC_OK = (b"PK\x03\x04", b"\x1f\x8b", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05")


def set_offline(value: bool = True) -> None:
    """Set/clear the Hugging Face offline environment flags."""
    import os

    for flag in _OFFLINE_FLAGS:
        if value:
            os.environ[flag] = "1"
        else:
            os.environ.pop(flag, None)


def is_offline() -> bool:
    import os

    return any(os.environ.get(flag) == "1" for flag in _OFFLINE_FLAGS)


def probe_host(host: str, *, port: int = 443, timeout: float = 3.0) -> bool:
    """TCP-reachability check.  Cheap, and enough to rule a host out."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@contextlib.contextmanager
def hf_unreachable_guard(*, timeout: float = 3.0):
    """Set HF offline flags for the duration when Hugging Face looks unreachable.

    Safe to nest and safe to call when the network is fine.  If HF responds, the
    environment is left exactly as it was.
    """
    if is_offline():
        yield False
        return

    reachable = probe_host("huggingface.co", timeout=timeout)
    if reachable:
        yield False
        return

    set_offline(True)
    try:
        yield True
    finally:
        set_offline(False)


def checkpoint_fetchable(url: str, *, timeout: float = 8.0) -> tuple[bool, str]:
    """Would fetching ``url`` actually give us a checkpoint?

    Returns ``(ok, reason)``.  A host that answers with an HTML error page is
    reported as a failure with the reason attached, instead of being mistaken for
    a successful download.
    """
    request = Request(url, method="GET", headers={"Range": "bytes=0-1023"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit http(s)
            head = response.read(1024)
            status = getattr(response, "status", 200)
    except HTTPError as exc:
        return False, f"HTTP {exc.code} from {url}"
    except URLError as exc:
        return False, f"cannot reach {url}: {exc.reason}"
    except OSError as exc:
        return False, f"network error for {url}: {exc}"

    if status >= 400:
        return False, f"HTTP {status} from {url}"
    if not head:
        return False, f"empty response from {url}"
    if head.lstrip()[:1] in (b"<", b"{") and not head.startswith(_MAGIC_OK):
        return False, f"{url} returned a document, not a checkpoint (first bytes: {head[:32]!r})"
    return True, "ok"
