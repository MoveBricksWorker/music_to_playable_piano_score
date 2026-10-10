"""Per-stem transcription by calling SheetSage2 in its own interpreter.

Why this is a subprocess and not an import
------------------------------------------
SheetSage2's released remote code requires ``transformers 4.45``: its model
constructor passes ``embed_tokens=`` to ``BartDecoder`` and its decode loop calls the
decoder with ``tgt=``/``memory=``.  A modern ``transformers`` accepts neither, so
the model cannot be loaded in this project's own interpreter without reimplementing
its decode path.  The pinned version is therefore installed in a separate
environment (``.venv-ss2``) and this backend shells out to
:file:`scripts/sheetsage2_stem.py`, which prints one JSON object of notes.

That keeps the dependency out of the core install: ``pip install -e .`` neither
needs nor notices SheetSage2, and a machine without it simply cannot select this
backend.

Licensing
---------
SheetSage2 and its MERT-v2 parent are **CC-BY-NC-4.0 (non-commercial)**, so this is
never chosen by ``auto`` -- see ``io/registry.py``.  Naming it explicitly is the
only way to get here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import BackendUnavailable
from ..core.models import Note, PartSpec
from ..io.console import hint, warn

__all__ = [
    "SheetSage2Transcriber",
    "find_ss2_python",
    "repo_root",
    "worker_path",
    "default_hf_home",
    "SS2_PYTHON_CANDIDATES",
]

# Where the pinned environment may live.  A venv beside the checkout is what
# `python -m venv .venv-ss2` produces; the rest are conveniences for machines where
# it was installed elsewhere.
SS2_PYTHON_CANDIDATES = (
    ".venv-ss2/Scripts/python.exe",   # Windows
    ".venv-ss2/bin/python",           # POSIX
    ".venv-sheetsage2/Scripts/python.exe",
    ".venv-sheetsage2/bin/python",
)

WORKER = "sheetsage2_stem.py"


def repo_root() -> Path:
    """The checkout or install prefix that holds ``scripts/`` and the venv.

    ``__file__`` is ``<root>/src/ensemble2score/analysis/sheetsage2_backend.py``, so
    the root is three levels up.  Every path in this module is derived from here
    rather than from ``Path.cwd()``: a run started from another directory would
    otherwise look for the environment in the wrong place and report it missing.
    """
    return Path(__file__).resolve().parents[3]


def worker_path() -> Path:
    """Path to the worker script, whether run from a checkout or an install."""
    candidate = repo_root() / "scripts" / WORKER
    if candidate.is_file():
        return candidate
    raise BackendUnavailable(
        f"SheetSage2 worker not found at {candidate}: this backend needs a source "
        "checkout, not only an installed package"
    )


def find_ss2_python(explicit: str | None = None) -> str | None:
    """Locate the interpreter that can import SheetSage2.

    An explicit path wins.  Otherwise the repo's own venv candidates are tried, then
    a suffixed ``python-ss2`` on ``PATH``.  Returns ``None`` when nothing looks
    usable, so the caller can say what to install rather than failing somewhere
    inside a subprocess.
    """
    if explicit:
        return explicit if Path(explicit).is_file() else None

    root = repo_root()
    for relative in SS2_PYTHON_CANDIDATES:
        candidate = root / relative
        if candidate.is_file():
            return str(candidate)

    for name in ("python-ss2", "python3-ss2"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _register_of(spec: PartSpec) -> str:
    """The MIDI span this instrument may occupy, as ``"low..high"``.

    Read from the same :data:`~ensemble2score.analysis.transcribe.PLAUSIBLE_RANGE`
    table the pipeline filters with, so the worker's voice choice can never be more
    permissive than the filter that follows it.
    """
    from .transcribe import PLAUSIBLE_RANGE

    key = spec.stem if spec.stem in PLAUSIBLE_RANGE else spec.instrument.lower()
    low, high = PLAUSIBLE_RANGE.get(key, PLAUSIBLE_RANGE["other"])
    shift = int(spec.sounding_pitch or 0)
    return f"{low + shift}..{high + shift}"


def _manifest_path() -> str:
    """A temp file for the batch manifest.

    Held in the system temp area rather than the output directory: it describes work
    in flight, not a result, and must not be mistaken for one or survive the run.
    """
    import tempfile

    handle = tempfile.NamedTemporaryFile(
        prefix="e2s_ss2_batch_", suffix=".json", delete=False
    )
    handle.close()
    return handle.name


@dataclass
class SheetSage2Transcriber:
    """Transcribe one stem with SheetSage2, in the pinned environment.

    ``python`` is the interpreter of the environment that holds
    ``transformers==4.45`` plus SheetSage2's dependencies.  ``hf_home`` is passed
    through so the ~5 GB of SheetSage2/MERT weights are found without re-downloading
    them, and so the cache writes somewhere the sandbox allows.
    """

    name: str = "sheetsage2"
    python: str | None = None
    device: str = "auto"
    hf_home: str | None = None
    timeout: float = 3600.0
    _resolved: str | None = None

    def __post_init__(self) -> None:
        self._resolved = find_ss2_python(self.python)
        if self._resolved is None:
            raise BackendUnavailable(
                "SheetSage2 needs its own environment (transformers==4.45) and none "
                "was found. Create one with:\n"
                "  python -m venv --system-site-packages .venv-ss2\n"
                "  .venv-ss2/Scripts/python.exe -m pip install \"transformers==4.45.2\" "
                "\"tokenizers<0.21\" timm\n"
                "or point --ss2-python at an existing interpreter."
            )

    # ------------------------------------------------------------------ #
    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        if self.hf_home:
            environment["HF_HOME"] = str(self.hf_home)
        environment.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        environment.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        environment.setdefault("TRANSFORMERS_VERBOSITY", "error")
        # Keep the worker's own stdout/stderr from being interleaved by a parent
        # progress bar, and never let a stray tokeniser thread stall the run.
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        return environment

    def transcribe(self, audio_path: str, spec: PartSpec) -> tuple[list[Note], list[str]]:
        """``audio_path`` -> performance-level notes on the same timeline."""
        results = self.transcribe_many([(spec.stem, audio_path, spec)])
        notes, messages = results[spec.stem]
        return notes, messages

    def transcribe_many(
        self, jobs: list[tuple[str, str, PartSpec]]
    ) -> dict[str, tuple[list[Note], list[str]]]:
        """Transcribe several stems in **one** worker process.

        Loading SheetSage2 costs 15-30 s while transcribing costs seconds, so a
        process per stem pays the load once per stem.  Measured on a 40 s excerpt the
        per-stem process spent 31 s to transcribe 9.9 s of audio; batching removes
        that repetition for every stem after the first.

        A stem that fails is reported as such instead of failing the batch, because
        the pipeline caches per stem and losing one should not cost the others.
        """
        if not jobs:
            return {}

        worker = worker_path()
        manifest = [
            {"key": stem, "audio": str(path), "register": _register_of(spec)}
            for stem, path, spec in jobs
        ]
        manifest_path = Path(_manifest_path())
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        command = [
            str(self._resolved),
            "-u",
            str(worker),
            "--batch",
            str(manifest_path),
            "--device",
            self.device,
        ]
        started = time.time()
        try:
            completed = subprocess.run(  # noqa: S603 - the interpreter is resolved above
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                # A batch is bounded by the audio it carries, so the ceiling scales
                # with the work rather than being a fixed per-stem timeout.
                timeout=self.timeout * max(1, len(jobs)),
                env=self._environment(),
                cwd=str(worker.parent.parent),
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendUnavailable(
                f"SheetSage2 timed out on a batch of {len(jobs)} stem(s)"
            ) from exc
        except OSError as exc:
            raise BackendUnavailable(f"could not run the SheetSage2 interpreter: {exc}") from exc
        finally:
            manifest_path.unlink(missing_ok=True)

        payload = self._parse(completed.stdout)
        if payload is None or "results" not in payload:
            detail = (completed.stderr or "").strip().splitlines()
            tail = " | ".join(detail[-4:]) if detail else "no output"
            if payload and payload.get("error"):
                tail = str(payload["error"])
            raise BackendUnavailable(
                f"SheetSage2 produced no readable batch result (exit "
                f"{completed.returncode}): {tail}"
            )

        elapsed = time.time() - started
        out: dict[str, tuple[list[Note], list[str]]] = {}
        for stem, _path, spec in jobs:
            if stem not in payload["results"]:
                # A worker that answers 0 notes for a stem it was not asked about is
                # reporting a partial result; say so rather than passing it off as a
                # silent stem, which would look identical downstream.
                out[stem] = (
                    [],
                    [f"SheetSage2 returned no entry for stem {stem!r}: nothing was "
                     "transcribed for it in this pass"],
                )
                continue
            entry = payload["results"].get(stem) or {}
            if entry.get("error"):
                out[stem] = ([], [f"SheetSage2 failed on this stem: {entry['error']}"])
                continue
            out[stem] = self._to_notes(entry, spec)
        shared = (
            f"SheetSage2: {len(jobs)} stem(s) in one pass, "
            f"{elapsed:.0f}s total on {payload.get('device', self.device)} "
            f"(the model is loaded once per pass)"
        )
        for stem, (_notes, messages) in out.items():
            messages.insert(0, shared)
        return out

    def _to_notes(self, payload: dict, spec: PartSpec) -> tuple[list[Note], list[str]]:
        notes = [
            Note(
                pitch=int(item["pitch"]),
                start=float(item["start"]),
                end=float(item["end"]),
                velocity=int(item.get("velocity", 80)),
                # SheetSage2 reports no per-note confidence, so every note it emits
                # is marked fully trusted.  That is the honest reading: the model
                # either produced the note or it did not, and there is no score to
                # threshold.  `--min-confidence` therefore has no effect on this
                # backend, which the run reports rather than leaving to be guessed.
                confidence=1.0,
            )
            for item in payload.get("notes", [])
        ]

        messages: list[str] = []
        voices = ", ".join(payload.get("voices_used") or []) or "none"
        messages.append(
            f"SheetSage2: {len(notes)} note(s) from the {voices} voice "
            f"(duty cycle {payload.get('duty_cycle', '?')}; its melody voice is a line, "
            "its chord voice is a stack)"
        )
        if payload.get("seconds"):
            messages.append(f"SheetSage2: this stem took {payload['seconds']}s")
        if payload.get("key"):
            messages.append(f"SheetSage2's own key reading for this stem: {payload['key']}")
        if payload.get("meter"):
            messages.append(f"SheetSage2's own meter reading for this stem: {payload['meter']}")
        messages.append(
            "SheetSage2 reports no per-note confidence: --min-confidence does not apply "
            "to this backend"
        )
        for message in payload.get("warnings") or []:
            messages.append(f"SheetSage2: {message}")

        if not notes:
            warn(
                f"{spec.stem}: SheetSage2 returned no notes. A silent stem yields an "
                "empty line rather than an invented one"
            )
        return notes, messages

    @staticmethod
    def _parse(stdout: str) -> dict | None:
        """Pull the JSON object out of stdout.

        The model prints progress to stderr, but a library may still write to stdout,
        so the JSON is taken from the last line that parses rather than assumed to be
        the whole stream.
        """
        text = (stdout or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None


def default_hf_home() -> str | None:
    """Where the SheetSage2 weights live, if anything is there.

    The workspace-local cache is preferred because a sandboxed run may not be able to
    write to ``~/.cache``; the user-level cache is the fallback for a normal install.
    """
    repo = Path(__file__).resolve().parents[3]
    for candidate in (repo / "_hf_home", repo / "hf_home", Path.home() / ".cache" / "huggingface"):
        if (candidate / "hub").is_dir() or (candidate / "models").is_dir():
            return str(candidate)
    return None


def license_note() -> str:
    return (
        "SheetSage2 and its MERT-v2 parent are CC-BY-NC-4.0 (non-commercial). This "
        "backend is never chosen by --transcriber auto; naming it is an explicit "
        "choice and the output must not be used commercially."
    )


if __name__ == "__main__":  # pragma: no cover - manual probe
    print(f"interpreter: {find_ss2_python()}")
    print(f"worker:      {worker_path() if find_ss2_python() else '-'}")
    print(f"hf home:     {default_hf_home()}")
    print(f"core:        {sys.version.split()[0]}")
    hint(license_note())
