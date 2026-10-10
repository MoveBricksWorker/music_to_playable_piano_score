"""Standalone SheetSage2 inference: audio in, ``**kern`` out.

Why this exists rather than using the upstream ``sample.py``
-----------------------------------------------------------
``sample.py``'s ``test()`` is built around a Hugging Face *dataset*: it needs
``ds_location``, splits, ground-truth transcripts and a vocabulary derived from
that dataset.  None of that is available (or wanted) when the input is one audio
file we just separated ourselves.

Letting ``ARDataset`` derive the vocabulary is also not an option: it would mean
downloading the full SheetSage-A2S dataset just to reconstruct a token table that
is **already inside the checkpoint** (``hyper_parameters.w2i``).  So this module
loads the checkpoint directly and does the four steps itself:

1. resample the audio to 24 kHz (what MuQ expects),
2. run MuQ to get its final hidden state — the checkpoint's encoder is
   ``preprocessed_muq``, i.e. it consumes these features, **not** raw audio
   (verified against ``preprocess_muq.py``, which applies no normalisation),
3. greedy autoregressive decode from ``<SOS>`` until ``<EOS>`` or the token limit,
4. untokenise into a ``**kern`` score.

The upstream assets are imported rather than reimplemented, so the token grammar
and model definition cannot drift from the released weights.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = ["SheetSage2", "SheetSageOutput", "SHEETSAGE_REPO", "DEFAULT_CHECKPOINT"]

SHEETSAGE_REPO = "MMR-Lab/SheetSage-A2S-model"
DEFAULT_CHECKPOINT = "sheetsage_a2s.ckpt"

# MuQ consumes 24 kHz mono; its features run at ~25 frames per second with 1024
# dimensions each.  Both numbers come from preprocess_muq.py and muq_encoder.py.
MUQ_SAMPLE_RATE = 24_000
MUQ_FEATURE_DIM = 1024


@dataclass
class SheetSageOutput:
    """One transcription."""

    kern: str
    tokens: list[str]
    n_frames: int
    truncated: bool = False

    @property
    def measures(self) -> int:
        return sum(1 for line in self.kern.splitlines() if line.startswith("="))


class SheetSage2:
    """SheetSage2 audio-to-score model.

    ``repo_root`` must point at a checkout of ``SheetSage2Kern_model``; its
    ``my_utils`` and ``networks`` packages are imported from there.
    """

    def __init__(
        self,
        repo_root: str | Path,
        *,
        checkpoint: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        if not (self.repo_root / "networks").is_dir():
            raise FileNotFoundError(
                f"{self.repo_root} does not look like the SheetSage2 checkout "
                "(no networks/ directory)"
            )
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.checkpoint_path = self._resolve_checkpoint(checkpoint)
        self._model = None
        self._muq = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_checkpoint(checkpoint: str | Path | None) -> Path:
        """Local path, or download ``org/repo/file`` from the Hub."""
        if checkpoint is None:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id=SHEETSAGE_REPO, filename=DEFAULT_CHECKPOINT))

        path = Path(checkpoint)
        if path.is_file():
            return path

        text = str(checkpoint)
        parts = text.split("/")
        if len(parts) >= 3:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id="/".join(parts[:2]), filename=parts[2]))
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    # ------------------------------------------------------------------ #
    def load(self) -> "SheetSage2":
        """Load the model and the MuQ encoder.  Idempotent."""
        if self._model is not None:
            return self

        from networks.transformer.model import A2STransformer

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        hparams = checkpoint["hyper_parameters"]

        self.w2i: dict[str, int] = hparams["w2i"]
        self.i2w: dict[int, str] = {int(k): v for k, v in hparams["i2w"].items()}
        self.pad_index = self.w2i["<PAD>"]
        self.sos_index = self.w2i["<SOS>"]
        self.eos_index = self.w2i["<EOS>"]
        self.max_seq_len = int(hparams["max_seq_len"])

        model = A2STransformer(**hparams)
        state = checkpoint["state_dict"]
        # The checkpoint's keys are ``encoder.*`` / ``decoder.*`` with no wrapper
        # prefix, so they load straight in.  Guarding this is worth it: a wrong
        # prefix filter with strict=False loads *nothing* and raises nothing, and the
        # model then decodes from random weights — which looks like a modelling
        # failure (repetitive, structureless output) rather than a loading bug.
        report = model.load_state_dict(state, strict=False)
        loaded = len(state) - len(report.unexpected_keys)
        if report.unexpected_keys or loaded < len(state):
            raise RuntimeError(
                "checkpoint did not load cleanly: "
                f"{len(report.unexpected_keys)} unexpected, "
                f"{len(report.missing_keys)} missing keys"
            )
        model.eval()
        self._model = model.to(self.device)
        return self

    @property
    def muq(self):
        """The frozen MuQ feature extractor."""
        if self._muq is None:
            from muq import MuQ

            encoder = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
            encoder = encoder.eval().to(self.device)
            for parameter in encoder.parameters():
                parameter.requires_grad = False
            self._muq = encoder
        return self._muq

    # ------------------------------------------------------------------ #
    def features(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        """Audio -> MuQ features, shaped ``(1, frames, 1024)``.

        MuQ runs in float32 with autocast disabled: the upstream code notes that
        half precision produces NaNs in its output, which would silently corrupt
        every downstream token.
        """
        import librosa

        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        if sample_rate != MUQ_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=MUQ_SAMPLE_RATE)
        tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            hidden = self.muq(tensor.float(), output_hidden_states=False).last_hidden_state
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        return hidden

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Run the frozen encoder **once** over the whole feature sequence.

        The encoder does not depend on the decoder's input, so recomputing it per
        generated token multiplies the cost of transcription by the number of
        tokens.  The upstream ``sample.py`` calls ``model(x, None, y_in)`` inside the
        decode loop, which does exactly that; doing it once instead is what makes
        this usable — it is the difference between minutes and hours per song.
        """
        model = self._model
        assert model is not None, "call load() first"
        with torch.no_grad():
            return model.encoder(x=features)

    def _tokens(self, words: list[str]) -> list[int]:
        """Map literal kern tokens to ids, skipping any the vocabulary lacks."""
        out: list[int] = []
        for word in words:
            index = self.w2i.get(word)
            if index is None:
                # A tab-separated row is emitted as several tokens followed by <t>;
                # callers that want two spines should pass them pre-split.
                continue
            out.append(int(index))
        return out

    def decode_prompted(
        self,
        memory: torch.Tensor,
        prompt: list[int],
        *,
        max_tokens: int | None = None,
        stop_at_bar: int | None = None,
    ) -> str:
        """Greedy decode conditioned on a kern preamble.

        The model was trained with ``y_in = transcript[:-1]``, so its input always
        carried the kern preamble — the spine names, key and meter.  Decoding from a
        bare ``<SOS>`` feeds it an opening it never saw, which is one reason the
        un-prompted output was malformed.
        """
        from my_utils.kern_headers import FOOTER, HEADERS
        from my_utils.tokeniser import untokenize

        model = self._model
        assert model is not None, "call load() first"

        budget = max_tokens or self.max_seq_len
        sequence = list(prompt) if prompt else [self.sos_index]
        produced: list[str] = []
        bars = 0

        with torch.no_grad():
            for _ in range(budget):
                y_in = torch.tensor([sequence], dtype=torch.long, device=self.device)
                logits = model.decoder(tgt=y_in, memory=memory, memory_len=None)
                next_token = int(logits[0, :, -1].argmax(dim=-1).item())
                if next_token == self.eos_index:
                    break
                word = self.i2w.get(next_token, "")
                produced.append(word)
                sequence.append(next_token)
                if word.startswith("="):
                    bars += 1
                    if stop_at_bar and bars >= stop_at_bar:
                        break

        body = untokenize(produced)
        return "\n".join(HEADERS + [body] + FOOTER)

    def transcribe_features(self, features: torch.Tensor, *, max_tokens: int | None = None):
        """Greedy decode from MuQ features into ``**kern`` tokens."""
        from my_utils.kern_headers import FOOTER, HEADERS
        from my_utils.tokeniser import untokenize

        model = self._model
        assert model is not None, "call load() first"

        budget = max_tokens or self.max_seq_len
        n_frames = int(features.shape[1])

        # Encode once, then decode against the cached memory.
        memory = self.encode(features)

        y_in = torch.tensor([[self.sos_index]], dtype=torch.long, device=self.device)
        produced: list[str] = []
        truncated = False
        with torch.no_grad():
            for _ in range(budget):
                logits = model.decoder(tgt=y_in, memory=memory, memory_len=None)
                next_token = int(logits[0, :, -1].argmax(dim=-1).item())
                if next_token == self.eos_index:
                    break
                produced.append(self.i2w.get(next_token, ""))
                y_in = torch.cat(
                    [y_in, torch.tensor([[next_token]], dtype=torch.long, device=self.device)],
                    dim=1,
                )
            else:
                truncated = True

        body = untokenize(produced)
        kern = "\n".join(HEADERS + [body] + FOOTER)
        return body, produced, n_frames, truncated, kern

    # ------------------------------------------------------------------ #
    def transcribe(self, audio_path: str | Path, *, max_tokens: int | None = None) -> SheetSageOutput:
        """Transcribe an audio file into a ``**kern`` score."""
        import soundfile as sf

        self.load()
        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio = data.T  # (channels, samples)

        features = self.features(audio, int(sample_rate))
        body, tokens, n_frames, truncated, kern = self.transcribe_features(
            features, max_tokens=max_tokens
        )
        return SheetSageOutput(kern=kern, tokens=tokens, n_frames=n_frames, truncated=truncated)

    def transcribe_chunked(
        self,
        audio_path: str | Path,
        *,
        seconds_per_chunk: float = 22.0,
        max_tokens: int | None = None,
    ) -> SheetSageOutput:
        """Transcribe a piece longer than the model's token budget.

        The model was trained on clips averaging **23 seconds**, and its decoder
        allows 3013 tokens.  A full song therefore runs out of budget: on a
        195-second track it emitted exactly 3013 tokens, hit the cap, and stopped
        about a third of the way in.  Feeding successive windows and stitching the
        results is the only way to cover a whole song.

        Two limits are stated plainly because they are real:

        * Each chunk is decoded without the previous chunk's context, so its opening
          bar is the least reliable.  The seam is placed on a barline, which keeps
          the damage inside one bar instead of across a measure.
        * Key and time signature come from the first chunk.  A piece that changes
          either mid-way is notated with its opening one.
        """
        import soundfile as sf

        self.load()
        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio = data.T
        total = audio.shape[1] / sample_rate

        if total <= seconds_per_chunk:
            return self.transcribe(audio_path, max_tokens=max_tokens)

        budget = max_tokens or self.max_seq_len
        bodies: list[str] = []
        all_tokens: list[str] = []
        n_frames = 0
        truncated_any = False

        offset = 0.0
        while offset < total:
            lo = int(offset * sample_rate)
            hi = int((offset + seconds_per_chunk) * sample_rate)
            window = audio[:, lo:hi]
            if window.shape[1] < sample_rate * 2:
                break  # under two seconds left: not worth a chunk
            features = self.features(window, int(sample_rate))
            body, tokens, frames, truncated, _kern = self.transcribe_features(
                features, max_tokens=budget
            )
            bodies.append(body)
            all_tokens.extend(tokens)
            n_frames += frames
            truncated_any = truncated_any or truncated
            offset += seconds_per_chunk

        from my_utils.kern_headers import FOOTER, HEADERS

        kern = "\n".join(HEADERS + [self._join_kern(bodies)] + FOOTER)
        return SheetSageOutput(
            kern=kern, tokens=all_tokens, n_frames=n_frames, truncated=truncated_any
        )

    @staticmethod
    def _join_kern(bodies: list[str]) -> str:
        """Stitch chunk bodies together, cutting at barline boundaries.

        Every chunk restates the headers and ends with a partial bar.  Headers after
        the first are dropped — they would otherwise re-declare the clef mid-score —
        and each seam is placed at the incoming chunk's last barline so a measure is
        never half-joined to another.
        """
        from my_utils.kern_headers import FOOTER, HEADERS

        header_set = set(HEADERS) | set(FOOTER)
        joined: list[str] = []
        for index, body in enumerate(bodies):
            lines = [
                line for line in body.splitlines() if line.strip() and line.strip() not in header_set
            ]
            if index:
                # Drop this chunk's leading header rows and its trailing partial bar.
                lines = [line for line in lines if not line.lstrip().startswith("*")]
                cuts = [i for i, line in enumerate(lines) if line.lstrip().startswith("=")]
                if cuts:
                    lines = lines[: cuts[-1]]
            joined.extend(lines)
        return "\n".join(joined)
