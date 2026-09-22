"""Real stems, straight off the console.

Everything else in this app exists because the original tracks are gone and
have to be guessed back out of a finished mix. For your own church's songs
they are not gone: a WING records every channel, so the kick really is the
kick microphone, not an estimate of one.

That makes the whole hard half of the problem disappear. No separation, no
instrument detection, no hole repair, no reconstruction - and no argument
about whether a stem is real, because it is the signal that came off the
desk. What is still worth having is the other half: tempo and key, the
section map, click, guide, MIDI, markers and a session that opens in one
click.

So this module takes a folder of recorded channels and hands the rest of the
pipeline a set of stems it can trust completely.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import load_audio, match_length

log = logging.getLogger(__name__)

AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".caf"}

# Channel name -> stem key. Ordered: the first pattern that matches wins, so
# the specific ones ("kick in") must come before the general ones ("in").
CHANNEL_PATTERNS: list[tuple[str, str]] = [
    (r"\b(click|metronome|cue)\b", "__click"),
    (r"\b(talkback|tb|com|intercom)\b", "__ignore"),
    (r"\b(kick|bd|bass\s*drum)\b", "kick"),
    (r"\b(snare|sn|sd)\b", "snare"),
    (r"\b(tom|rack|floor)\b", "toms"),
    (r"\b(hh|hi-?hat|hat)\b", "hihat"),
    (r"\b(oh|overhead|cymbal|ride|crash)\b", "cymbals"),
    (r"\b(drum|kit)\b", "drums"),
    (r"\b(bass|bs|di\s*bass|b\.?gtr)\b", "bass"),
    (r"\b(eg|e\.?gtr|elec.*gtr|electric|lead\s*gtr)\b", "electric_guitar"),
    (r"\b(ag|a\.?gtr|acoustic)\b", "acoustic_guitar"),
    (r"\b(gtr|guitar)\b", "electric_guitar"),
    (r"\b(pad|strings|synth|ambient)\b", "pads"),
    (r"\b(keys|key|piano|kb|rhodes|organ|wurli)\b", "keys"),
    (r"\b(bgv|bv|choir|backing|harmony|harm)\b", "bgv"),
    (r"\b(lead\s*vox|lead\s*vocal|lv|worship\s*leader|wl)\b", "lead_vocal"),
    (r"\b(vox|vocal|voc|mic|mc)\b", "lead_vocal"),
    (r"\b(track|playback|loop)\b", "pads"),
]

SIDE_RE = re.compile(r"[\s_\-(]*(l|r|left|right)[\s_\-)]*$", re.I)


@dataclass
class Channel:
    path: Path
    name: str
    key: str
    side: str = ""  # "l", "r" or ""
    channel_index: int | None = None


@dataclass
class MultitrackReport:
    folder: str
    channels: list[dict] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    click_channel: str | None = None
    sample_rate: int = 44100
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "folder": self.folder,
            "channels": self.channels,
            "unmatched": self.unmatched,
            "ignored": self.ignored,
            "click_channel": self.click_channel,
            "sample_rate": self.sample_rate,
            "duration": round(self.duration, 2),
        }


def clean_name(raw: str) -> str:
    """Strip the numbering a console puts in front of every file.

    Consoles also glue the instrument number onto the name - EG1, BGV2,
    Tom3, Vox1 - which defeats word-boundary matching entirely, so the
    letters and the number are separated here.
    """
    name = raw
    name = re.sub(r"^\s*(tr(ack)?|ch(an(nel)?)?)?[\s_\-]*\d{1,3}[\s_\-]*", "", name, flags=re.I)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\b([A-Za-z]{1,6})(\d{1,2})\b", r"\1 \2", name)
    return re.sub(r"\s+", " ", name).strip()


def classify(name: str) -> tuple[str, str]:
    """Return (stem key, side). An unrecognised name keeps its own track."""
    base = clean_name(name).lower()
    side = ""
    m = SIDE_RE.search(base)
    if m:
        side = m.group(1)[0].lower()
        base = base[: m.start()].strip()

    for pattern, key in CHANNEL_PATTERNS:
        if re.search(pattern, base, re.I):
            return key, side
    return "", side


def discover(folder: Path, overrides: dict[str, str] | None = None) -> list[Channel]:
    """Find the channel files and work out what each one is.

    A `channels.json` in the folder overrides the guesses: {"Track 07": "keys"}.
    """
    folder = Path(folder)
    overrides = dict(overrides or {})
    override_file = folder / "channels.json"
    if override_file.exists():
        try:
            overrides.update(json.loads(override_file.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read %s: %s", override_file, exc)

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT)
    channels: list[Channel] = []
    for path in files:
        name = path.stem
        if name in overrides:
            key, side = overrides[name], ""
            m = SIDE_RE.search(clean_name(name))
            if m:
                side = m.group(1)[0].lower()
        else:
            key, side = classify(name)
        channels.append(Channel(path=path, name=name, key=key, side=side))
    return channels


def load_multitrack(
    folder: str | Path,
    sr: int | None = None,
    group_channels: bool = True,
    overrides: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, int, MultitrackReport]:
    """Load a folder of console channels into stems plus a reference mix.

    The mix is the sum of the channels, so `sum(stems) == mix` is true by
    construction rather than by reconstruction. Sides pan hard, mono channels
    sit in the middle - a flat reference, not a mix decision.
    """
    folder = Path(folder)
    channels = discover(folder, overrides)
    if not channels:
        raise FileNotFoundError(f"no audio files in {folder}")

    report = MultitrackReport(folder=str(folder))
    loaded: list[tuple[Channel, np.ndarray]] = []
    target_sr = sr
    longest = 0

    for ch in channels:
        if ch.key == "__ignore":
            report.ignored.append(ch.name)
            continue
        audio, file_sr = load_audio(ch.path, sr=target_sr)
        if target_sr is None:
            target_sr = file_sr
        longest = max(longest, audio.shape[-1])
        if ch.key == "__click":
            report.click_channel = ch.name
        loaded.append((ch, audio))
        report.channels.append({"file": ch.path.name, "name": ch.name, "stem": ch.key or "(own track)", "side": ch.side})
        if not ch.key:
            report.unmatched.append(ch.name)

    if not loaded:
        raise ValueError(f"every file in {folder} was ignored")

    target_sr = target_sr or 44100
    stems: dict[str, np.ndarray] = {}

    for ch, audio in loaded:
        audio = match_length(audio, longest)
        if ch.side == "l":
            audio = np.stack([audio[0], np.zeros_like(audio[0])])
        elif ch.side == "r":
            audio = np.stack([np.zeros_like(audio[0]), audio[-1]])

        key = ch.key
        if key in {"__click", "__ignore"}:
            continue
        if not key or not group_channels:
            # An unrecognised channel keeps its own name rather than being
            # swept into a bucket - it is a real recording, not a leftover.
            key = key or f"ch_{re.sub(r'[^a-z0-9]+', '_', clean_name(ch.name).lower()) or 'unnamed'}"

        if key in stems:
            stems[key] = stems[key] + audio
        else:
            stems[key] = audio.copy()

    mix = np.zeros((2, longest), dtype=np.float32)
    for key, audio in stems.items():
        mix = mix + audio

    # Leave headroom without changing the balance between channels.
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.99:
        scale = 0.99 / peak
        mix = mix * scale
        stems = {k: v * scale for k, v in stems.items()}

    report.sample_rate = target_sr
    report.duration = longest / target_sr
    return stems, mix.astype(np.float32), target_sr, report


# Which stems to feed the transcriber, given what the console gave us.
def pitched_stems(stems: dict[str, np.ndarray]) -> list[str]:
    percussive = {"kick", "snare", "toms", "hihat", "cymbals", "drums"}
    return [k for k in stems if k not in percussive and not k.startswith("__")]
