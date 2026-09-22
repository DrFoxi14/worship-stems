"""Guide track: the spoken cues that tell the band what is coming.

Cues are spoken *before* the section they announce, aligned so the last
syllable lands just before the downbeat - that is how a guide track is
actually usable on stage. Speech is rendered by whatever is on the machine:

  macOS `say`    - built in, has Romanian (Ioana) and English voices
  piper          - if installed, best quality and fully offline
  espeak-ng      - last resort, robotic but intelligible

If none of them exist the guide still gets made, using distinct tonal
markers per section type, so the track is never silently empty.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .analyze import Analysis, Section
from .audio import load_audio

log = logging.getLogger(__name__)

PHRASES = {
    "ro": {
        "intro": "Intro", "verse": "Strofa", "prechorus": "Pre refren", "chorus": "Refren",
        "bridge": "Bridge", "inst": "Instrumental", "solo": "Solo", "break": "Pauză",
        "outro": "Final", "end": "Sfârșit",
        "numbers": ["", "unu", "doi", "trei", "patru", "cinci", "șase", "șapte", "opt"],
        "count": ["unu", "doi", "trei", "patru"],
        "last_time": "Ultima dată",
        "big_finish": "Final",
    },
    "en": {
        "intro": "Intro", "verse": "Verse", "prechorus": "Pre chorus", "chorus": "Chorus",
        "bridge": "Bridge", "inst": "Instrumental", "solo": "Solo", "break": "Break",
        "outro": "Outro", "end": "End",
        "numbers": ["", "one", "two", "three", "four", "five", "six", "seven", "eight"],
        "count": ["one", "two", "three", "four"],
        "last_time": "Last time",
        "big_finish": "Big finish",
    },
}

VOICES = {
    "ro": ["Ioana", "ro_RO"],
    "en": ["Samantha", "Daniel", "en_US"],
}

# Fallback marker tones, one per section type (Hz).
MARKER_TONES = {
    "intro": 523.25, "verse": 392.00, "prechorus": 440.00, "chorus": 659.25,
    "bridge": 587.33, "inst": 349.23, "solo": 698.46, "break": 293.66, "outro": 261.63,
}


# --------------------------------------------------------------------------
# Speech backends
# --------------------------------------------------------------------------


def available_tts() -> str | None:
    if platform.system() == "Darwin" and shutil.which("say"):
        return "say"
    if shutil.which("piper"):
        return "piper"
    if shutil.which("espeak-ng"):
        return "espeak-ng"
    if shutil.which("espeak"):
        return "espeak"
    return None


def _synth_speech(text: str, lang: str, sr: int, engine: str | None) -> np.ndarray | None:
    if not engine:
        return None
    work = Path(tempfile.mkdtemp(prefix="ws-tts-"))
    out = work / "cue.wav"
    try:
        if engine == "say":
            voice = _pick_say_voice(lang)
            cmd = ["say", "-o", str(out), "--data-format=LEF32@22050"]
            if voice:
                cmd += ["-v", voice]
            cmd += ["-r", "180", text]
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        elif engine == "piper":
            proc = subprocess.run(
                ["piper", "--output_file", str(out)],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=30,
            )
            _ = proc
        else:
            subprocess.run(
                [engine, "-v", "ro" if lang == "ro" else "en", "-s", "160", "-w", str(out), text],
                check=True,
                capture_output=True,
                timeout=30,
            )
        if not out.exists():
            return None
        audio, file_sr = load_audio(out, sr=sr, mono=True)
        return audio
    except Exception as exc:  # noqa: BLE001
        log.warning("TTS failed for %r: %s", text, exc)
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


_say_voice_cache: dict[str, str | None] = {}


def _pick_say_voice(lang: str) -> str | None:
    if lang in _say_voice_cache:
        return _say_voice_cache[lang]
    voice = None
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
        for candidate in VOICES.get(lang, []):
            for line in listing.splitlines():
                if line.startswith(candidate) or f" {candidate}" in line.split("#")[0]:
                    voice = line.split()[0]
                    break
            if voice:
                break
    except Exception:  # noqa: BLE001
        voice = None
    _say_voice_cache[lang] = voice
    return voice


def _marker_tone(label: str, sr: int, count: int = 1) -> np.ndarray:
    freq = MARKER_TONES.get(label, 440.0)
    beep = []
    for i in range(max(1, count)):
        n = int(sr * 0.13)
        t = np.arange(n) / sr
        env = np.minimum(1.0, np.minimum(t * 200, (n / sr - t) * 200))
        beep.append((np.sin(2 * np.pi * freq * t) * env * 0.5).astype(np.float32))
        beep.append(np.zeros(int(sr * 0.06), dtype=np.float32))
    return np.concatenate(beep)[None, :]


# --------------------------------------------------------------------------
# Guide assembly
# --------------------------------------------------------------------------


def cue_text(section: Section, lang: str) -> str:
    words = PHRASES.get(lang, PHRASES["en"])
    base = words.get(section.label, section.label.title())
    if section.label in {"verse", "chorus", "bridge"} and section.index > 0:
        numbers = words["numbers"]
        if section.index < len(numbers):
            return f"{base} {numbers[section.index]}"
        return f"{base} {section.index}"
    return base


def build_guide(
    analysis: Analysis,
    sr: int,
    total_samples: int,
    offset: float = 0.0,
    lang: str = "ro",
    lead_beats: float = 4.0,
    count_in: bool = True,
    count_in_bars: int = 2,
    stereo: bool = True,
    engine: str | None = "auto",
) -> tuple[np.ndarray, list[dict]]:
    """Return (guide audio, cue list). `offset` matches the click's count-in shift."""
    if engine == "auto":
        engine = available_tts()

    beats = np.asarray(analysis.beats, dtype=float)
    period = float(np.median(np.diff(beats))) if beats.size > 1 else 0.5
    bpb = max(1, analysis.beats_per_bar)

    length = total_samples + int(np.ceil(offset * sr))
    buf = np.zeros((1, length), dtype=np.float32)
    cues: list[dict] = []

    def place(audio: np.ndarray, end_time: float, text: str, kind: str) -> None:
        """Align so speech *finishes* just before `end_time`."""
        if audio is None or audio.size == 0:
            return
        dur = audio.shape[-1] / sr
        start = end_time - dur - 0.12
        at = int(round((start + offset) * sr))
        if at < 0:
            at = 0
        end = min(length, at + audio.shape[-1])
        if end <= at:
            return
        buf[..., at:end] += audio[..., : end - at]
        # Report on the exported timeline (count-in included), not the source one.
        cues.append({"time": round(at / sr, 3), "text": text, "kind": kind})

    # Count-in: "one two three four" over the last count-in bar.
    if count_in and beats.size:
        words = PHRASES.get(lang, PHRASES["en"])["count"]
        first = float(beats[0])
        for i in range(bpb):
            t = first - period * (bpb - i)
            word = words[i % len(words)]
            speech = _synth_speech(word, lang, sr, engine)
            if speech is None:
                speech = _marker_tone("break", sr, 1) * 0.4
            place(speech, t + period * 0.45, word, "count")

    # Section cues, one lead-in before each section start.
    for section in analysis.sections:
        if section.start <= 0.05 and section.label == "intro":
            target = max(0.4, section.start + period * 0.5)
        else:
            target = section.start - period * (lead_beats - 1.0)
        text = cue_text(section, lang)
        speech = _synth_speech(text, lang, sr, engine)
        if speech is None:
            speech = _marker_tone(section.label, sr, max(1, section.index or 1))
        place(speech, target, text, "section")

    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 0:
        buf = buf * (0.7 / peak)
    if stereo:
        buf = np.repeat(buf, 2, axis=0)
    cues.sort(key=lambda c: c["time"])
    return np.clip(buf, -1.0, 1.0).astype(np.float32), cues
