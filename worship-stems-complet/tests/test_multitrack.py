"""Real console stems: the easy half of the problem, done properly.

Builds a folder that looks like a WING multitrack export - numbered files
with console channel names - and checks that the channels are recognised,
grouped the way a worship team expects, and turned into a full package with
no separation anywhere in sight.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worship_stems import audio as A  # noqa: E402
from worship_stems.config import Options  # noqa: E402
from worship_stems.multitrack import classify, clean_name, load_multitrack  # noqa: E402
from worship_stems.pipeline import Pipeline  # noqa: E402

SR = 44100
DUR = 24.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(23)


def tone(f, amp=0.2, harm=5):
    y = sum((1.0 / h) * np.sin(2 * np.pi * f * h * t) for h in range(1, harm + 1))
    return (amp * y).astype(np.float32)


def hits(rate, decay, amp):
    y = np.zeros(N)
    step = int(SR / rate)
    for i in range(0, N, step):
        L = min(int(SR * 0.3), N - i)
        tt = np.arange(L) / SR
        y[i : i + L] += amp * rng.standard_normal(L) * np.exp(-tt * decay)
    return y.astype(np.float32)


CHANNELS = {
    "01 Kick In": tone(55, 0.35, 2) * (np.abs(np.sin(2 * np.pi * t)) ** 10),
    "02 Kick Out": tone(58, 0.15, 2) * (np.abs(np.sin(2 * np.pi * t)) ** 10),
    "03 Snare Top": hits(2.0, 25, 0.3),
    "04 HiHat": hits(8.0, 60, 0.08),
    "05 OH L": hits(1.0, 4, 0.10),
    "06 OH R": hits(1.0, 4, 0.10),
    "07 Bass DI": tone(65.4, 0.3, 4),
    "08 EG1": tone(196.0, 0.2),
    "09 AG": tone(261.6, 0.15),
    "10 Keys L": tone(329.6, 0.14),
    "11 Keys R": tone(392.0, 0.14),
    "12 Lead Vox": tone(440.0, 0.25),
    "13 BGV 1": tone(523.3, 0.10),
    "14 BGV 2": tone(659.3, 0.08),
    "15 Click": hits(2.0, 200, 0.5),
    "16 Talkback": (0.01 * rng.standard_normal(N)).astype(np.float32),
    "17 Shruti Box": tone(146.8, 0.06),  # deliberately unrecognisable
}


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return cond


def main() -> int:
    ok = True
    work = Path(tempfile.mkdtemp(prefix="ws-mt-"))
    folder = work / "Ce mare esti Tu"
    folder.mkdir(parents=True)
    try:
        for name, audio in CHANNELS.items():
            A.save_audio(folder / name, np.stack([audio, audio]), SR, "wav16")

        print("\n[1] console names are understood")
        cases = [
            ("01 Kick In", "kick"), ("03 Snare Top", "snare"), ("04 HiHat", "hihat"),
            ("05 OH L", "cymbals"), ("07 Bass DI", "bass"), ("08 EG1", "electric_guitar"),
            ("09 AG", "acoustic_guitar"), ("10 Keys L", "keys"), ("12 Lead Vox", "lead_vocal"),
            ("13 BGV 1", "bgv"), ("15 Click", "__click"), ("16 Talkback", "__ignore"),
        ]
        for name, expect in cases:
            key, _side = classify(name)
            if key != expect:
                ok &= check(f"{name} -> {expect}", False, f"got '{key}'")
        else:
            ok &= check("all channel names classified correctly", all(classify(n)[0] == e for n, e in cases))
        ok &= check("numbering is stripped", clean_name("07 Bass DI") == "Bass DI", clean_name("07 Bass DI"))
        ok &= check("sides are detected", classify("05 OH L")[1] == "l")

        print("\n[2] the folder loads into stems")
        stems, mix, sr, report = load_multitrack(folder)
        print(f"      {len(report.channels)} channels -> {len(stems)} stems: {sorted(stems)}")
        ok &= check("kick channels grouped", "kick" in stems)
        ok &= check("overheads grouped into cymbals", "cymbals" in stems)
        ok &= check("BGVs grouped", "bgv" in stems)
        ok &= check("talkback ignored", "Talkback" in " ".join(report.ignored))
        ok &= check("click spotted and kept out", report.click_channel is not None, str(report.click_channel))
        ok &= check("unknown channel kept as its own track",
                    any(k.startswith("ch_") for k in stems), str([k for k in stems if k.startswith("ch_")]))

        print("\n[3] the sum is exact because nothing was separated")
        err = A.reconstruction_error_db(mix, list(stems.values()))
        ok &= check("stems sum to the reference mix", err < -100, f"{err:.1f} dB")

        print("\n[4] a full package, with no separation involved")
        opt = Options(quality="fast", output_format="wav16", make_ambient_pad=False,
                      transcribe_midi=False, note_informed_restore=False)
        result = Pipeline(opt).run_multitrack(folder, work / "out")
        names = {f.key for f in result.files}
        print(f"      {len(result.files)} tracks in {result.folder.name}")
        ok &= check("click was made", "click" in names)
        ok &= check("guide was made", "guide" in names)
        ok &= check("real stems exported", {"kick", "bass", "lead_vocal"} <= names)
        ok &= check("minus one built", "minus_one" in names)
        ok &= check("no separation warnings", not any("Separation" in w for w in result.warnings),
                    str(result.warnings)[:120])
        ok &= check("it says where the stems came from",
                    "console" in result.metrics.get("Source", ""), result.metrics.get("Source", ""))
        ok &= check("unrecognised channel reported",
                    any("not recognised" in w for w in result.warnings))
        ok &= check("a session file was written", (result.folder / "session.json").exists())
        ok &= check("a Reaper project was written", any(result.folder.glob("*.RPP")))

        print("\n[5] channels.json overrides the guesses")
        (folder / "channels.json").write_text('{"17 Shruti Box": "pads"}', encoding="utf-8")
        stems2, _mix2, _sr2, report2 = load_multitrack(folder)
        ok &= check("override applied", not any(k.startswith("ch_") for k in stems2),
                    str([k for k in stems2 if k.startswith("ch_")]))
        ok &= check("nothing unmatched now", not report2.unmatched, str(report2.unmatched))

        print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
