"""Verify the separation tree without needing model weights.

Instrument gating is switched off and the kit split switched on here on
purpose: this file is about the partition maths, and the kit is the deepest
branch of the tree to test it with. In normal use drums stay as one track. Whether a stem is a real instrument is tested separately
in test_inventory.py.

A fake engine stands in for audio-separator and returns deliberately bad
estimates - wrong gains, bleed between stems, missing energy - because that
is what real models do. The point of the test is that the tree still
partitions the mix exactly and that a failing sub-model degrades instead of
crashing the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worship_stems import audio as A  # noqa: E402
from worship_stems.config import DEMUCS_6S, DRUMSEP, Options  # noqa: E402
from worship_stems.pipeline import Pipeline  # noqa: E402

SR = 44100
N = SR * 20
rng = np.random.default_rng(7)


def st(x: np.ndarray) -> np.ndarray:
    return np.stack([x, np.roll(x, 7) * 0.95]).astype(np.float32)


t = np.arange(N) / SR
TRUTH = {
    "lead": st(0.30 * np.sin(2 * np.pi * 330 * t)),
    "bgv": st(0.10 * np.sin(2 * np.pi * 495 * t)),
    "kick": st(0.40 * np.sin(2 * np.pi * 60 * t) * (np.abs(np.sin(2 * np.pi * 1 * t)) ** 12)),
    "snare": st(0.25 * rng.standard_normal(N).astype(np.float32) * (np.abs(np.sin(2 * np.pi * 1 * t + 1.5)) ** 12)),
    "hihat": st(0.08 * rng.standard_normal(N).astype(np.float32) * (np.abs(np.sin(2 * np.pi * 4 * t)) ** 6)),
    "bass": st(0.35 * np.sin(2 * np.pi * 98 * t)),
    "gtr": st(0.20 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 660 * t)),
    "keys": st(0.18 * np.sin(2 * np.pi * 262 * t)),
    "pads": st(0.12 * np.sin(2 * np.pi * 147 * t) + 0.04 * rng.standard_normal(N).astype(np.float32)),
}
MIX = sum(TRUTH.values()).astype(np.float32)
DRUMS = TRUTH["kick"] + TRUTH["snare"] + TRUTH["hihat"]
VOCALS = TRUTH["lead"] + TRUTH["bgv"]
INST = MIX - VOCALS


def dirty(x: np.ndarray, bleed: np.ndarray, gain: float = 0.8, amount: float = 0.12) -> np.ndarray:
    """What a real model hands back: wrong level, some bleed, a bit of noise."""
    return (x * gain + bleed * amount + 0.002 * rng.standard_normal(x.shape).astype(np.float32)).astype(np.float32)


class FakeEngine:
    def __init__(self, fail: set[str] | None = None):
        self.fail = fail or set()
        self.calls: list[str] = []

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def describe_device() -> str:
        return "fake"

    def release(self) -> None:
        pass

    def run(self, audio, sr, model=None, ensemble_preset=None, wanted=None):
        tag = ensemble_preset or model
        self.calls.append(tag)
        if tag in self.fail:
            raise RuntimeError(f"model {tag} unavailable")

        if tag and "karaoke" in str(tag):
            return {"lead_vocal": dirty(TRUTH["lead"], TRUTH["bgv"]), "bgv": dirty(TRUTH["bgv"], TRUTH["lead"])}
        if tag == DRUMSEP:
            return {
                "kick": dirty(TRUTH["kick"], TRUTH["snare"]),
                "snare": dirty(TRUTH["snare"], TRUTH["kick"]),
                "hihat": dirty(TRUTH["hihat"], TRUTH["snare"]),
            }
        if tag == DEMUCS_6S:
            return {"electric_guitar": dirty(TRUTH["gtr"], TRUTH["keys"]), "keys": dirty(TRUTH["keys"], TRUTH["gtr"])}
        if tag and "demucs" in str(tag):
            return {"drums": dirty(DRUMS, TRUTH["bass"]), "bass": dirty(TRUTH["bass"], TRUTH["kick"]), "other": dirty(TRUTH["pads"], TRUTH["gtr"])}
        return {"vocals": dirty(VOCALS, INST, gain=0.9), "instrumental": dirty(INST, VOCALS, gain=0.95)}


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return condition


def main() -> int:
    ok = True

    print("\n[1] full tree, exact partition")
    pipe = Pipeline(Options(quality="maximum", partition="exact", gate_instruments=False, split_drums=True), engine=FakeEngine())
    stems, warnings, _ = pipe.separate_tree(MIX, SR)
    expected = {"vocals", "instrumental", "lead_vocal", "bgv", "drums", "bass", "electric_guitar", "keys", "pads", "kick", "snare", "hihat"}
    ok &= check("all stems present", expected <= set(stems), f"missing {expected - set(stems)}")
    ok &= check("no warnings", not warnings, str(warnings))

    leaves = ["lead_vocal", "bgv", "kick", "snare", "hihat", "bass", "electric_guitar", "keys", "pads"]
    err = A.reconstruction_error_db(MIX, [stems[k] for k in leaves])
    ok &= check("leaves sum to the mix", err < -100, f"{err:.1f} dB")

    inner = A.reconstruction_error_db(stems["drums"], [stems[k] for k in ("kick", "snare", "hihat")])
    ok &= check("drum kit sums to the drum bus", inner < -100, f"{inner:.1f} dB")

    print("\n[2] separation quality vs the raw estimates")
    pairs = [("lead_vocal", "lead"), ("bass", "bass"), ("keys", "keys")]
    for key, truth in pairs:
        sdr = A.si_sdr(TRUTH[truth], stems[key])
        ok &= check(f"{key} SI-SDR positive", sdr > 0, f"{sdr:.1f} dB")

    print("\n[3] a failing sub-model degrades, it does not crash")
    pipe = Pipeline(Options(quality="maximum", gate_instruments=False, split_drums=True), engine=FakeEngine(fail={DRUMSEP, DEMUCS_6S}))
    stems2, warnings2, _ = pipe.separate_tree(MIX, SR)
    ok &= check("still produced drums and bass", {"drums", "bass"} <= set(stems2))
    ok &= check("kit split skipped", "kick" not in stems2)
    ok &= check("warnings reported", len(warnings2) >= 2, str(warnings2))
    leaves2 = ["lead_vocal", "bgv", "drums", "bass", "pads"]
    err2 = A.reconstruction_error_db(MIX, [stems2[k] for k in leaves2])
    ok &= check("fallback leaves still sum to the mix", err2 < -100, f"{err2:.1f} dB")

    print("\n[4] raw mode keeps model output and books the difference")
    pipe = Pipeline(Options(quality="standard", partition="raw", gate_instruments=False, split_drums=True), engine=FakeEngine())
    stems3, _, _ = pipe.separate_tree(MIX, SR)
    err3 = A.reconstruction_error_db(MIX, [stems3[k] for k in ("lead_vocal", "bgv", "drums", "bass", "electric_guitar", "keys", "pads")])
    ok &= check("raw mode also reconstructs", err3 < -100, f"{err3:.1f} dB")

    print("\n[5] fast tier skips the expensive branches")
    eng = FakeEngine()
    pipe = Pipeline(Options(quality="fast", gate_instruments=False), engine=eng)
    stems4, _, _ = pipe.separate_tree(MIX, SR)
    ok &= check("no ensemble used", not any("vocal_" in str(c) or c == "karaoke" for c in eng.calls), str(eng.calls))
    ok &= check("no drum kit split", "kick" not in stems4)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
