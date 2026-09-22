"""Verify that instruments which are not in the song do not get exported.

The mix here genuinely contains no keys and no toms. The fake engine still
returns a stem for both, exactly as a real separator would - it always
returns something. The test is that the app refuses to call that something
an instrument, and that refusing does not lose any audio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worship_stems import audio as A  # noqa: E402
from worship_stems import spatial as SP  # noqa: E402
from worship_stems.analyze import Analysis, rebuild_grid  # noqa: E402
from worship_stems.config import DEMUCS_6S, DRUMSEP, Options  # noqa: E402
from worship_stems.pipeline import Pipeline  # noqa: E402
from worship_stems.refine import fold_back  # noqa: E402

SR = 44100
N = SR * 20
rng = np.random.default_rng(11)
t = np.arange(N) / SR


def st(x: np.ndarray) -> np.ndarray:
    return np.stack([x, np.roll(x, 7) * 0.95]).astype(np.float32)


def plucked(freq: float, rate: float, amp: float, seed: int) -> np.ndarray:
    """A pitched instrument with real note attacks and stacked harmonics."""
    r = np.random.default_rng(seed)
    out = np.zeros(N, dtype=np.float32)
    step = int(SR / rate)
    for i in range(0, N, step):
        L = min(step, N - i)
        tt = np.arange(L) / SR
        f = freq * (2 ** (r.integers(-2, 3) / 12))
        env = np.exp(-tt * 3.0)
        note = sum((1.0 / h) * np.sin(2 * np.pi * f * h * tt) for h in range(1, 7))
        out[i : i + L] += (note * env).astype(np.float32)
    return amp * out


# A song with vocals, drums, bass, guitar and pads - and NO keys, NO toms.
TRUTH = {
    "lead": st(plucked(330, 2.0, 0.30, 1)),
    "bgv": st(plucked(495, 1.0, 0.10, 2)),
    "kick": st(0.40 * np.sin(2 * np.pi * 60 * t) * (np.abs(np.sin(2 * np.pi * 1 * t)) ** 12)),
    "snare": st(0.25 * rng.standard_normal(N).astype(np.float32) * (np.abs(np.sin(2 * np.pi * 1 * t + 1.5)) ** 12)),
    "hihat": st(0.08 * rng.standard_normal(N).astype(np.float32) * (np.abs(np.sin(2 * np.pi * 4 * t)) ** 6)),
    "bass": st(plucked(98, 2.0, 0.35, 3)),
    "gtr": st(plucked(220, 4.0, 0.22, 4)),
    "pads": st(0.12 * np.sin(2 * np.pi * 147 * t) + 0.10 * np.sin(2 * np.pi * 220 * t)),
}
MIX = sum(TRUTH.values()).astype(np.float32)
DRUMS = TRUTH["kick"] + TRUTH["snare"] + TRUTH["hihat"]
VOCALS = TRUTH["lead"] + TRUTH["bgv"]
INST = MIX - VOCALS


def dirty(x, bleed, gain=0.8, amount=0.12):
    return (x * gain + bleed * amount + 0.002 * rng.standard_normal(x.shape).astype(np.float32)).astype(np.float32)


class GhostEngine:
    """Returns a keys stem and a toms stem that have no original behind them."""

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
        if tag and "karaoke" in str(tag):
            return {"lead_vocal": dirty(TRUTH["lead"], TRUTH["bgv"]), "bgv": dirty(TRUTH["bgv"], TRUTH["lead"])}
        if tag == DRUMSEP:
            return {
                "kick": dirty(TRUTH["kick"], TRUTH["snare"]),
                "snare": dirty(TRUTH["snare"], TRUTH["kick"]),
                "hihat": dirty(TRUTH["hihat"], TRUTH["snare"]),
                # No toms in this song: the model scrapes up a faint smear.
                "toms": (0.05 * TRUTH["kick"] + 0.04 * TRUTH["snare"]).astype(np.float32),
            }
        if tag == DEMUCS_6S:
            return {
                "electric_guitar": dirty(TRUTH["gtr"], TRUTH["pads"]),
                # No keys in this song: a quiet shadow of the pads.
                "keys": (0.16 * TRUTH["pads"] + 0.03 * TRUTH["gtr"]).astype(np.float32),
            }
        if tag and "demucs" in str(tag):
            return {"drums": dirty(DRUMS, TRUTH["bass"]), "bass": dirty(TRUTH["bass"], TRUTH["kick"]), "other": dirty(TRUTH["pads"], TRUTH["gtr"])}
        return {"vocals": dirty(VOCALS, INST, gain=0.9), "instrumental": dirty(INST, VOCALS, gain=0.95)}


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return cond


def main() -> int:
    ok = True

    print("\n[1] a ghost instrument is refused")
    opt = Options(quality="maximum", gate_instruments=True, keep_uncertain=False, split_drums=True)
    pipe = Pipeline(opt, engine=GhostEngine())
    stems, warnings, verdicts = pipe.separate_tree(MIX, SR)

    for key, label in (("keys", "Piano / Keys"), ("toms", "Toms")):
        v = verdicts.get(key)
        ok &= check(f"{key} judged not present", v is not None and v["decision"] != "present",
                    f"{v['decision']} @ {v['confidence']}" if v else "no verdict")
        ok &= check(f"{key} not exported", key not in stems)
    ok &= check("the refusal is explained", any("No " in w for w in warnings), str(warnings[:2]))

    print("\n[2] instruments that ARE there survive")
    for key in ("bass", "electric_guitar", "kick", "snare", "lead_vocal"):
        ok &= check(f"{key} kept", key in stems,
                    f"{verdicts[key]['decision']} @ {verdicts[key]['confidence']}" if key in verdicts else "")

    print("\n[3] refusing loses no audio - the sum is still exact")
    leaves = [k for k in ("lead_vocal", "bgv", "kick", "snare", "toms", "hihat", "cymbals",
                          "bass", "electric_guitar", "keys", "pads") if k in stems]
    drum_leaves = [k for k in ("kick", "snare", "toms", "hihat", "cymbals") if k in stems]
    top = [k for k in leaves if k not in drum_leaves] + (drum_leaves or ["drums"])
    err = A.reconstruction_error_db(MIX, [stems[k] for k in top])
    ok &= check("leaves still sum to the mix", err < -100, f"{err:.1f} dB")
    ok &= check("drum kit still sums to the bus",
                A.reconstruction_error_db(stems["drums"], [stems[k] for k in drum_leaves]) < -100)

    print("\n[4] gating off keeps everything")
    pipe2 = Pipeline(Options(quality="maximum", gate_instruments=False, split_drums=True), engine=GhostEngine())
    stems2, _, verdicts2 = pipe2.separate_tree(MIX, SR)
    ok &= check("keys kept when gating is off", "keys" in stems2)
    ok &= check("but still reported as doubtful", verdicts2["keys"]["decision"] != "present",
                verdicts2["keys"]["decision"])

    print("\n[5] fold_back never loses energy")
    parts = {"a": TRUTH["gtr"], "b": TRUTH["pads"], "c": TRUTH["bass"]}
    total_before = sum(parts.values())
    kept = fold_back(parts, ["b"], "a")
    ok &= check("b is gone", "b" not in kept)
    ok &= check("its audio moved into a",
                A.reconstruction_error_db(total_before, list(kept.values())) < -100)

    print("\n[6] position split only when the position cue exists")
    centred = st(plucked(220, 4.0, 0.2, 9))
    d = SP.pan_diversity(centred, SR)
    ok &= check("centred source refused", not d["usable"], d["reason"])

    left = plucked(220, 4.0, 0.22, 21)
    right = plucked(330, 3.0, 0.20, 22)
    panned = np.stack([left * 0.95 + right * 0.12, left * 0.12 + right * 0.95]).astype(np.float32)
    d2 = SP.pan_diversity(panned, SR)
    ok &= check("panned pair accepted", d2["usable"], f"spread {d2['spread']}, {d2['reason']}")
    layers = SP.split_by_position(panned, SR)
    ok &= check("layers sum back exactly",
                A.reconstruction_error_db(panned, list(layers.values())) < -100)
    # The layer keeps the source's own panning, so the reference is what the
    # left source contributes to each channel, not a centred copy of it.
    ref_left = np.stack([left * 0.95, left * 0.12]).astype(np.float32)
    lz = A.si_sdr(ref_left, layers["left"])
    base = A.si_sdr(ref_left, panned)
    ok &= check("left layer is closer to the left source than the mix was",
                lz > base + 3.0, f"{lz:.1f} dB vs {base:.1f} dB unseparated")

    print("\n[7] the musician can set the tempo")
    a = Analysis(bpm=76.0, beats=[1.0 + i * (60 / 76) for i in range(64)],
                 downbeats=[1.0 + i * (60 / 76) * 4 for i in range(16)], beats_per_bar=4,
                 key="G", mode="major", key_confidence=0.9, camelot="9B", duration=60.0)
    a = rebuild_grid(a, 90.0)
    ok &= check("bpm applied", abs(a.bpm - 90.0) < 0.01, f"{a.bpm}")
    spacing = np.diff(a.beats)
    ok &= check("grid is steady at the new tempo",
                abs(float(np.median(spacing)) - 60.0 / 90.0) < 1e-6, f"{np.median(spacing):.4f}s")
    ok &= check("grid still starts on the first downbeat", abs(a.beats[0] - 1.0) < 1e-6)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
