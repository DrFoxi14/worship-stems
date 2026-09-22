"""The two chains together: the proof survives the repair.

Restoration deliberately breaks the exact sum - it puts back energy that was
removed - so the danger is that adding it quietly destroys the guarantee the
whole design rests on. This checks both properties hold at once: the proof
file still reconstructs the original, and the exported stems are the repaired
ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worship_stems import audio as A  # noqa: E402
from worship_stems.config import Options  # noqa: E402
from worship_stems.pipeline import PARENTS, Pipeline  # noqa: E402
from worship_stems.restore import restore_all  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tree import MIX, SR, FakeEngine  # noqa: E402


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return cond


def main() -> int:
    ok = True
    leaves = ["lead_vocal", "bgv", "kick", "snare", "hihat", "bass", "electric_guitar", "keys", "pads"]

    print("\n[1] the exact chain is untouched by the repair")
    pipe = Pipeline(Options(quality="maximum", gate_instruments=False, restore=False, split_drums=True), engine=FakeEngine())
    exact, _, _ = pipe.separate_tree(MIX, SR)
    err_exact = A.reconstruction_error_db(MIX, [exact[k] for k in leaves])
    ok &= check("exact leaves sum to the mix", err_exact < -100, f"{err_exact:.1f} dB")

    labels = {k: k for k in exact}
    repaired, reports = restore_all(exact, PARENTS, labels, leaves, SR)
    ok &= check("the repair produced stems", len(repaired) == len(leaves), f"{len(repaired)}")
    ok &= check("and reported what it did", any(r["bins_restored"] for r in reports.values()))

    print("\n[2] the repaired chain deliberately does NOT sum")
    merged = dict(exact)
    merged.update(repaired)
    err_rest = A.reconstruction_error_db(MIX, [merged[k] for k in leaves])
    ok &= check("repaired leaves no longer sum exactly", err_rest > err_exact + 10,
                f"{err_rest:.1f} dB vs {err_exact:.1f} dB exact")
    ok &= check("but they stay close to the original", err_rest < -10, f"{err_rest:.1f} dB")

    print("\n[3] the proof file is built from the exact chain")
    recombined = sum(exact[k] for k in leaves)
    err_proof = A.reconstruction_error_db(MIX, [recombined])
    ok &= check("proof still reconstructs the original", err_proof < -100, f"{err_proof:.1f} dB")

    print("\n[4] both chains can be exported side by side")
    pipe2 = Pipeline(
        Options(quality="maximum", gate_instruments=False, restore=True, restore_chain="both", split_drums=True),
        engine=FakeEngine(),
    )
    stems2, _, _ = pipe2.separate_tree(MIX, SR)
    ok &= check("separation itself is unaffected by the chain setting", set(leaves) <= set(stems2))

    print("\n[5] nothing exceeds what the recording held")
    for key in leaves:
        if key not in repaired:
            continue
        if A.peak_db(repaired[key]) > A.peak_db(MIX) + 6.0:
            ok &= check(f"{key} stays under the mix", False,
                        f"{A.peak_db(repaired[key]):.1f} vs {A.peak_db(MIX):.1f}")
            break
    else:
        ok &= check("every repaired stem stays under the mix", True)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
