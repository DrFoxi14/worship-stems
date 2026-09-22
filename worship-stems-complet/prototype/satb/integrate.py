"""Does the symbolic layer actually connect to the audio pipeline?

The claim to test: an SATB part that was never in the recording can be
generated symbolically, given the timbre measured from the singers who *are*
in the recording, and rendered as a playable stem.

If that works, the choir stems stop being a separation problem - which is
where they were always going to be worst - and become a generation problem,
where nothing has to be pulled out of a mix at all.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/claude/worship-stems/worship-stems")
sys.path.insert(0, "/home/claude/midi_audit")

from harmonize import harmonize  # noqa: E402
from rules import check_all  # noqa: E402
from synth import SR, render as synth_render  # noqa: E402

from worship_stems import audio as A  # noqa: E402
from worship_stems.harmonic import Note  # noqa: E402
from worship_stems.render import render_stem  # noqa: E402
from worship_stems.resynth import learn_timbre  # noqa: E402

VOICE_KEYS = ["soprano", "alto", "tenor", "bass"]
STEM_KEY = {"soprano": "bgv", "alto": "bgv", "tenor": "bgv", "bass": "bgv"}


def satb_to_notes(chords, beats_per_chord=4.0, bpm=76.0):
    """Generated voicings -> one Note list per voice, on the real grid."""
    spb = 60.0 / bpm
    dur = beats_per_chord * spb
    out = {v: [] for v in VOICE_KEYS}
    for i, ch in enumerate(chords):
        t0 = i * dur
        for vi, vname in enumerate(VOICE_KEYS):
            out[vname].append(Note(start=t0, end=t0 + dur * 0.95,
                                   pitch=int(ch[vi]), amplitude=0.6))
    return out


def main():
    bpm = 76.0
    prog = ["G", "D", "Em", "C"] * 2
    melody = [71, 69, 67, 64, 71, 69, 67, 64]

    print("\n[1] generez SATB din acorduri + melodie (fără nicio înregistrare)")
    r = harmonize(prog, melody)
    viol = sum(len(x) for x in check_all(r.chords).values())
    print(f"    {len(prog)} acorduri, {viol} încălcări")

    notes = satb_to_notes(r.chords, beats_per_chord=4.0, bpm=bpm)
    total = sum(len(v) for v in notes.values())
    print(f"    {total} note simbolice, {len(notes)} partide")

    print("\n[2] învăț timbrul de la cântăreții care CHIAR sunt în înregistrare")
    # Stand-in for a real lead-vocal stem: a sung line with its own harmonics.
    lead_notes = [(i * 4 * 60.0 / bpm, (i * 4 + 3.8) * 60.0 / bpm, p)
                  for i, p in enumerate(melody)]
    lead_audio = synth_render(lead_notes, patch="voice", seed=3)
    lead_stereo = np.stack([lead_audio, lead_audio * 0.98]).astype(np.float32)
    lead_as_notes = [Note(s, e, p, 0.6) for s, e, p in lead_notes]
    timbre = learn_timbre(lead_stereo, lead_as_notes, SR)
    prof = timbre.profile[:6]
    print(f"    timbru măsurat din {timbre.observations} note; "
          f"primele armonice {np.round(prof / max(prof.max(), 1e-9), 2)}")

    print("\n[3] randez fiecare partidă de cor cu timbrul învățat")
    duration = len(prog) * 4 * 60.0 / bpm
    stems = {}
    for vname in VOICE_KEYS:
        a, label = render_stem(STEM_KEY[vname], notes[vname], duration, SR,
                               timbre=timbre, patch_name="learned")
        stems[vname] = a
        print(f"    {vname:<9} {a.shape[-1]/SR:5.1f}s  vârf {A.peak_db(a):6.1f} dBFS  [{label}]")

    choir = sum(stems.values())
    print(f"\n[4] corul complet: {choir.shape} la {SR} Hz, "
          f"vârf {A.peak_db(choir):.1f} dBFS")

    out = Path("/home/claude/cor_generat")
    out.mkdir(exist_ok=True)
    for vname, a in stems.items():
        A.save_audio(out / f"{vname}.wav", a, SR, "wav24")
    A.save_audio(out / "cor_complet.wav", choir, SR, "wav24")
    print(f"    scris în {out}")
    print("\n    Niciuna din cele 4 partide nu a fost extrasă din vreun mix.")


if __name__ == "__main__":
    main()
