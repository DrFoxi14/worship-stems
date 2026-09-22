# Ce s-a schimbat, 22 septembrie 2026

Toate modificările sunt deja aplicate în fișierele din acest arhivă. Nu e
nimic de aplicat manual.

Fiecare afirmație de mai jos a fost măsurată, nu presupusă. Scripturile care
au produs cifrele sunt în `tests/audit/`, deci se pot rula din nou.

---

## 1. `audioread` — separarea murea după partea lentă

**Simptom:** doctorul instalării trecea, iar separarea crăpa la primul decode.

**Cauză:** `audioread` își încarcă backend-urile lene, în `available_backends()`,
nu la import. Backend-ul `rawread` importă `aifc`, `sunau` și `audioop`, pe
care PEP 594 le-a scos din stdlib în Python 3.13. Doar `audioread` 3.1.0
declară backport-urile; 3.0.x spune `Requires-Python: >=3.6`, deci pip îl
instalează fără să se plângă pe 3.13. Iar `install_macos.sh` acceptă 3.13.

Reprodus pe Python 3.13.13:

    import audioread            -> OK      (de-asta trecea doctorul)
    available_backends()        -> ModuleNotFoundError: No module named 'aifc'

**Reparat în:**
- `requirements.txt` — `audioread>=3.1`
- `worship_stems/separate.py` — `decoder_error()` nou, care chiar apelează
  `available_backends()`; `import_error()` îl cheamă
- `install_macos.sh` — aceeași verificare în doctor

---

## 2. Transcrierea nu pornea deloc după installer

**Cauză:** `pip install --no-deps basic-pitch` ține TensorFlow afară, corect,
dar lasă afară și `resampy` și `mir-eval`, pe care `basic_pitch.inference` le
importă prin `note_creation`. Pachetul se importa bine, primul apel crăpa.

`transcription_available()` importa doar `basic_pitch`, deci răspundea `True`.
Importul din `transcribe()` era în afara lui `try`, deci excepția urca prin
`_transcribe` și omora tot pipeline-ul — după separare.

**Reparat în:**
- `requirements.txt` — `resampy`, `mir-eval`
- `worship_stems/harmonic.py` — `transcription_error()` importă modulul real;
  importul mutat în `try`
- `worship_stems/pipeline.py` — mesajul de eroare spune ce lipsește, și
  instrucțiunea de instalare nu mai reproduce bug-ul
- `install_macos.sh` — doctorul numește modulul lipsă

---

## 3. Parametrii de transcriere, greșiți pentru material susținut

Măsurat cu `mir_eval` pe material sintetic cu ground truth exact
(toleranță onset 50 ms, pitch 50 cenți, offset ignorat).

Cu ce era în cod (`0.5 / 0.3 / 60 ms`):

| material | ref | estimat | F1 |
|---|---|---|---|
| linie vocală solo | 14 | 105 | 0.22 |
| cor SATB | 32 | 534 | 0.10 |
| bas | 8 | 17 | 0.64 |
| pian | 32 | 46 | 0.82 |

Recall-ul era 0.91–1.00 — nu rata notele, le fragmenta. `min_note_len_ms=60`
e jumătate din default-ul basic-pitch, iar pragul de onset 0.5 declanșa pe
fiecare pulsație de vibrato.

Cu `0.8 / 0.6 / 400 ms`: SATB **0.10 → 0.91**, solo **0.22 → 1.00**,
bas **0.64 → 1.00**, zero erori de octavă. Ordinea se păstrează pe material
murdar (timing uman, reverb de biserică, zgomot): SATB coboară la 0.74 în cel
mai rău caz, dar tot de la 0.10.

Asta nu strica doar MIDI-ul: `note_informed_restore` și `resynthesize`
reconstruiesc audio pe baza acestor note.

**Reparat în:** `worship_stems/harmonic.py` — `TRANSCRIBE_PROFILES` per stem;
`worship_stems/pipeline.py` — pasează `stem_key`.

**Atenție:** pragurile sunt calibrate pe sintetic. Ordinea ține, valorile
trebuie reverificate pe o înregistrare reală.

Scripturi: `tests/audit/audit.py`, `sweep.py`, `robust.py`

---

## 4. Timbrul nu putea reprezenta un formant

**Cauză:** `Timbre.profile` era indexat pe *numărul armonicii*. Un formant stă
la o frecvență fixă, dar armonica care cade pe el se schimbă cu nota. Aceeași
vocală (formanți la 700/1150/2600 Hz) măsurată pe G3 și pe G4:

    G3 (196 Hz)   armonice 1-8: [0.07 0.31 0.76 1.00 0.89 0.71 0.35 0.10]
    G4 (392 Hz)   armonice 1-8: [0.31 1.00 0.71 0.10 0.01 0.15 0.26 0.02]

Ambele au vârful la 784 Hz în frecvență absolută — formantul e găsit corect.
Dar profilurile pe armonice sunt complet diferite, iar `learn_timbre` le
mediază: amestecă 196·h cu 392·h în aceeași căsuță. Pentru voce asta e cel mai
grav, fiindcă identitatea vocalei *este* structura de formanți.

**Reparat:** `Timbre` are acum `envelope` + `env_freqs` în Hz absoluți, pe
lângă vechiul `profile` (păstrat ca rezervă). `synthesize()` și `render()` cer
amplitudinile **per notă** — asta era schimbarea esențială.

Rezultat, centrul spectral 400–3500 Hz pe aceeași voce transpusă:

| | model vechi | cu anvelopă |
|---|---|---|
| −5 semitonuri | 872 Hz | 1063 Hz |
| 0 semitonuri | 1087 Hz | 1080 Hz |
| +5 semitonuri | 1349 Hz | 1098 Hz |

477 Hz de drift pe zece semitonuri → 35 Hz. De treisprezece ori mai puțin.
Transpunerea pe partidele resintetizate nu mai are nevoie de phase vocoder.

Potrivirea spectrală cu înregistrarea: 8% mai bună. Modest, dar real.

Lățimea nucleului (0.16 octave) e calibrată pe vocale cu formanți cunoscuți,
șase lățimi scorate față de curba reală — vezi `tests/audit/sweep_width.py`.

**Limită:** anvelopa e destul de bună ca să randeze, dar nu e un detector de
formanți fiabil. Pe opt note găsește F1 corect, nu scoate F2 și F3 ca vârfuri
separate. Produsul are nevoie doar de prima.

**De reverificat pe voce reală:** lățimea nucleului, cele 96 de binuri, și
limita de 16 armonice — la un bas cu f0 de 80 Hz ele se opresc la 1.3 kHz,
sub F3. Probabil trebuie număr variabil, nu fix.

Scripturi: `tests/audit/formant_test.py`, `formant_test2.py`, `sweep_width.py`,
`inspect_env.py`

---

## Testele

Toate cele șase teste din `tests/` trec după modificări. Se rulează
individual, nu cu pytest:

    python3 tests/test_chains.py

Cele din `tests/audit/` au nevoie în plus de `mir_eval` și `music21`.

---

## Ce NU e reparat, și merită știut

- **Fișierul „proof" nu descrie stem-urile livrate.** Când
  `restore_chain="restored"`, pachetul conține stem-uri reparate plus un
  „Recombined (proof)" construit din lanțul exact. Cine dă mute pe proof și
  adună restul ajunge la ~30 dB distanță. Metrica „Recombined vs original"
  descrie lanțul exact, nu ce e în pachet. Eticheta ar trebui să spună care
  lanț măsoară.
- **`PARENTS` și `STEMS_BY_KEY` sunt dicționare la nivel de modul** și se
  mută în timpul rulării (`spatial_split`, `restore_chain="both"`,
  `render_parts`). Într-un server de lungă durată cresc de la o piesă la
  alta. Efectul e mai ales memorie, nu rezultate greșite, dar e o scurgere.
- **Al doilea pas de separare nu există.** E doar un comentariu orfan la
  `pipeline.py:559`. Și, așa cum era formulat — cu proof-ul recombinat ca
  input — nu poate funcționa: `partition_exact` garantează că frunzele
  însumează părintele sample cu sample, deci `recombined` *este* mix-ul
  (−142 dB, sub precizia float32). Ar da modelului același semnal.
