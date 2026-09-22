# Worship Stems

Turn any stereo mix into a multitrack package the band can actually play to:
separated stems, a click that follows the real tempo, a spoken Guide track,
section markers, an ambient pad in the song's key, and a Reaper project that
loads all of it in one double-click.

Everything runs on your own machine. No uploads, no account, no per-song fee.

---

## Install (macOS, Apple Silicon)

```bash
./install_macos.sh                    # app + separation models
./install_macos.sh --with-structure   # also installs allin1 (better Guide)
```

Then:

```bash
./run.sh                              # opens the UI in your browser
./run.sh song.mp3                     # one file, no UI
./run.sh ~/Music/worship --batch      # a whole folder
./run.sh --doctor                     # check what is installed and what is missing
```

If separation ever reports a missing module, `--doctor` names it and
`pip install -r requirements.txt --upgrade` fixes it.

The first separation downloads about 2 GB of model weights. After that the
app works with no internet at all.

---

## What comes out

For `Ce mare esti Tu.mp3` you get a folder
`Ce mare esti Tu [G 76bpm]/` containing:

| File | What it is |
|---|---|
| `00_Original_Mix` | the source, for reference |
| `01_Click` | click with a 2-bar count-in, following the detected beat grid |
| `02_Guide` | spoken cues: "Strofa unu", "Refren", "Bridge" |
| `03_Click_Guide` | both together, for one in-ear channel |
| `10_Vocals_All` / `11_Lead_Vocal` / `12_BGVs` | vocal bus, then lead vs backing |
| `20_Instrumental` | everything except vocals |
| `30_Drums` | the whole kit as one track (`--drumsep` splits it) |
| `40_Bass` | bass |
| `50_Guitars` | guitars |
| `60_Keys` | piano / keys |
| `70_Pads_Synth` | pads, strings, synth — everything not claimed above |
| `80_Ambient_Pad` | a generated pad in the song's key |
| `90_Minus_One` / `91_Acapella` | backing track / vocals only |
| `99_Recombined` | proof: all stems summed back together |
| `CHART.md` | key, tempo, section chart with bar counts |
| `markers.csv`, `sections.cue`, `tempo_map.json`, `session.json` | for import elsewhere |
| `MIDI/*.mid` | the notes each instrument played, transcribed |
| `*.RPP` | Reaper project: every stem on its own track, markers in place |

---

## How the separation works

Each stem is cut by the model that is measurably best at that stem, not by
one model doing everything. Rankings come from the SDR scores shipped with
`audio-separator` (115 models, MUSDB18-style evaluation):

| Stem | Best family | Why |
|---|---|---|
| Vocals / instrumental | BS-RoFormer, Mel-RoFormer | ~11.6 dB vs ~8 dB for Demucs |
| Bass | `hdemucs_mmi` | 12.98 dB, best of everything measured |
| Drums | `htdemucs_ft` | 9.68 dB |
| Lead vs backing | Mel-RoFormer karaoke ensemble | the only models trained for it |
| Kick / snare / toms / hats | MDX23C DrumSep | purpose-trained on kit pieces |
| Guitar / piano | `htdemucs_6s` | the only local 6-stem model |

On "maximum" quality the vocal and karaoke stages run as multi-model
ensembles, which is slower and noticeably cleaner on busy worship mixes.

### The part that makes the sum exact

Every separation model returns estimates that do not add up to the input —
they overlap in places and leave holes in others. Shipping raw estimates plus
a "residual" bucket hides that error rather than fixing it.

Instead, each split converts its estimates into soft masks over the parent's
own spectrogram, and renormalises the masks so they sum to one at every
time-frequency bin:

```
mask_i = |X_i|^p / Σ_j |X_j|^p          Y_i = mask_i · X_parent
```

Because the masks sum to one and the iSTFT is linear, the parts add back up
to the parent sample for sample, carrying the parent's original phase. The
splits are nested, so this holds all the way down the tree:

```
mix
├── vocals ──────── lead_vocal, bgv
└── instrumental ── drums ── kick, snare, toms, hihat, cymbals
                    bass
                    guitars
                    keys
                    pads          (whatever is left)
```

`99_Recombined` is therefore a real check, not a tautology: it is the sum of
the leaf files as written to disk. In testing it lands around **-145 dB**
relative to the original, which is float precision, not an approximation.

The renormalisation also *improves* the stems: it corrects each model's gain
errors and removes double-counted bleed. On the test signals, bass improves
from 19.1 dB to 25.4 dB SI-SDR versus the raw model output.

Set `--partition raw` if you would rather have the untouched model outputs
with the difference booked to one residual track.

### The second pass

The first pass shares every contested time-frequency bin proportionally,
which is why a first-pass stem still has its neighbours faintly behind it.
The second pass goes back over the same bins with the first result as its
prior and does two things:

- **sharpening** — raises the mask exponent from 2 to 5, so a contested bin
  goes almost entirely to whichever stem already owns it
- **band gating** — demotes energy where that instrument physically cannot
  be playing; a bass has nothing to say at 8 kHz, a hi-hat nothing at 40 Hz

Both only move mask weight around, and the masks are renormalised to sum to
one afterwards, so the sum stays exact. Turn it off with `--no-refine`.

### The third pass: putting back what the cut removed

Masking does not damage a stem by adding noise. It damages it by *removing*
things. Where the guitar and the snare land on the same bin, the bin goes to
one of them, and the guitar is left with a hole exactly where the snare hit.
That hole is the warbling, underwater sound — it is an absence, not an
addition, which is why cutting harder can never fix it.

So the third pass puts back what was taken, using only what can be worked
out rather than anything invented:

- **time continuation** — a partial steady before the collision and steady
  after it was steady during it. Reading across 30 ms of a held note is a
  line between two known points.
- **its own phase** — the rebuilt partial continues at the rate measured from
  the frames just before the gap. The obvious shortcut is to borrow the mix's
  phase at that bin, but during a snare hit the mix's phase *is* the snare's:
  in testing that put the right energy in the wrong place and scored −0.4 dB.
  Continuing the partial's own phase scored +0.4 dB instead.
- **the mix ceiling** — the rebuilt value never exceeds what the recording
  actually held in that bin. Energy can be restored; it is never created.
- **refusal** — if a note starts inside the collision and is never heard
  cleanly, there is nothing to continue from and the hole stays. The report
  says how many bins were left alone for that reason.

This means the exported stems no longer sum to the original — they contain
energy that was restored rather than allocated. That is the point, and it is
why both chains exist:

| Chain | Sums to the mix | What it is for |
|---|---|---|
| exact | −142 dB | the proof that nothing was lost in the cut |
| restored | ~−30 dB | the tracks you actually play |

`99_Recombined` is always built from the **exact** chain, so the guarantee
survives the repair. `--both-chains` exports both sets side by side for A/B.
`--no-restore` turns the third pass off.

### The fourth pass: knowing what note was playing

Time-continuation can only read *along* one frequency row, so it fails
whenever the collision lasts longer than the note stays still — which is most
of the time. In testing, a case built with one-second collisions left it with
**0 bins repaired and 6,896 refused**.

Knowing the notes changes the problem. A note at 196 Hz has partials at 196,
392, 588, 784… and a collision does not bury all of them equally. If the third
partial is gone while the first, second and fifth are clean *at that same
instant*, the third is not unknown — it is determined by the others, because
the partials of one note keep a roughly fixed balance while their overall
level moves together. On the same case that defeated the time pass, this
recovered **+3.1 dB SI-SDR**.

The balance is measured, never assumed: it comes from the frames of that same
note where nothing was covering it. Knowing the note also gives the exact
phase advance — a partial at `h·f0` turns by `2π·h·f0·hop/sr` per frame, with
nothing to estimate.

Transcription is polyphonic and fully offline (a 2 MB bundled ONNX model,
about 1 second per 30 seconds of audio). Each instrument also gets a `.mid`
file in `MIDI/`, which is useful on its own for charts and for re-triggering
sounds.

**Two guards, because a wrong note is worse than a hole.** A wrong pitch still
finds energy — some of its harmonics land near a real note's partials by
coincidence — and would then rebuild the rest out of nothing. In testing, a
pitch that was not in the signal at all happily "rebuilt" 4,528 partial-frames
before these were added:

- **it has to be a peak** — owning a frequency bin is not the same as having
  something in it. An empty bin looks "clean" simply because nobody else is
  there either. A real partial stands above its neighbours.
- **it has to fit** — the note's level is solved from the partials that are
  visible, and if the prediction does not match what is actually there, the
  hypothesis is rejected before it says anything about the partials that are
  not.

With both in place the wrong pitch rebuilds nothing. On the real note, the
strict version rebuilds 372 partial-frames instead of 2,580 — for exactly the
same +3.1 dB. It does seven times less and achieves the same.

### Does the record still come out the same?

The final check is the one that makes all of this honest rather than
plausible. Each stem is capped against the mix on its own, but five stems can
each stay under the mix and still add up to more than it ever contained. So
after rebuilding: if every instrument really played what we now claim it
played, does the original recording still come out the same?

Where it does not, the *rebuilt part* is scaled back until it fits. The exact
chain is the floor, so this can only remove invention, never real signal.

One subtlety worth recording: this has to be computed on the complex sum, not
the sum of magnitudes. The exact stems add up to the recording bit for bit,
but their *magnitudes* add up to far more than it wherever their phases
differ — measuring that way flagged a completely honest set of stems as over
budget in 1.3 million bins.

---

## Only the instruments that are actually there

A separator cannot answer "no". Ask it for piano and it returns a piano file
even when there is no piano in the song — it gathers whatever frequencies
look piano-ish and hands them over with complete confidence. Exporting that
file is a lie, and the band hears it before you do.

So the separator does not get to decide what exists. A separate check does,
using the things an ear actually uses:

| Test | What a ghost does |
|---|---|
| audibility | sits 100 dB under everything else — silence with a good CV |
| distinctness | moves exactly when its neighbours move, because it was built from them |
| onsets | never starts a note, just smears |
| harmonicity | noise rather than stacked partials |
| range | energy where that instrument cannot play |

Distinctness and audibility carry veto power, because they are the two tests
a ghost cannot pass. Everything else it inherits for free from the
instruments it was scraped out of — which is exactly how a fake tom track
scores well on attacks and frequency range while being nothing at all.

A rejected stem is **not deleted**. Its audio folds back into its group's
residual, so the leaves still sum to the mix exactly. What disappears is the
false label, not the sound. The report tells you what was refused and why.

Borderline cases are exported and flagged rather than silently dropped
(`keep_uncertain`, on by default) — "listen to this one" is a more honest
answer than a confident guess in either direction. `--no-gate` exports
everything; the strictness slider moves the threshold.

Optionally, CLAP adds a second opinion taken from the **original mix** rather
than from a stem, so it cannot be fooled by something the separator invented.

---

## Two violins, two guitars

Short answer: only when the mix actually tells you where they are.

The 2025 Interspeech work on string quartets settles this. Two violins have
no timbre difference to exploit, so what separates them is position — and
their model reaches 8.26 dB on violin 2 only because it was trained on that
instrument pair in that room. Move the players and it collapses to −0.36 dB.
There is no pretrained model that does this for an arbitrary mix.

What a stereo file does still contain is the panning. If the engineer put two
guitars left and right, that cue survives and can be recovered. If everything
sits in the middle, the information is gone and no processing brings it back.

So the app measures first. `--spatial` reports how much positional
information the material actually holds, and splits only when there is
something to split; on a centred source it says so and does nothing. In
testing, a hard-panned pair went from 1.6 dB to 14.6 dB SI-SDR. Refusing to
split a centred source is the correct answer, not a failure.

One more thing worth saying for worship: most "string sections" in these
mixes are a pad from a keyboard — one patch, one source. There is no violin 1
and violin 2 to recover, because there never were two. A track named
"Violin 2" would be pure fiction.

---

## Does it sound right

SDR is the standard number and it has a ceiling: past about 9 dB on vocals,
listeners stop hearing the difference in blind tests. The 2025 study on
evaluating separation puts figures on it — BSS-Eval metrics correlate 0.6–0.7
with human judgement for masking models like this one, but below 0.5 for
generative ones, where MERT-L12 embedding distance is the better proxy.

Since this pipeline masks rather than generates, SDR still means something
here. But it answers the wrong question for a sound engineer, who needs to
know *what is wrong with this track*, not how many dB it scored. So each stem
is reported on what you would actually listen for:

- **bleed** — how much of its envelope is explained by its neighbours
- **artefacts** — isolated time-frequency islands, the signature of that
  watery, metallic sound
- **verdict** — clean / usable / rough, with the reason in plain words

There is a real tension worth knowing about: generative models sound *better*
to a human, because they resynthesise what is missing instead of cutting. That
is also why they invent, and why they break the guarantee that the parts sum
to the original. You can have the proof or the most natural sound, but not
from the same chain.

---

## Tempo the musician sets

Two modes:

- **follow the recording** (default) — the click tracks the record's own beat
  grid, drift included, so it lines up with the audio
- **fixed** — a steady grid at a BPM you choose, anchored to the first
  downbeat, for when the band plays *to* the click rather than with the record

In the UI it is a BPM box with ÷2 and ×2 buttons, because beat trackers
routinely land an octave out and that should take one click to fix. From the
command line it is `--tempo 80`.

---

## Click and Guide

The click is built from the detected beat grid, so it tracks a live
recording that drifts instead of fighting it. Downbeats get a different,
louder tick; the count-in is placed before bar 1 and every other track is
shifted to match, so the whole folder lines up in any DAW.

The Guide announces each section so the last syllable lands just before the
downbeat. Cues come out in Romanian by default (`Strofa unu`, `Refren`,
`Bridge`, `Final`) or English with `--lang en`. Speech uses macOS `say`
(Romanian voice: Ioana), `piper` or `espeak-ng`, whichever is installed; with
none of them the guide falls back to distinct tones per section rather than
being silently empty.

You get Click and Guide as separate files and combined — all three.

### Section detection

This is the hard part, not the separation. Two backends:

- **allin1** — a trained structure model: tempo, beats, downbeats and
  functional segments (intro / verse / chorus / bridge / inst / solo / break
  / outro). Install with `--with-structure`.
- **built-in** — librosa beat tracking plus Laplacian structure
  segmentation and heuristic labelling. Needs nothing extra, less accurate
  on labels.

Either way the UI shows you the proposed sections on a timeline before
anything is rendered. Play the song, fix what is wrong — relabel, split,
delete, nudge by a bar — then render. Boundaries snap to downbeats.

---

## Borrowing from the other choruses

Everything above *infers* what was in a hole. This does not infer anything.

A worship song plays its chorus three or four times, and the collisions land
differently each time. If the guitar is buried under a crash in chorus 1,
chorus 2 almost certainly has that bar in the clear — so instead of working
out what the guitar probably did, take the bar where it was actually
recorded. That is the strongest evidence anywhere in this pipeline, and it is
why worship music is an easier target than most: it repeats.

Two things make it work rather than smear:

- **alignment** — repeats are never the same length, so the instances are
  warped onto each other bar by bar using the downbeats already detected for
  the Guide. Bar 3 lines up with bar 3 even when the tempo drifted.
- **agreement** — the last chorus usually is *not* the same as the first:
  extra instruments, a different ending. Before borrowing, the two instances
  are compared where both are clean, and if they do not already agree there,
  nothing is taken. In testing, a deliberately different last chorus was
  rejected rather than smeared over the others.

Only the magnitude is borrowed. Two performances are not phase-locked, so a
borrowed phase would fight the host instead of joining it — but the donor's
phase *advance* is the partial's true frequency, and that does transfer. Take
the frequency trajectory from the donor, anchor the phase to the host's last
clean frame.

That lesson turned up three separate times in this project — in the time
pass, in repeat borrowing, and in the resynthesis fill. Each time, dropping in
a foreign phase made the result *worse than the hole*, and each time, taking
the frequency trajectory while anchoring the phase locally fixed it. If you
extend this code, assume it applies to whatever you add next.

---

## Rebuilding the instrument from its own score

The full loop:

```
notes ──> timbre ──> synthesis ──> residual ──> missed notes
  ^                                                  │
  └──────────────────────────────────────────────────┘
```

Transcribe what the instrument played, measure how *that* instrument sounds,
rebuild it from scratch, then check the rebuild against the recording. What
the rebuild fails to explain is the residual — and the residual is not an
error to hide, it is the list of what was got wrong, which is exactly where
to look for the notes the first pass missed. Each round it shrinks; when it
stops shrinking, the score and the timbre are as much of that instrument as
can be accounted for.

The timbre is learned from the recording itself — the median balance of
partials across the notes heard clearly. Nothing generic, nothing pretrained:
this guitar, in this room, on this day. On a test signal with a deliberately
odd timbre (strong 3rd harmonic, weak 2nd), the learned profile came back
`[0.378, 0.055, 0.260, 0.076, 0.136]` against an actual
`[0.375, 0.056, 0.262, 0.075, 0.131]`.

The number this produces — **what share of the stem the score explains** — is
worth more than any dB figure, because it is checkable: you can listen to the
residual and hear what was missed. A number you cannot resynthesise is a
number you cannot verify.

The synthesis has no holes in it, so it also serves as a clean source for the
bins the cut destroyed — capped by the recording, restricted to its own
partials, and phase-anchored as above.

---

## Your own songs: skip all of this

Everything above exists because the original tracks are gone and have to be
guessed back out of a finished mix. **For your own church's songs they are not
gone.** A WING records every channel, so the kick really is the kick
microphone, not an estimate of one.

```bash
./run.sh ~/Recordings/"Ce mare esti Tu" --multitrack
```

Point it at a folder of console channels and the entire hard half of the
problem disappears — no separation, no instrument detection, no hole repair,
no reconstruction, and no argument about whether a stem is real. What is still
worth having is the other half, and you get all of it: tempo, key, the section
map, click, guide, MIDI, markers, and a session that opens in one click.

Channels are recognised from their console names — `01 Kick In`, `05 OH L`,
`08 EG1`, `13 BGV 2` — including the glued-on numbering every desk produces.
Related channels are grouped the way a worship team expects (kick in + kick
out → Kick, OH L + R → Cymbals, BGV 1-4 → BGVs). A click channel is spotted
and kept out of the mix, talkback is dropped, and anything unrecognised keeps
its own track rather than being swept into a bucket — it is a real recording.
Drop a `channels.json` in the folder to name anything it guessed wrong:

```json
{ "17 Shruti Box": "pads" }
```

Here `sum(stems) == mix` is not a reconstruction guarantee. It is just true.

---

## Playing the part instead of extracting it

Once the notes and the timbre are separated from each other, the timbre stops
being fixed. The same part can be played back with the instrument that
recorded it — or with a completely different one, because the score does not
care.

That matters more than it sounds. An extracted keys stem out of a dense mix
is always going to be smeared, because the keys were sharing every bar with a
guitar and two vocals. But the *part* is simple: a handful of held chords.
Rendered with a clean patch it has no bleed, no holes and no artefacts at
all. It is not the record any more — and for a band playing along, it is
often the more useful of the two.

```bash
./run.sh song.mp3 --play-parts
```

Each instrument gets a second track, `…P - <song>.wav`, played from its own
notes. Nothing here is a guess about what was played: the notes come from the
transcription, and the same notes are in `MIDI/` for you to open and check.

Eight patches, deliberately plain rather than emulations: piano, electric
piano, organ, warm pad, strings, clean guitar, bass, soft lead — plus
`learned`, which uses the timbre measured from this recording.

---

## Putting the song on a grid

A studio multitrack is grid-locked: bar 17 starts exactly where the click
says it does. A recording of a band playing is not — it drifts, a little
faster into the chorus, a little slower at the end of a phrase. Give a band a
click taken from a drifting record and they spend the whole song fighting it.

The fix is not a better click. It is to move the audio: stretch each beat
interval by exactly what is needed for it to land where a constant tempo says
it should.

```bash
./run.sh song.mp3 --regrid              # lock to the song's own median tempo
./run.sh song.mp3 --regrid --regrid-bpm 78
```

Every stem gets the *same* map, so they stay locked to each other. In testing,
a performance that wandered **960 ms** away from a steady 90 BPM came back
with a worst spacing error of **30 ms**.

The stretching is a phase vocoder driven by a rate that changes every frame —
librosa's own takes a single constant rate, which cannot express "speed up
through bar 12 and slow down again", so this one takes the frame positions
per output frame.

What it costs: the performance is no longer exactly what was played. A
drummer's deliberate pull-back before a chorus gets flattened. That is a real
loss, which is why it is off by default.

---

## Changing key

Worship teams change key constantly — whoever is leading this week sits a tone
lower. Pitch-shifting a finished mix smears everything at once, because one
set of settings has to serve a kick drum and a cymbal and a voice. With the
stems separated, each one moves on its own terms:

```bash
./run.sh song.mp3 --transpose -2
./run.sh song.mp3 --to-key E        # work out the move, taking the shorter way
```

**Percussion is left exactly where it was.** A kick has no key; shifting it
down makes it flabby and shifting it up makes it ring, for no musical gain.
The ambient pad is re-rendered in the new key rather than shifted, and the key
in every filename, chart and session file follows.

`--to-key` takes the shorter route: G to E is down three, not up nine.

---

## Knowing before you wait

Separation quality is mostly decided by the recording, not the settings. A
sparse arrangement comes apart cleanly; a dense, heavily limited mix does not,
and waiting fifteen minutes does not change that. So the forecast runs first,
in about a second, and says what to expect:

```
forecast: easy (0.16) - open arrangement with room between the parts
```

It measures how much of the spectrum is occupied, the crest factor (a crushed
master has had its transients flattened, and transients are most of what
distinguishes a snare from a bass note), stereo width, register crowding and
brightness. When a song scores badly it also says what to do — usually "find a
less-compressed version; that will beat any setting here".

This is a heuristic, not a trained predictor: a weather forecast, useful for
deciding whether to bother.

---

## Where this differs from MultiTracks

MultiTracks sells the **original studio stems**. Those were recorded
separately, so EG1 and EG2 really are two files and every drum mic really is
its own track. That cannot be reconstructed from a finished mix by anyone,
with any model — the information is not in the file.

What you get here instead:

- any song, including ones nobody sells tracks for
- your own recordings and your church's own songs
- Romanian worship, which is barely covered commercially
- no per-song cost and no subscription
- drum kit pieces, lead vs backing vocals, and a key pad, which most
  separation tools do not attempt

Where it will be weaker than real stems: two guitars in the same register
will not come apart, orchestral arrangements defeat every model, and long
reverb tails smear across stems. Pads and strings land in `70_Pads_Synth`
together — for worship that is usually what you want anyway.

---

## Command line

```
./run.sh song.mp3 --quality maximum --lang ro --format wav24
./run.sh folder/ --batch --quality standard
./run.sh song.mp3 --partition raw --no-pad --count-in 1
```

| Option | Default | |
|---|---|---|
| `--quality` | `maximum` | `fast` / `standard` / `maximum` |
| `--vocal-preset` | `balanced` | `clean` = less bleed, `full` = keeps tails |
| `--lang` | `ro` | Guide language |
| `--format` | `wav24` | `wav24` / `wav16` / `flac` |
| `--partition` | `exact` | `raw` keeps untouched model output |
| `--count-in` | `2` | count-in bars |
| `--tempo` | off | force a steady BPM instead of following the recording |
| `--no-gate` | off | export every stem, even instruments not in the song |
| `--no-refine` | off | skip the second sharpening pass |
| `--spatial` | off | also split guitars by stereo position, where the mix allows |
| `--no-restore` | off | skip rebuilding the holes the cut leaves |
| `--both-chains` | off | export the exact stems AND the rebuilt ones, for A/B |
| `--no-midi` | off | skip transcription and the note-informed repair |
| `--multitrack` | off | the input is a folder of console channels, not a mix |
| `--no-repeats` | off | do not borrow from other instances of a section |
| `--no-resynth` | off | skip rebuilding each instrument from its score |
| `--play-parts` | off | also render each instrument from its notes, clean |
| `--regrid` | off | stretch the song onto a steady tempo |
| `--regrid-bpm` | song's own | tempo to lock to |
| `--transpose` | 0 | move the key by N semitones |
| `--to-key` | | move the song to this key, e.g. `E` |
| `--drumsep` | off | split the kit into kick/snare/toms/hats |
| `--no-forecast` | off | skip the difficulty check |
| `--no-click`, `--no-guide`, `--no-pad` | | skip a stage |
| `--normalize` | off | match stem loudness to -18 LUFS |
| `--doctor` | | check the install and exit |

---

## Speed

On an M-series Mac with 48 GB, a five-minute song at `maximum` takes roughly
8-15 minutes: the ensembles run several models over the whole track. Drop to
`standard` for about a third of that, `fast` for a quick check. Quality is
prioritised over speed everywhere, as intended.

## Tests

```bash
python3 tests/make_fake_song.py     # builds a song with known structure
python3 tests/test_tree.py          # the tree, exactness, graceful fallbacks
python3 tests/test_inventory.py     # ghost instruments, position splits, tempo
python3 tests/test_restore.py       # hole repair, measured against a known source
python3 tests/test_chains.py        # the proof survives the repair
python3 tests/test_harmonic.py     # note-informed repair, guards, mix consistency
python3 tests/test_repeat.py       # alignment, borrowing, refusing a different chorus
python3 tests/test_resynth.py      # timbre learning, the loop, filling from the score
python3 tests/test_multitrack.py   # console channel names, grouping, the no-separation path
python3 tests/test_features.py     # rendering, re-grid, forecast, transposition
```

Both use a fake engine that returns deliberately bad estimates, so they check
the maths without needing any model weights. `test_inventory.py` runs on a mix
that genuinely contains no keys and no toms while the fake engine returns a
stem for both — exactly as a real separator would — and checks that the app
refuses to call them instruments, explains why, and still reconstructs the
mix to −142 dB afterwards.
