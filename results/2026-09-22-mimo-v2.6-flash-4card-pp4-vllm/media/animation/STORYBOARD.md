# MiMo on the 170HX — storyboard

> **As shipped** (differs from the draft below): the lead is **Mitu, the Mi Bunny** (Xiaomi's mascot, redrawn as fan art without the logo) instead of Clawd; shot A is replaced by a **hook opening** whose frame 0 shows the payoff (117 tok/s, 309B, three glowing cards); karaoke **lyrics** run along the bottom; length is 35.4 s to fit the chosen take. Shot times were snapped to the take's word timestamps (see `START` and `LYRICS` in `mimo.js`).

Logline: Clawd wants to run a giant AI brain on three old mining cards, but it's far too big, so Clawd slices it into three and passes the tokens down the line — and a little helper that guesses ahead makes it fly.

Format: 1080×1080 (1:1), ~34 s, p5.brush papery style (ClaudeAnimationBase engine), locked to the song's tempo (124 BPM → 1 beat = 0.48 s, 1 bar = 1.94 s). Audio = original YuE2 song (below) + light SFX only where the song leaves room.

World: a night workbench in a paper workshop. Palette: dusty sage green (PCB cards), slate blue night, terracotta Clawd, and one warm accent — gold light from the cards. **Colour arc: cold slate → warm gold** as the cards wake up and the tokens flow.

Motif: **three cards in a row**. Asleep and dark at the opening; glowing gold, with Clawd sitting on top, at the close.

Clawd's arc: sleepy → determined (slices the brain) → proud/conducting (pipeline) → excited (helper, 117) → surprised/laughing (prefill) → starstruck (32 streams) → proud (rhyme ending).

Text policy: the engine's rule is no text, but here the numbers are the message, so painted labels appear once per number shot (F–I) plus `MiMo` on the brain in C and a small `64 GB` in B. Everything else stays wordless.

## Song (lyrics sent to YuE2; 3 styles × 2 seeds generating)

```
[Verse]
Three old mining cards, sixty-four gigs each
Slice the giant brain in three, now it's in reach
Pipeline, pipeline, tokens down the line
Guess two tokens ahead, every layer right on time

[Chorus]
Decode one-seventeen, tokens every second
Prefill four thousand, faster than you reckoned
Thirty-two streams, five-sixty-one flow
MiMo on the one-seventy, watch it go
```

Numbers (measured, x16 links): decode 117 tok/s (1 stream, MTP k=2, greedy) · prefill ~4,100 tok/s (c=1, 21k-token prompt) · 561 tok/s aggregate at 32 concurrent streams.

Styles: electro synth-pop · K-pop dance-pop · chiptune electropop. I pick the take with the clearest words and best hook, then the shot times snap to its word timestamps (times below are approximate until then).

## Shots (one shot per lyric line; every cut on a bar line)

```
A  0.0–2.0   [in: iris open from black on the middle card]
   Three dark, sleeping mining cards on a workbench. Clawd asleep against one (zzz). EVENT: first fan twitches, Clawd stirs.

B  2.0–6.0   "Three old mining cards, sixty-four gigs each"   [cut on action]
   Clawd taps each card awake on the beat; fans spin up, memory-chip rows fill ("64 GB" painted small on a card).

C  6.0–10.0  "Slice the giant brain in three, now it's in reach"   [shadow-fall transition]
   The huge pebble-brain "MiMo" lowers in; Clawd (hard hat, DETERMINED) chops it chop-chop-chop; three slabs land on the three cards, each glows gold.

D  10.0–14.0 "Pipeline, pipeline, tokens down the line"   [match cut: slab glow → token bead]
   Gold token beads hop card 1 → 2 → 3 on the beat; Clawd conducts (PROUD).

E  14.0–18.0 "Guess two tokens ahead, every layer right on time"   [helper zips in from screen right]
   A mini-Clawd helper runs ahead dropping two guessed beads; two sparkle ticks = accepted; the relay speeds up.

F  18.0–22.0 "Decode one-seventeen, tokens every second"   [push-in to a painted speedometer]
   Painted gauge; needle swings from 75 up to 117 on "one-seventeen"; painted "117 tok/s". Clawd EXCITED, fans roaring.

G  22.0–26.0 "Prefill four thousand, faster than you reckoned"   [whip pan to a long paper scroll]
   A long paper scroll (the prompt) unrolls toward the cards and gets gulped in one swoop; painted "4,000 tok/s". Clawd SURPRISED then laughing.

H  26.0–30.0 "Thirty-two streams, five-sixty-one flow"   [camera pulls wide]
   32 thin bead streams fan out from the three cards like a fountain, each ending at a tiny glowing cup; painted "32 streams → 561 tok/s". Clawd STARSTRUCK.

I  30.0–34.0 "MiMo on the one-seventy, watch it go"   [streams swirl back into the cards]
   Rhyme ending: three cards glowing gold, the brain above them as a warm lantern, Clawd on top, PROUD. Painted "MiMo · 3× CMP 170HX".
   [out: iris closes on Clawd]
```

Text: the numbers are the point now, so painted labels appear only in F–I (one per shot) plus "MiMo" in C. Everything else stays wordless.

## Checks

- Event in every shot: yes (wake, tap, brain arrives, chops, relay, roar, helper, number jump).
- Reads: one at a time, ≥1.5 s each, cuts on bar lines.
- Transitions at every seam: iris in, cut-on-action, shadow fall, whip pan, match cut, push-in, helper entry, swirl, iris out.
- Text: painted number labels in F–I, `MiMo` in C, `64 GB` in B; nothing else.
- Ending rhymes with opening: same three cards, dark/asleep → gold/alive, Clawd asleep → Clawd proud on top.

## Not in this cut (30 s is tight)

The "why it was hard" beats — the driver bug that crashed every boot, the SM90-only kernels, the QKV bug — would need ~10 more seconds. Option: a 4-second "gremlin" beat in shot C/D (a memory gremlin eating the top of a card, Clawd traps it in a jar). Say if you want it.
