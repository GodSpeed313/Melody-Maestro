# Melody Maestro — The Architect

![Python Linting](https://github.com/GodSpeed313/Melody-Maestro/actions/workflows/lint.yml/badge.svg)
[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://melody-maestro-a8qfdpv46bjrtbakbd39gh.streamlit.app)

An AI-powered music theory and beat analysis tool built for FL Studio producers. Upload a track or MIDI file and get elite, genre-specific production advice in seconds.

---

## Features

### Audio Analysis (MP3 / WAV)

- BPM and key detection — upgraded to **Krumhansl-Schmuckler** algorithm (`chroma_cens` + Pearson correlation across all 24 keys simultaneously)
- K-S confidence score displayed alongside the detected key (green ≥ 90%, amber ≥ 75%, red = ambiguous)
- Frequency band energy split — Low (kick/bass), Mid (body), High (snares/hats)
- Spectral centroid (brightness indicator)

### Three-Move Production Advice

- AI-generated advice covering **Drums**, **Texture**, and **Mix**
- Tailored to detected BPM, key, and selected genre
- Every move ends with a concrete FL Studio shortcut or stock plugin tip

### Executive Producer Audit

- **Vibe Score** (1–10) with genre-fit reasoning
- Arrangement critique tied to actual track duration and BPM
- Kick/bass balance and high-end balance notes referencing specific FL Studio tools
- **Frequency clash detector** — 8-band spectral analysis flags congested zones (e.g. "200–400Hz: bass warmth / kick punch"); shows clean confirmation when no buildup detected

### Pi Script Runtime Resolver

- 6 genre and sonic health constraints evaluated before every AI call
- **Corrective injection** — resolver findings prepend the GPT system prompt to prevent BPM/genre mismatches and flag frequency imbalances before advice is generated
- **Policy audit card** — collapsible pass/fail transparency layer below the EP Audit (auto-expands on violations)
- Constraints: BPM/Genre Coherence · Low-End Balance · High-End Presence · Track Duration · Drum Presence (MIDI) · Melody Content (MIDI)
- **Genre Rules Engine** — 6 declarative genre floor policies in `governance/genre_rules.json`; context-scoped `context_rule` fires pre-generation injection enforcing sonic refs, forbidden terms, and style boundaries per genre
- Audit card shows Genre Floor section (◆ active / — skipped) + Session Coherence section (✅ / ⚠️ / —)
- Shape-disciplined to the [Continuum](https://github.com/GodSpeed313/Continuum) resolver — `equality_rule`, `range_rule`, `threshold_rule`, `context_rule` semantics match exactly for future migration

### MIDI Analysis

- BPM, key signature, instrument list, note counts, pitch range, duration
- Three-Move advice informed by the actual instrument arrangement

### Piano Roll Theory Tab

- Scale notes, diatonic chords, and semitone intervals for any key
- Two-octave interactive piano keyboard — scale notes lit in purple, chord tones in blue
- **Dissonance detector** — every diatonic chord labeled stable / tension / dissonant based on harmonic function; every genre progression labeled by its peak tension tier
- Genre-specific chord progressions with Roman numeral labels
- **Export MIDI** button on every progression — generates a 4-bar chord clip at the detected BPM, ready to drop into FL Studio
- Auto-populates from your analyzed track, or set the key manually

### Producer Style Tracker

- Accumulates data from every upload within a session (BPM, key, genre, K-S confidence, clash bands)
- **Session Profile** card appears in the sidebar after 2+ uploads — shows avg BPM + range, dominant key tendency, dominant genre, avg K-S confidence, and recurring frequency clash bands
- Session-state only for MVP — resets on browser close; Supabase cross-session persistence planned for v2

### Genre Support

Rap · Hip-Hop · R&B · Old School R&B / Hip-Hop · Pop · Alternative Rock

---

## Setup

Requirements: Python 3.11+

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv sync
```

Add your API key to a `.env` file:

```env
GROQ_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here
```

Groq is used by default (free tier, fast). Falls back to OpenAI if no Groq key is found.

Run:

```bash
streamlit run main.py
```

---

## Governance

An automated watcher runs every 12 hours via GitHub Actions. It checks that the README stays coherent with the codebase and posts a per-constraint pass/fail report to Discord. Trace files are committed back to `governance/traces/` after each run.

Powered by the [Continuum](https://github.com/GodSpeed313/Continuum) Pi Script engine.
