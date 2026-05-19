# Melody Maestro — The Architect

![Python Linting](https://github.com/GodSpeed313/Melody-Maestro/actions/workflows/lint.yml/badge.svg)

An AI-powered music theory and beat analysis tool built for FL Studio producers. Upload a track or MIDI file and get elite, genre-specific production advice in seconds.

---

## Features

### Audio Analysis (MP3 / WAV)

- BPM and key detection via librosa
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

### MIDI Analysis

- BPM, key signature, instrument list, note counts, pitch range, duration
- Three-Move advice informed by the actual instrument arrangement

### Piano Roll Theory Tab

- Scale notes, diatonic chords, and semitone intervals for any key
- Two-octave interactive piano keyboard — scale notes lit in purple, chord tones in blue
- Genre-specific chord progressions with Roman numeral labels
- Auto-populates from your analyzed track, or set the key manually

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
