import os
import tempfile
import datetime
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
import librosa
import numpy as np
import pretty_midi
from openai import OpenAI

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Architect",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header { text-align: center; padding: 2rem 0 1rem 0; }
    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .main-header p { font-size: 0.95rem; color: #9ca3af; margin-top: 0; }

    .stat-box {
        background: linear-gradient(135deg, #1e1b4b, #1e293b);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stat-label {
        font-size: 0.7rem; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
    }
    .stat-value {
        font-size: 1.9rem; font-weight: 700; color: #a78bfa;
        line-height: 1.2; margin-top: 0.2rem;
    }
    .stat-value-sm {
        font-size: 1.25rem; font-weight: 700; color: #a78bfa;
        line-height: 1.3; margin-top: 0.2rem;
    }

    .move-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.65rem;
        border-left: 4px solid #a78bfa;
    }
    .move-title {
        font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: #a78bfa; margin-bottom: 0.35rem;
    }
    .move-body { font-size: 0.9rem; color: #cbd5e1; line-height: 1.6; }

    /* ── Executive Producer Audit styles ── */
    .audit-header {
        font-size: 1.05rem; font-weight: 700; color: #f1f5f9;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;
    }

    .vibe-score-box {
        background: linear-gradient(135deg, #0f172a, #1e1b4b);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        margin-bottom: 0.65rem;
    }
    .vibe-score-label {
        font-size: 0.7rem; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .vibe-score-number {
        font-size: 3.2rem; font-weight: 800; line-height: 1;
        margin-bottom: 0.3rem;
    }
    .vibe-score-number.hot   { color: #34d399; }
    .vibe-score-number.mid   { color: #fbbf24; }
    .vibe-score-number.cold  { color: #f87171; }
    .vibe-score-tag {
        display: inline-block;
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; padding: 0.2rem 0.6rem;
        border-radius: 999px; margin-bottom: 0.6rem;
    }
    .vibe-score-tag.hot  { background: #064e3b; color: #34d399; }
    .vibe-score-tag.mid  { background: #451a03; color: #fbbf24; }
    .vibe-score-tag.cold { background: #450a0a; color: #f87171; }
    .vibe-reason { font-size: 0.85rem; color: #94a3b8; line-height: 1.55; }

    .audit-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.65rem;
    }
    .audit-card-title {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 0.5rem;
    }
    .audit-card-title.arrangement { color: #60a5fa; border-left: 3px solid #60a5fa; padding-left: 0.5rem; }
    .audit-card-title.sonic       { color: #f472b6; border-left: 3px solid #f472b6; padding-left: 0.5rem; }
    .audit-card-body { font-size: 0.88rem; color: #cbd5e1; line-height: 1.65; }

    .freq-bar-wrap { margin: 0.6rem 0 0.2rem 0; }
    .freq-bar-label {
        font-size: 0.72rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.15rem; display: flex;
        justify-content: space-between;
    }
    .freq-bar-track {
        height: 6px; border-radius: 4px; background: #1e293b; overflow: hidden;
        margin-bottom: 0.35rem;
    }
    .freq-bar-fill {
        height: 100%; border-radius: 4px;
        transition: width 0.5s ease;
    }
    .freq-bar-fill.low  { background: #f472b6; }
    .freq-bar-fill.mid  { background: #a78bfa; }
    .freq-bar-fill.high { background: #38bdf8; }

    /* ── MIDI instrument list ── */
    .midi-inst-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid #1e293b;
        font-size: 0.85rem; color: #cbd5e1;
    }
    .midi-inst-row:last-child { border-bottom: none; }
    .midi-inst-name { color: #a78bfa; font-weight: 600; }
    .midi-inst-notes { color: #64748b; font-size: 0.78rem; }
    .midi-drum-badge {
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; background: #1e1b4b; color: #a78bfa;
        border-radius: 4px; padding: 0.1rem 0.35rem; margin-left: 0.4rem;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 1.25rem 0;
    }

    .sidebar-header {
        font-size: 1rem; font-weight: 700; color: #a78bfa;
        margin-bottom: 0.25rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e293b;
    }
    .sidebar-context {
        background: #0f172a; border: 1px solid #1e293b; border-radius: 8px;
        padding: 0.65rem 0.85rem; font-size: 0.78rem; color: #94a3b8;
        margin-bottom: 0.75rem; line-height: 1.5;
    }
    .sidebar-context span { color: #a78bfa; font-weight: 600; }

    .footer-note {
        text-align: center; color: #475569; font-size: 0.72rem; margin-top: 1.5rem;
    }

    [data-testid="stChatMessage"] {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        margin-bottom: 0.4rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ──────────────────────────────────────────────────────────
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "midi_analysis" not in st.session_state:
    st.session_state.midi_analysis = None


# ── Genre context dictionary ────────────────────────────────────────────────────
GENRE_CONTEXT = {
    "Rap": {
        "sonic_refs": ["808s", "trap hi-hat rolls", "hard-hitting kicks", "dark melodies", "street energy", "punchy snares", "layered ad-libs"],
        "drums":   "trap patterns, 808 sub kicks, rolling hi-hats, snappy snares",
        "texture": "dark synth pads, vocal chops, ominous melodies, heavy bass lines",
        "mix":     "punchy low-end, clear vocal space, controlled 808 sub",
        "forbidden": ["house", "club", "dance floor", "EDM", "pop crossover", "four-on-the-floor"],
    },
    "Hip-Hop": {
        "sonic_refs": ["boom bap or trap influence", "sample-flipping", "heavy kicks", "snappy snares", "melodic loops", "raw energy"],
        "drums":   "boom bap or trap patterns, weighty kicks, crisp snares, swinging hi-hats",
        "texture": "sampled loops, vocal layers, warm pads, melodic hooks",
        "mix":     "balanced low-end, clear mid presence, punchy transients",
        "forbidden": ["house", "club", "dance floor", "EDM", "pop crossover"],
    },
    "R&B": {
        "sonic_refs": ["smooth chord voicings", "layered harmonies", "lush pads", "silky vocal presence", "emotional progressions", "warm bass"],
        "drums":   "laid-back groove, soft kick, snappy snare or clap, subtle hi-hats",
        "texture": "lush pads, warm Rhodes, smooth synth leads, rich chord stacks",
        "mix":     "warm vocal-forward mix, smooth low-end, polished high-end",
        "forbidden": ["trap", "808 sub", "street energy", "hard-hitting", "EDM", "house"],
    },
    "Old School R&B / Hip-Hop": {
        "sonic_refs": ["soul samples", "boom bap grooves", "warm Rhodes", "MPC-style chops", "vinyl crackle texture", "dusty drum breaks", "classic hip-hop energy"],
        "drums":   "boom bap breaks, dusty snares, swinging hi-hats, MPC-chopped samples",
        "texture": "soul samples, warm Rhodes, vinyl texture, classic loop-based arrangements",
        "mix":     "warm slightly lo-fi character, punchy but not over-compressed",
        "forbidden": ["trap", "808", "modern trap", "EDM", "house", "club", "dance floor"],
    },
    "Pop": {
        "sonic_refs": ["catchy hooks", "polished production", "layered vocals", "bright synths", "punchy drums", "radio-ready mix"],
        "drums":   "punchy kick, bright snare, driving hi-hats, clean programming",
        "texture": "bright synth leads, layered vocals, wide stereo pads, sparkly high-end",
        "mix":     "loud bright wide mix, modern loudness standard, hook-forward",
        "forbidden": [],
    },
    "Alternative Rock": {
        "sonic_refs": ["distorted guitars", "live drum feel", "raw mix energy", "powerful riffs", "dynamic range", "gritty texture"],
        "drums":   "live-feeling kick and snare, driving hi-hats, crash cymbals, natural room sound",
        "texture": "distorted guitars, overdriven bass, raw synth layers, layered guitars",
        "mix":     "raw wide dynamic mix, natural transients, not over-compressed",
        "forbidden": ["808", "trap", "EDM", "house", "club"],
    },
}


def genre_context_block(genre: str) -> str:
    ctx = GENRE_CONTEXT.get(genre, GENRE_CONTEXT["Rap"])
    refs = ", ".join(ctx["sonic_refs"])
    forbidden = ", ".join(f"'{t}'" for t in ctx["forbidden"]) if ctx["forbidden"] else "none"
    return (
        f"GENRE RULES — {genre.upper()}:\n"
        f"Sonic references to use: {refs}.\n"
        f"Drums feel: {ctx['drums']}.\n"
        f"Texture feel: {ctx['texture']}.\n"
        f"Mix character: {ctx['mix']}.\n"
        f"Forbidden terms (never use these): {forbidden}.\n"
        f"CRITICAL: Do NOT infer genre from BPM. BPM alone never determines genre.\n\n"
    )


# ── OpenAI client ───────────────────────────────────────────────────────────────
def get_openai_client():
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if groq_key:
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    if openai_key:
        return OpenAI(api_key=openai_key)
    st.error("No API key found. Add GROQ_API_KEY or OPENAI_API_KEY to your .env file.")
    st.stop()


# ── Audio analysis ──────────────────────────────────────────────────────────────
def analyze_audio(file_bytes: bytes, filename: str):
    suffix = ".wav" if filename.lower().endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, sr=None, mono=True)
    os.unlink(tmp_path)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)

    # Key detection via chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_idx = int(np.argmax(chroma_mean))
    root = note_names[root_idx]
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
    shifted_chroma = np.roll(chroma_mean, -root_idx)
    shifted_chroma /= (shifted_chroma.sum() + 1e-9)
    major_score = np.dot(shifted_chroma, major_template / major_template.sum())
    minor_score = np.dot(shifted_chroma, minor_template / minor_template.sum())
    mode = "Major" if major_score >= minor_score else "Minor"
    key = f"{root} {mode}"

    # Frequency band energies via STFT
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_mask  = freqs <= 200
    mid_mask  = (freqs > 200) & (freqs <= 4000)
    high_mask = freqs > 4000
    low_e  = float(S[low_mask].sum())
    mid_e  = float(S[mid_mask].sum())
    high_e = float(S[high_mask].sum())
    total_e = low_e + mid_e + high_e + 1e-9
    low_pct  = round(low_e  / total_e * 100, 1)
    mid_pct  = round(mid_e  / total_e * 100, 1)
    high_pct = round(high_e / total_e * 100, 1)

    # Track duration in seconds
    duration = round(float(len(y) / sr), 1)

    # Spectral centroid (brightness)
    centroid_mean = round(float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()), 0)

    return bpm, key, {
        "low_pct": low_pct,
        "mid_pct": mid_pct,
        "high_pct": high_pct,
        "duration": duration,
        "centroid_hz": centroid_mean,
    }


# ── MIDI analysis ───────────────────────────────────────────────────────────────
def analyze_midi(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    midi = pretty_midi.PrettyMIDI(tmp_path)
    os.unlink(tmp_path)

    # Tempo — use the first tempo change if available
    tempo_change_times, tempos = midi.get_tempo_changes()
    bpm = round(float(tempos[0]), 1) if len(tempos) > 0 else 120.0

    # Key signature (first one in the file, if present)
    key_sig = None
    if midi.key_signature_changes:
        key_sig = pretty_midi.key_number_to_key_name(
            midi.key_signature_changes[0].key_number
        )

    # Instruments and note counts
    instruments = []
    all_pitches = []
    for inst in midi.instruments:
        name = inst.name.strip() if inst.name.strip() else \
               pretty_midi.program_to_instrument_name(inst.program)
        note_count = len(inst.notes)
        instruments.append({
            "name": name,
            "is_drum": inst.is_drum,
            "note_count": note_count,
        })
        all_pitches.extend(n.pitch for n in inst.notes)

    # Pitch range across all instruments
    pitch_min = int(min(all_pitches)) if all_pitches else 0
    pitch_max = int(max(all_pitches)) if all_pitches else 0
    pitch_min_name = pretty_midi.note_number_to_name(pitch_min) if all_pitches else "—"
    pitch_max_name = pretty_midi.note_number_to_name(pitch_max) if all_pitches else "—"

    # Total duration
    duration = round(float(midi.get_end_time()), 1)

    return {
        "bpm": bpm,
        "key": key_sig,
        "instruments": instruments,
        "duration": duration,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_min_name": pitch_min_name,
        "pitch_max_name": pitch_max_name,
    }


# ── Three-Move advice ───────────────────────────────────────────────────────────
def get_advice(bpm: float, key: str, client: OpenAI, genre: str = "Rap"):
    system_prompt = (
        f"{genre_context_block(genre)}"
        "You are The Architect — an elite FL Studio producer and sound designer. "
        "You give precise, actionable advice using the Three-Move Rule: every response "
        "covers exactly three areas in order — Drums, Texture, and Mix. "
        "Each move must be tailored to the specific BPM and key provided, and must end "
        "with either a concrete FL Studio keyboard shortcut (labeled 'Shortcut:') OR a "
        "specific FL Studio stock plugin name and how to use it (labeled 'Plugin:'). "
        "Reference real FL Studio stock plugins like Fruity Peak Controller, Fruity Granulizer, "
        "Fruity Parametric EQ 2, Maximus, Fruity Stereo Enhancer, Fruity Fast Dist, "
        "Fruity Reeverb 2, Fruity Delay 3, Fruity Blood Overdrive, Fruity Flanger, "
        "Fruity Multiband Compressor, Fruity Limiter, Harmor, Sytrus, FLEX, etc. "
        "Reference real FL Studio shortcuts like Alt+U, Ctrl+Q, Shift+drag, etc. "
        "No filler. No generic tips. One shortcut or plugin per move. "
        "Sound like a world-class producer giving a peer a session breakdown."
    )
    user_prompt = (
        f"Genre: {genre}\n"
        f"Analyze this track: BPM = {bpm}, Key = {key}.\n\n"
        "Apply the Three-Move Rule. For each move, give specific advice tied to the BPM "
        "and key, then end with a concrete FL Studio shortcut or stock plugin tip.\n\n"
        "Format your response EXACTLY like this example structure "
        "(use the exact section labels, and end each move with either 'Shortcut:' or 'Plugin:'):\n\n"
        "MOVE 1 — DRUMS:\n"
        "<specific drum advice for the BPM/key>. Shortcut: <real FL Studio shortcut and what it does>\n\n"
        "MOVE 2 — TEXTURE:\n"
        "<specific texture advice for the BPM/key>. Plugin: <FL Studio stock plugin name> — <how to use it>\n\n"
        "MOVE 3 — MIX:\n"
        "<specific mix advice for the BPM/key>. Plugin: <FL Studio stock plugin name> — <how to use it>"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        timeout=25,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=700,
        temperature=0.85,
    )
    return response.choices[0].message.content.strip()


def parse_moves(advice_text: str):
    moves = {}
    sections = [
        ("MOVE 1 — DRUMS", "Drums"),
        ("MOVE 2 — TEXTURE", "Texture"),
        ("MOVE 3 — MIX", "Mix"),
    ]
    for header, label in sections:
        if header + ":" in advice_text:
            after = advice_text.split(header + ":")[1]
            for next_header, _ in sections:
                if next_header + ":" in after:
                    after = after.split(next_header + ":")[0]
            moves[label] = after.strip()
    return moves


# ── Executive Producer Audit ────────────────────────────────────────────────────
def get_beat_grade(bpm: float, key: str, metrics: dict, client: OpenAI, genre: str = "Rap"):
    duration = metrics["duration"]
    low_pct  = metrics["low_pct"]
    mid_pct  = metrics["mid_pct"]
    high_pct = metrics["high_pct"]
    centroid = metrics["centroid_hz"]

    system_prompt = (
        f"{genre_context_block(genre)}"
        "You are an Executive Producer reviewing a track from a session in FL Studio. "
        "You write like a world-class A&R — direct, specific, no filler. "
        "Always use FL Studio terminology. "
        "Reference specific tools like: Piano Roll velocity lanes, "
        "the Fruity Parametric EQ 2 high-shelf, Fruity Limiter ceiling, "
        "Mixer send tracks, the Pattern Block view in the Playlist, "
        "Patcher, Fruity Multiband Compressor, Maximus side-chain, "
        "Fruity Peak Controller, the Step Sequencer velocity buttons, etc. "
        "Your critique must be tied to the actual numbers given. "
        "Never give generic advice."
    )

    user_prompt = (
        f"Track data:\n"
        f"  Genre: {genre}\n"
        f"  BPM: {bpm} | Key: {key} | Duration: {duration}s\n"
        f"  Frequency energy split — Low (kick/bass, ≤200Hz): {low_pct}% | "
        f"Mid (body, 200Hz–4kHz): {mid_pct}% | High (snares/hats, >4kHz): {high_pct}%\n"
        f"  Spectral centroid: {centroid:.0f} Hz\n\n"
        "Give an Executive Producer Audit with EXACTLY this structure and these labels:\n\n"
        "VIBE SCORE: <single integer 1-10>\n"
        "VIBE REASON: <1–2 sentences explaining the score — tie it to genre fit, key, BPM feel, "
        "and whether the energy balance suits the style>\n\n"
        "ARRANGEMENT CRITIQUE:\n"
        "<2–3 sentences. Comment on whether the track duration suggests an intro that's too long "
        f"(typical {genre} track has an 8–16 bar intro at {bpm} BPM, "
        f"that's about {round(8 * (60/bpm) * 4, 1)}–{round(16 * (60/bpm) * 4, 1)} seconds). "
        "Comment on the transition to the hook — does it need more impact? "
        "Give a concrete FL Studio fix using Playlist or Piano Roll terminology.>\n\n"
        "SONIC BALANCE — KICK/BASS:\n"
        "<2 sentences. Based on the low-end percentage, tell me if the kick and bass are clashing "
        "or sitting well. Give a concrete fix using Fruity Parametric EQ 2 or Fruity Multiband Compressor.>\n\n"
        "SONIC BALANCE — HIGH END:\n"
        "<2 sentences. Based on the high-frequency percentage and spectral centroid, tell me if "
        "the snares/hats are too loud, too dull, or balanced. "
        "Give a concrete fix using FL Studio tools.>"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        timeout=25,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=700,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def parse_beat_grade(raw: str):
    result = {
        "vibe_score": None,
        "vibe_reason": "",
        "arrangement": "",
        "sonic_kick": "",
        "sonic_high": "",
    }
    lines = raw.splitlines()
    current = None
    buffer = []

    def flush(key):
        if key and buffer:
            result[key] = " ".join(" ".join(buffer).split())

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("VIBE SCORE:"):
            flush(current); buffer = []
            try:
                result["vibe_score"] = int("".join(filter(str.isdigit, stripped.split(":", 1)[1])))
            except Exception:
                result["vibe_score"] = 7
            current = None
        elif stripped.startswith("VIBE REASON:"):
            flush(current); current = "vibe_reason"
            rest = stripped.split(":", 1)[1].strip()
            buffer = [rest] if rest else []
        elif stripped.startswith("ARRANGEMENT CRITIQUE:"):
            flush(current); current = "arrangement"
            rest = stripped.split(":", 1)[1].strip()
            buffer = [rest] if rest else []
        elif stripped.startswith("SONIC BALANCE — KICK/BASS:"):
            flush(current); current = "sonic_kick"
            rest = stripped.split(":", 1)[1].strip()
            buffer = [rest] if rest else []
        elif stripped.startswith("SONIC BALANCE — HIGH END:"):
            flush(current); current = "sonic_high"
            rest = stripped.split(":", 1)[1].strip()
            buffer = [rest] if rest else []
        elif stripped and current:
            buffer.append(stripped)

    flush(current)
    if result["vibe_score"] is None:
        result["vibe_score"] = 7
    return result


# ── MIDI Three-Move advice ──────────────────────────────────────────────────────
def get_midi_advice(midi_data: dict, client: OpenAI, genre: str = "Rap"):
    bpm         = midi_data.get("bpm", 120)
    key         = midi_data.get("key") or "Unknown"
    duration    = midi_data.get("duration", 0)
    pitch_low   = midi_data.get("pitch_min_name", "—")
    pitch_high  = midi_data.get("pitch_max_name", "—")
    instruments = midi_data.get("instruments", [])

    inst_lines = []
    for inst in instruments:
        tag = " [DRUMS]" if inst.get("is_drum") else ""
        inst_lines.append(f"  - {inst['name']}{tag}: {inst['note_count']} notes")
    inst_block = "\n".join(inst_lines) if inst_lines else "  - (no instruments detected)"

    system_prompt = (
        "You are The Architect — an elite FL Studio producer and sound designer. "
        "You give precise, actionable advice using the Three-Move Rule: every response "
        "covers exactly three areas in order — Drums, Texture, and Mix. "
        "Each move must be tailored to the specific BPM, key, and instrument list provided, "
        "and must end with either a concrete FL Studio keyboard shortcut (labeled 'Shortcut:') "
        "OR a specific FL Studio stock plugin name and how to use it (labeled 'Plugin:'). "
        "Treat the instrument list as the arrangement skeleton — give advice based on what is "
        "actually present. Reference real FL Studio stock plugins like Fruity Peak Controller, "
        "Fruity Granulizer, Fruity Parametric EQ 2, Maximus, Fruity Stereo Enhancer, "
        "Fruity Fast Dist, Fruity Reeverb 2, Fruity Delay 3, Fruity Blood Overdrive, "
        "Fruity Flanger, Fruity Multiband Compressor, Fruity Limiter, Harmor, Sytrus, FLEX, etc. "
        "Reference real FL Studio shortcuts like Alt+U, Ctrl+Q, Shift+drag, etc. "
        "No filler. No generic tips. One shortcut or plugin per move. "
        "Sound like a world-class producer giving a peer a session breakdown. "
        f"{genre_context_block(genre)}"
    )
    user_prompt = (
        f"Genre: {genre}\n"
        f"Analyze this MIDI arrangement:\n"
        f"  BPM: {bpm} | Key: {key} | Duration: {duration}s\n"
        f"  Pitch range: {pitch_low} – {pitch_high}\n"
        f"Instruments present:\n{inst_block}\n\n"
        "Apply the Three-Move Rule. Each move must reference the actual instruments listed above "
        "and be tied to the BPM and key. End each move with a concrete FL Studio shortcut or "
        "stock plugin tip.\n\n"
        "Format your response EXACTLY like this example structure "
        "(use the exact section labels, and end each move with either 'Shortcut:' or 'Plugin:'):\n\n"
        "MOVE 1 — DRUMS:\n"
        "<specific drum advice for the BPM/key/instruments>. Shortcut: <real FL Studio shortcut and what it does>\n\n"
        "MOVE 2 — TEXTURE:\n"
        "<specific texture advice for the BPM/key/instruments>. Plugin: <FL Studio stock plugin name> — <how to use it>\n\n"
        "MOVE 3 — MIX:\n"
        "<specific mix advice for the BPM/key/instruments>. Plugin: <FL Studio stock plugin name> — <how to use it>"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        timeout=25,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=700,
        temperature=0.85,
    )
    return response.choices[0].message.content.strip()


# ── Chat function ───────────────────────────────────────────────────────────────
def chat_with_architect(user_message: str, client: OpenAI):
    analysis = st.session_state.analysis
    if analysis:
        beat_grade_ctx = ""
        if analysis.get("beat_grade_raw"):
            beat_grade_ctx = f"\nThe Executive Producer Audit was:\n{analysis['beat_grade_raw']}\n"
        context = (
            f"The user has analyzed a track with these results:\n"
            f"  - BPM: {analysis['bpm']}\n"
            f"  - Key: {analysis['key']}\n"
            f"  - File: {analysis['filename']}\n\n"
            f"The Three-Move advice given was:\n{analysis['advice']}\n"
            f"{beat_grade_ctx}\n"
            "Use this as context when answering follow-up questions about this track."
        )
    else:
        context = "No track has been analyzed yet. Answer general music production questions."

    system_prompt = (
        "You are The Architect — an elite FL Studio producer, sound designer, and music theory expert. "
        "You give concise, precise, actionable answers to music production questions. "
        "When relevant, mention specific FL Studio plugins, shortcuts, or techniques. "
        "Be conversational but expert. No padding.\n\n" + context
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.chat_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        timeout=25,
        messages=messages,
        max_tokens=600,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


# ── Helpers ─────────────────────────────────────────────────────────────────────
def score_tier(score):
    if score >= 8:
        return "hot", "HEAT"
    elif score >= 6:
        return "mid", "SOLID"
    else:
        return "cold", "NEEDS WORK"


def freq_bar_html(label, pct, css_class):
    return (
        f'<div class="freq-bar-label"><span>{label}</span><span>{pct}%</span></div>'
        f'<div class="freq-bar-track">'
        f'<div class="freq-bar-fill {css_class}" style="width:{min(pct, 100)}%"></div>'
        f'</div>'
    )


def _save_buttons(text: str, filename: str, key_prefix: str = "") -> None:
    col_dl, col_cp, _ = st.columns([1, 1, 6])
    with col_dl:
        st.download_button(
            label="⬇️ Download",
            data=text,
            file_name=filename,
            mime="text/plain",
            key=f"{key_prefix}_download",
        )
    with col_cp:
        safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        btn_id = f"cp-btn-{key_prefix}"
        components.html(
            f"""
            <button id="{btn_id}"
              onclick="navigator.clipboard.writeText(`{safe}`).then(function(){{
                var b=document.getElementById('{btn_id}');
                b.innerText='✅ Copied!';
                setTimeout(function(){{b.innerText='📋 Copy';}},2000);
              }});"
              style="background:#1e293b;color:#94a3b8;border:1px solid #334155;
                     border-radius:6px;padding:5px 14px;font-size:0.82rem;
                     cursor:pointer;font-family:sans-serif;margin-top:2px;">
              📋 Copy
            </button>
            """,
            height=42,
        )


# ── Sidebar chat ────────────────────────────────────────────────────────────────
with st.sidebar:
    selected_genre = st.selectbox(
        "🎵 Genre",
        ["Rap", "Hip-Hop", "R&B", "Old School R&B / Hip-Hop", "Pop", "Alternative Rock"],
        index=0,
        key="genre"
    )

    st.markdown('<div class="sidebar-header">💬 Ask The Architect</div>', unsafe_allow_html=True)

    analysis = st.session_state.analysis
    if analysis:
        st.markdown(
            f'<div class="sidebar-context">'
            f'Context: <span>{analysis["filename"]}</span><br>'
            f'<span>{analysis["bpm"]} BPM</span> · <span>{analysis["key"]}</span><br>'
            f'Genre: <span>{selected_genre}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-context">No track loaded yet — ask any music production question.</div>',
            unsafe_allow_html=True,
        )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your track or anything FL Studio…")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    client = get_openai_client()
                    reply = chat_with_architect(user_input, client)
                except Exception as e:
                    reply = f"Sorry, I couldn't reach the AI: {e}"
            st.markdown(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    if st.session_state.chat_messages:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()


# ── Main panel ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏛️ The Architect</h1>
    <p>Upload your track. Get elite production advice in three moves.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎵 Audio", "🎹 MIDI"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AUDIO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader(
        "Drop an MP3 or WAV file",
        type=["mp3", "wav"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        if st.button("🔄 Re-analyze", key="reanalyze_audio", help="Clear cached results and run fresh AI analysis"):
            st.session_state.analysis = None

        cached = st.session_state.analysis
        if cached is None or cached.get("filename") != uploaded.name or cached.get("genre") != selected_genre:
            with st.spinner("Analyzing audio…"):
                try:
                    file_bytes = uploaded.read()
                    bpm, key, metrics = analyze_audio(file_bytes, uploaded.name)
                except Exception as e:
                    st.error(f"Audio analysis failed: {e}")
                    st.stop()

            client = get_openai_client()
            with st.spinner("The Architect is writing your Three Moves…"):
                try:
                    advice = get_advice(bpm, key, client, genre=selected_genre)
                except Exception:
                    st.session_state.analysis = {
                        "filename": uploaded.name,
                        "genre": selected_genre,
                        "bpm": bpm, "key": key, "metrics": metrics,
                        "advice": None, "beat_grade_raw": "", "ai_error": True,
                        "analyzed_at": datetime.datetime.now(),
                    }
                    st.warning("⚠️ The AI service is temporarily unavailable. Click **Re-analyze** to try again.")
                    st.stop()

            with st.spinner("Running Executive Producer Audit…"):
                try:
                    beat_grade_raw = get_beat_grade(bpm, key, metrics, client, genre=selected_genre)
                except Exception:
                    beat_grade_raw = ""

            st.session_state.analysis = {
                "filename": uploaded.name,
                "genre": selected_genre,
                "bpm": bpm,
                "key": key,
                "metrics": metrics,
                "advice": advice,
                "beat_grade_raw": beat_grade_raw,
                "analyzed_at": datetime.datetime.now(),
            }

        analysis = st.session_state.analysis
        bpm     = analysis["bpm"]
        key     = analysis["key"]
        advice  = analysis.get("advice") or ""
        metrics = analysis.get("metrics", {})
        beat_grade_raw = analysis.get("beat_grade_raw", "")
        ai_error = analysis.get("ai_error", False)

        st.audio(uploaded)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Genre indicator ─────────────────────────────────────────────────────
        _genre_label = analysis.get("genre", "")
        if _genre_label:
            st.markdown(
                f'<div style="margin-bottom:0.75rem;">'
                f'<span style="font-size:0.72rem;font-weight:500;'
                f'background:#1e293b;color:#94a3b8;border:1px solid #334155;'
                f'border-radius:999px;padding:0.15rem 0.6rem;'
                f'letter-spacing:0.04em;text-transform:uppercase;">Genre: {_genre_label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── BPM / Key stat boxes ────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">BPM</div>
                <div class="stat-value">{bpm}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Key</div>
                <div class="stat-value">{key}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if ai_error:
            st.warning("⚠️ The AI service is temporarily unavailable. Click **Re-analyze** to try again.")
            st.stop()

        # ── Three-Move advice ───────────────────────────────────────────────────
        advice_genre = analysis.get("genre", "")
        analyzed_at = analysis.get("analyzed_at")
        analyzed_at_str = analyzed_at.strftime("%I:%M %p").lstrip("0") if analyzed_at else ""
        st.markdown(
            f'#### 🎯 Three-Move Production Advice'
            f'<span style="margin-left:0.6rem;font-size:0.72rem;font-weight:500;'
            f'background:#1e293b;color:#94a3b8;border:1px solid #334155;'
            f'border-radius:999px;padding:0.15rem 0.55rem;vertical-align:middle;'
            f'letter-spacing:0.02em;">Genre: {advice_genre}</span>'
            + (f'<span style="margin-left:0.5rem;font-size:0.72rem;color:#64748b;vertical-align:middle;">'
               f'Last analyzed at {analyzed_at_str}</span>' if analyzed_at_str else ""),
            unsafe_allow_html=True,
        )

        moves = parse_moves(advice)
        icons = {"Drums": "🥁", "Texture": "🌊", "Mix": "🎛️"}
        labels = ["Drums", "Texture", "Mix"]

        if moves:
            for i, label in enumerate(labels, 1):
                content = moves.get(label, "")
                if content:
                    st.markdown(f"""
                    <div class="move-card">
                        <div class="move-title">Move {i} — {icons.get(label, "")} {label}</div>
                        <div class="move-body">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="move-card">
                <div class="move-body">{advice}</div>
            </div>
            """, unsafe_allow_html=True)

        _save_buttons(
            f"Three-Move Production Advice\nGenre: {advice_genre}\n\n{advice}",
            "three_move_advice.txt",
            "tab1_moves",
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Executive Producer Audit ────────────────────────────────────────────
        st.markdown(
            f'#### 🎤 Executive Producer Audit'
            f'<span style="margin-left:0.6rem;font-size:0.72rem;font-weight:500;'
            f'background:#1e293b;color:#94a3b8;border:1px solid #334155;'
            f'border-radius:999px;padding:0.15rem 0.55rem;vertical-align:middle;'
            f'letter-spacing:0.02em;">Genre: {advice_genre}</span>',
            unsafe_allow_html=True,
        )

        if beat_grade_raw:
            grade = parse_beat_grade(beat_grade_raw)
            score = grade["vibe_score"] or 7
            tier, tag_label = score_tier(score)

            left_col, right_col = st.columns([1, 2])

            with left_col:
                st.markdown(f"""
                <div class="vibe-score-box">
                    <div class="vibe-score-label">Vibe Score</div>
                    <div class="vibe-score-number {tier}">{score}<span style="font-size:1.4rem;font-weight:400;color:#475569">/10</span></div>
                    <div><span class="vibe-score-tag {tier}">{tag_label}</span></div>
                    <div class="vibe-reason">{grade["vibe_reason"]}</div>
                </div>
                """, unsafe_allow_html=True)

                if metrics:
                    bars_html = (
                        '<div class="audit-card" style="margin-top:0;">'
                        '<div class="audit-card-title sonic">Frequency Balance</div>'
                        '<div class="freq-bar-wrap">'
                        + freq_bar_html("Low (Kick/Bass ≤200Hz)", metrics["low_pct"], "low")
                        + freq_bar_html("Mid (Body 200Hz–4kHz)", metrics["mid_pct"], "mid")
                        + freq_bar_html("High (Snares/Hats >4kHz)", metrics["high_pct"], "high")
                        + f'<div style="font-size:0.7rem;color:#475569;margin-top:0.4rem;">Centroid: {metrics["centroid_hz"]:.0f} Hz · Duration: {metrics["duration"]}s</div>'
                        + '</div></div>'
                    )
                    st.markdown(bars_html, unsafe_allow_html=True)

            with right_col:
                if grade["arrangement"]:
                    st.markdown(f"""
                    <div class="audit-card">
                        <div class="audit-card-title arrangement">📐 Arrangement Critique</div>
                        <div class="audit-card-body">{grade["arrangement"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if grade["sonic_kick"]:
                    st.markdown(f"""
                    <div class="audit-card">
                        <div class="audit-card-title sonic">🔊 Sonic Balance — Kick &amp; Bass</div>
                        <div class="audit-card-body">{grade["sonic_kick"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if grade["sonic_high"]:
                    st.markdown(f"""
                    <div class="audit-card">
                        <div class="audit-card-title sonic">✨ Sonic Balance — High End</div>
                        <div class="audit-card-body">{grade["sonic_high"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

            _save_buttons(
                f"Executive Producer Audit\nGenre: {advice_genre}\n\n{beat_grade_raw}",
                "executive_producer_audit.txt",
                "tab1_audit",
            )

        else:
            st.info("Executive Producer Audit could not be generated for this track.")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        with st.expander("Raw GPT responses"):
            st.text("── Three-Move Advice ──\n" + advice)
            if beat_grade_raw:
                st.text("\n── Executive Producer Audit ──\n" + beat_grade_raw)

    else:
        if st.session_state.analysis is not None:
            st.session_state.analysis = None

        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #475569;">
            <div style="font-size:3rem; margin-bottom:1rem;">🎵</div>
            <div style="font-size:1rem;">Upload an MP3 or WAV to begin.</div>
            <div style="font-size:0.8rem; margin-top:0.5rem; color:#334155;">
                Supports files up to ~50 MB · Analysis takes ~5 seconds
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MIDI
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    midi_file = st.file_uploader(
        "Drop a MIDI file",
        type=["mid", "midi"],
        label_visibility="collapsed",
        key="midi_uploader",
    )

    if midi_file is not None:
        if st.button("🔄 Re-analyze", key="reanalyze_midi", help="Clear cached results and run fresh AI analysis"):
            st.session_state.midi_analysis = None

        cached_midi = st.session_state.midi_analysis
        if cached_midi is None or cached_midi.get("filename") != midi_file.name or cached_midi.get("genre") != selected_genre:
            with st.spinner("Parsing MIDI file…"):
                try:
                    midi_data = analyze_midi(midi_file.read())
                except Exception as e:
                    st.error(f"MIDI analysis failed: {e}")
                    st.stop()

            with st.spinner("The Architect is writing your Three Moves…"):
                try:
                    client = get_openai_client()
                    midi_advice = get_midi_advice(midi_data, client, genre=selected_genre)
                except Exception as e:
                    midi_advice = ""

            st.session_state.midi_analysis = {
                "filename": midi_file.name,
                "genre": selected_genre,
                "midi_data": midi_data,
                "advice": midi_advice,
                "analyzed_at": datetime.datetime.now(),
            }

        m      = st.session_state.midi_analysis
        md     = m["midi_data"]
        advice = m.get("advice", "")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Genre indicator ─────────────────────────────────────────────────────
        _midi_genre_label = m.get("genre", "")
        if _midi_genre_label:
            st.markdown(
                f'<div style="margin-bottom:0.75rem;">'
                f'<span style="font-size:0.72rem;font-weight:500;'
                f'background:#1e293b;color:#94a3b8;border:1px solid #334155;'
                f'border-radius:999px;padding:0.15rem 0.6rem;'
                f'letter-spacing:0.04em;text-transform:uppercase;">Genre: {_midi_genre_label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Stat boxes: BPM, Duration, Key, Pitch Range ──────────────────────
        key_display = md["key"] if md["key"] else "—"
        pitch_range_display = (
            f'{md["pitch_min_name"]} – {md["pitch_max_name"]}'
            if md["pitch_min_name"] != "—" else "—"
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">BPM</div>
                <div class="stat-value">{md["bpm"]}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Duration</div>
                <div class="stat-value">{md["duration"]}s</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Key Signature</div>
                <div class="stat-value-sm">{key_display}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Pitch Range</div>
                <div class="stat-value-sm">{pitch_range_display}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Instrument list ──────────────────────────────────────────────────
        st.markdown("#### 🎹 Instruments")

        if md["instruments"]:
            rows_html = ""
            for inst in md["instruments"]:
                drum_badge = '<span class="midi-drum-badge">DRUMS</span>' if inst["is_drum"] else ""
                rows_html += (
                    f'<div class="midi-inst-row">'
                    f'<span><span class="midi-inst-name">{inst["name"]}</span>{drum_badge}</span>'
                    f'<span class="midi-inst-notes">{inst["note_count"]:,} notes</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div class="audit-card" style="padding:0.8rem 1.2rem;">{rows_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No instrument tracks found in this MIDI file.")

        # ── Three-Move advice ────────────────────────────────────────────────
        if advice:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            midi_genre = m.get("genre", "")
            midi_analyzed_at = m.get("analyzed_at")
            midi_analyzed_at_str = midi_analyzed_at.strftime("%-I:%M %p") if midi_analyzed_at else ""
            st.markdown(
                f'#### 🎯 Three-Move Production Advice'
                f'<span style="margin-left:0.6rem;font-size:0.72rem;font-weight:500;'
                f'background:#1e293b;color:#94a3b8;border:1px solid #334155;'
                f'border-radius:999px;padding:0.15rem 0.55rem;vertical-align:middle;'
                f'letter-spacing:0.02em;">Genre: {midi_genre}</span>'
                + (f'<span style="margin-left:0.5rem;font-size:0.72rem;color:#64748b;vertical-align:middle;">'
                   f'Last analyzed at {midi_analyzed_at_str}</span>' if midi_analyzed_at_str else ""),
                unsafe_allow_html=True,
            )

            moves = parse_moves(advice)
            icons  = {"Drums": "🥁", "Texture": "🌊", "Mix": "🎛️"}
            labels = ["Drums", "Texture", "Mix"]

            if moves:
                for i, label in enumerate(labels, 1):
                    content = moves.get(label, "")
                    if content:
                        st.markdown(f"""
                        <div class="move-card">
                            <div class="move-title">Move {i} — {icons.get(label, "")} {label}</div>
                            <div class="move-body">{content}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="move-card">
                    <div class="move-body">{advice}</div>
                </div>
                """, unsafe_allow_html=True)

            _save_buttons(
                f"Three-Move Production Advice (MIDI)\nGenre: {midi_genre}\n\n{advice}",
                "three_move_advice_midi.txt",
                "tab2_moves",
            )

    else:
        if st.session_state.midi_analysis is not None:
            st.session_state.midi_analysis = None

        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #475569;">
            <div style="font-size:3rem; margin-bottom:1rem;">🎹</div>
            <div style="font-size:1rem;">Upload a .mid or .midi file to inspect it.</div>
            <div style="font-size:0.8rem; margin-top:0.5rem; color:#334155;">
                Extracts BPM · Key · Instruments · Pitch Range · Duration
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    The Architect · Powered by librosa + GPT-4o · Three-Move Rule · Executive Producer Audit
</div>
""", unsafe_allow_html=True)

