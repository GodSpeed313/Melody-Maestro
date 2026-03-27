import os
import io
import tempfile
import streamlit as st
import librosa
import numpy as np
from openai import OpenAI

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Architect",
    page_icon="🏛️",
    layout="centered",
)

# ── Custom styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }

    .main-header p {
        font-size: 1rem;
        color: #9ca3af;
        margin-top: 0;
    }

    .stat-box {
        background: linear-gradient(135deg, #1e1b4b, #1e293b);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
        line-height: 1.2;
        margin-top: 0.25rem;
    }

    .move-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #a78bfa;
    }

    .move-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #a78bfa;
        margin-bottom: 0.4rem;
    }

    .move-body {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 1.5rem 0;
    }

    .footer-note {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── OpenAI client ───────────────────────────────────────────────────────────────
def get_openai_client():
    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    if not base_url or not api_key:
        st.error("OpenAI integration not configured. Please contact the workspace admin.")
        st.stop()
    return OpenAI(base_url=base_url, api_key=api_key)


# ── Audio analysis ──────────────────────────────────────────────────────────────
def analyze_audio(file_bytes: bytes, filename: str):
    """Return (bpm, key_str) from raw audio bytes."""
    suffix = ".wav" if filename.lower().endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, sr=None, mono=True)
    os.unlink(tmp_path)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)

    # Key — chroma-based
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_idx = int(np.argmax(chroma_mean))
    root = note_names[root_idx]

    # Major vs minor — compare energy profiles
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
    shifted_chroma = np.roll(chroma_mean, -root_idx)
    shifted_chroma /= (shifted_chroma.sum() + 1e-9)
    major_score = np.dot(shifted_chroma, major_template / major_template.sum())
    minor_score = np.dot(shifted_chroma, minor_template / minor_template.sum())
    mode = "Major" if major_score >= minor_score else "Minor"

    return bpm, f"{root} {mode}"


# ── GPT advice ──────────────────────────────────────────────────────────────────
def get_advice(bpm: float, key: str, client: OpenAI):
    """Call GPT-4o with the Three-Move Rule prompt and return structured advice."""
    system_prompt = (
        "You are The Architect — an elite music producer and sound designer who gives "
        "precise, actionable advice. You follow the Three-Move Rule: every response "
        "must cover exactly three areas in order — Drums, Texture, and Mix — and each "
        "section must be specific, practical, and tailored to the supplied BPM and key. "
        "No filler. No generic tips. Sound like a world-class producer talking to a peer."
    )

    user_prompt = (
        f"Analyze this track: BPM = {bpm}, Key = {key}.\n\n"
        "Apply the Three-Move Rule and give me production advice.\n\n"
        "Format your response EXACTLY like this (use the exact section labels):\n\n"
        "MOVE 1 — DRUMS:\n<your drums advice>\n\n"
        "MOVE 2 — TEXTURE:\n<your texture advice>\n\n"
        "MOVE 3 — MIX:\n<your mix advice>"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=700,
        temperature=0.85,
    )
    return response.choices[0].message.content.strip()


def parse_moves(advice_text: str):
    """Parse the three-move response into a dict of {move_label: content}."""
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


# ── UI ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏛️ The Architect</h1>
    <p>Upload your track. Get elite production advice in three moves.</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop an MP3 or WAV file",
    type=["mp3", "wav"],
    label_visibility="collapsed",
)

if uploaded is not None:
    st.audio(uploaded, format="audio/mp3" if uploaded.name.lower().endswith(".mp3") else "audio/wav")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.spinner("Analyzing audio..."):
        try:
            file_bytes = uploaded.read()
            bpm, key = analyze_audio(file_bytes, uploaded.name)
        except Exception as e:
            st.error(f"Audio analysis failed: {e}")
            st.stop()

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
    st.markdown("#### 🎯 Three-Move Production Advice")

    with st.spinner("The Architect is thinking..."):
        try:
            client = get_openai_client()
            advice = get_advice(bpm, key, client)
        except Exception as e:
            st.error(f"Could not reach AI: {e}")
            st.stop()

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

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.expander("Raw GPT response"):
        st.text(advice)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #475569;">
        <div style="font-size:3rem; margin-bottom:1rem;">🎵</div>
        <div style="font-size:1rem;">Upload an MP3 or WAV to begin.</div>
        <div style="font-size:0.8rem; margin-top:0.5rem; color:#334155;">
            Supports files up to ~50 MB · Analysis takes ~5 seconds
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    The Architect · Powered by librosa + GPT-4o · Three-Move Rule
</div>
""", unsafe_allow_html=True)
