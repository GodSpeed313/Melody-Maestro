import os
import tempfile
import streamlit as st
import librosa
import numpy as np
from openai import OpenAI

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Architect",
    page_icon="🏛️",
    layout="wide",
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

    /* Style the Streamlit chat bubbles to match the dark theme */
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
    st.session_state.analysis = None   # {bpm, key, advice, filename}


# ── OpenAI client ───────────────────────────────────────────────────────────────
def get_openai_client():
    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    if not base_url or not api_key:
        st.error("OpenAI integration not configured.")
        st.stop()
    return OpenAI(base_url=base_url, api_key=api_key)


# ── Audio analysis ──────────────────────────────────────────────────────────────
def analyze_audio(file_bytes: bytes, filename: str):
    suffix = ".wav" if filename.lower().endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, sr=None, mono=True)
    os.unlink(tmp_path)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)

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

    return bpm, f"{root} {mode}"


# ── Three-Move advice ───────────────────────────────────────────────────────────
def get_advice(bpm: float, key: str, client: OpenAI):
    system_prompt = (
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


# ── Chat function ───────────────────────────────────────────────────────────────
def chat_with_architect(user_message: str, client: OpenAI):
    """Send a message to GPT-4o with track context and return the reply."""
    analysis = st.session_state.analysis
    if analysis:
        context = (
            f"The user has analyzed a track with these results:\n"
            f"  - BPM: {analysis['bpm']}\n"
            f"  - Key: {analysis['key']}\n"
            f"  - File: {analysis['filename']}\n\n"
            f"The Three-Move advice given was:\n{analysis['advice']}\n\n"
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
        model="gpt-4o",
        messages=messages,
        max_tokens=600,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


# ── Sidebar chat ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">💬 Ask The Architect</div>', unsafe_allow_html=True)

    analysis = st.session_state.analysis
    if analysis:
        st.markdown(
            f'<div class="sidebar-context">'
            f'Context: <span>{analysis["filename"]}</span><br>'
            f'<span>{analysis["bpm"]} BPM</span> · <span>{analysis["key"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-context">No track loaded yet — ask any music production question.</div>',
            unsafe_allow_html=True,
        )

    # Render existing chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
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

uploaded = st.file_uploader(
    "Drop an MP3 or WAV file",
    type=["mp3", "wav"],
    label_visibility="collapsed",
)

if uploaded is not None:
    # Only re-analyze if the file changed
    cached = st.session_state.analysis
    if cached is None or cached.get("filename") != uploaded.name:
        st.audio(uploaded)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        with st.spinner("Analyzing audio..."):
            try:
                file_bytes = uploaded.read()
                bpm, key = analyze_audio(file_bytes, uploaded.name)
            except Exception as e:
                st.error(f"Audio analysis failed: {e}")
                st.stop()

        with st.spinner("The Architect is thinking..."):
            try:
                client = get_openai_client()
                advice = get_advice(bpm, key, client)
            except Exception as e:
                st.error(f"Could not reach AI: {e}")
                st.stop()

        st.session_state.analysis = {
            "filename": uploaded.name,
            "bpm": bpm,
            "key": key,
            "advice": advice,
        }
        st.rerun()

    else:
        # Render from cached analysis
        analysis = st.session_state.analysis
        bpm = analysis["bpm"]
        key = analysis["key"]
        advice = analysis["advice"]

        st.audio(uploaded)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

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
    # Clear cached analysis if file is removed
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

st.markdown("""
<div class="footer-note">
    The Architect · Powered by librosa + GPT-4o · Three-Move Rule
</div>
""", unsafe_allow_html=True)
