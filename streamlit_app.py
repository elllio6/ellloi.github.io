import streamlit as st
from PIL import Image
import io
import os
import json
import random
import tempfile

from model_utils import Predictor, save_user_image, fine_tune_on_user_data, ensure_user_dirs

# TTS backends (best-effort)
try:
    import pyttsx3
    TTS_PYTTSX3 = True
except Exception:
    TTS_PYTTSX3 = False

try:
    from gtts import gTTS
    TTS_GTTS = True
except Exception:
    TTS_GTTS = False

# Paths / storage
USER_DATA_DIR = "user_data"
MODEL_FILE = "user_model.h5"
CLASSES_FILE = "classes.json"
STATE_FILE = "state.json"
VOICES_FILE = "voices.json"

ensure_user_dirs(USER_DATA_DIR)
predictor = Predictor(user_model_path=MODEL_FILE, classes_path=CLASSES_FILE)

# Persistent simple points (optional)
if "points" not in st.session_state:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                st.session_state.update(json.load(f))
        except Exception:
            st.session_state.points = 0
    else:
        st.session_state.points = 0

def save_state():
    with open(STATE_FILE, "w") as f:
        json.dump({"points": st.session_state.points}, f)

# Voices management
def default_voices():
    return {
        "Shy": {
            "templates": [
                "Um... I think this might be {label}.",
                "Hmm, maybe it's {label}? I'm not very sure.",
                "Well... possibly {label}."
            ],
            "py_rate": 150,
            "py_voice_index": None
        },
        "Sarcastic": {
            "templates": [
                "Oh sure, definitely {label}... or not.",
                "{label}? Yeah, right. Maybe.",
                "If I had to guess—{label}. Don't quote me."
            ],
            "py_rate": 180,
            "py_voice_index": None
        },
        "Confident": {
            "templates": [
                "I'm pretty sure this is {label}.",
                "This looks like {label} to me.",
                "Confidently saying: {label}."
            ],
            "py_rate": 120,
            "py_voice_index": None
        }
    }

def load_voices():
    if os.path.exists(VOICES_FILE):
        try:
            with open(VOICES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Basic validation: ensure templates exist
            if not isinstance(data, dict) or not data:
                return default_voices()
            return data
        except Exception:
            return default_voices()
    else:
        return default_voices()

def save_voices(voices):
    tmp = VOICES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(voices, f, indent=2, ensure_ascii=False)
    try:
        os.replace(tmp, VOICES_FILE)
    except Exception:
        os.remove(tmp)

voices = load_voices()

# Page config
st.set_page_config(page_title="FunnyImageAI — Voice Maker", layout="wide")
st.title("FunnyImageAI — Always a little bad (create your own voice)")

# Sidebar: TTS backend & voice selection
st.sidebar.header("Audio & Voice")
available_backends = []
if TTS_PYTTSX3:
    available_backends.append("pyttsx3 (offline)")
if TTS_GTTS:
    available_backends.append("gTTS (online)")
available_backends.append("None")
default_idx = 0 if available_backends else -1
tts_backend = st.sidebar.selectbox("TTS backend", options=available_backends, index=default_idx if default_idx >= 0 else 0)

voice_names = list(voices.keys())
voice_names_sorted = sorted(voice_names)
selected_voice = st.sidebar.selectbox("Choose voice script", options=voice_names_sorted, index=0 if voice_names_sorted else -1)

st.sidebar.markdown("---")
st.sidebar.header("Player")
st.sidebar.write(f"Points: {st.session_state.points}")
if st.sidebar.button("Reset points"):
    st.session_state.points = 0
    save_state()
    st.sidebar.success("Points reset")

# Voice management area toggle
edit_voices = st.sidebar.checkbox("Open Voice Maker / Edit Voices", value=False)

# Main UI: upload and guess
st.header("Upload an image — the AI will guess and speak its top guess")
uploaded = st.file_uploader("Choose an image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("AI is guessing..."):
        # always keep it a little bad (funny) — per project decision
        preds = predictor.predict_pil(image, top=3, always_bad=True)

    st.write("AI guesses (top 3):")
    for i, (label, score) in enumerate(preds):
        st.write(f"{i+1}. **{label}** — {score:.2%}")

    # Prepare a spoken phrase using the selected voice
    if selected_voice in voices:
        voice = voices[selected_voice]
        if voice.get("templates"):
            top_label = preds[0][0]
            template = random.choice(voice["templates"])
            phrase = template.format(label=top_label)
        else:
            phrase = f"I think this is {preds[0][0]}."
    else:
        phrase = f"I think this is {preds[0][0]}."

    # Generate audio using selected backend (best-effort)
    audio_bytes = None
    audio_format = None

    if tts_backend.startswith("pyttsx3") and TTS_PYTTSX3:
        try:
            engine = pyttsx3.init()
            rate = voice.get("py_rate", 150) if selected_voice in voices else 150
            engine.setProperty("rate", rate)
            voices_list = engine.getProperty("voices")
            if selected_voice in voices and voice.get("py_voice_index") is not None:
                idx = voice.get("py_voice_index")
                if isinstance(idx, int) and 0 <= idx < len(voices_list):
                    try:
                        engine.setProperty("voice", voices_list[idx].id)
                    except Exception:
                        pass
            else:
                # try deterministic pick by name for variety
                try:
                    idx = abs(hash(selected_voice)) % len(voices_list)
                    engine.setProperty("voice", voices_list[idx].id)
                except Exception:
                    pass
            tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmpf.close()
            engine.save_to_file(phrase, tmpf.name)
            engine.runAndWait()
            with open(tmpf.name, "rb") as f:
                audio_bytes = f.read()
                audio_format = "audio/wav"
            try:
                os.remove(tmpf.name)
            except Exception:
                pass
        except Exception as e:
            st.warning(f"pyttsx3 TTS failed: {e}")
            audio_bytes = None

    if audio_bytes is None and tts_backend.startswith("gTTS") and TTS_GTTS:
        try:
            tts = gTTS(text=phrase, lang="en")
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            audio_bytes = buf.read()
            audio_format = "audio/mp3"
        except Exception as e:
            st.warning(f"gTTS failed: {e}")
            audio_bytes = None

    if audio_bytes:
        st.audio(audio_bytes, format=audio_format)
    else:
        if tts_backend == "None":
            st.info("Audio disabled (None selected).")
        else:
            st.info("No working TTS backend available or generation failed. Install pyttsx3 (offline) or gTTS (online).")

    # Interaction: user confirms or corrects
    cols = st.columns(3)
    if cols[0].button("AI correct"):
        st.session_state.points += 1
        save_state()
        st.success("Point awarded!")
    if cols[1].button("AI incorrect"):
        st.info("Provide the correct label to teach the AI")
        correct_label = st.text_input("Correct label (exact):", key="correct_label_input")
        if st.button("Save labeled image", key="save_labeled"):
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)
            save_user_image(buf, correct_label, USER_DATA_DIR)
            st.success(f"Saved image as label: {correct_label}")
    if cols[2].button("Skip"):
        st.write("Skipped")

st.markdown("---")
st.header("Training (optional)")
st.write("You can collect labeled images by saving incorrect guesses, then fine-tune a small model on your dataset.")
st.write("Note: the AI will remain slightly 'funny/wrong' even after training (project choice).")

if st.button("Run short fine-tune now"):
    fine_tune_on_user_data(USER_DATA_DIR, MODEL_FILE, CLASSES_FILE, epochs=3)
    predictor.reload_if_updated()
    st.success("Fine-tune finished. The model is saved locally — but the AI will still keep a little playful badness.")

# Voice Maker / Editor area
if edit_voices:
    st.markdown("---")
    st.header("Voice Maker — create or edit voice scripts")
    st.write("Create new voices or edit existing ones. Each voice contains phrase templates (use {label} where the AI's guess should go).")
    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("Existing voices")
        sel = st.selectbox("Select voice to edit", options=voice_names_sorted, index=0 if voice_names_sorted else -1, key="edit_select")
        if sel and sel in voices:
            v = voices[sel]
            st.text_input("Voice name", value=sel, key="edit_name")
            templates_text = "\n".join(v.get("templates", []))
            templates_in = st.text_area("Phrase templates (one per line). Use {label} where the guess goes.", value=templates_text, height=150, key="edit_templates")
            py_rate_in = st.number_input("pyttsx3 rate (optional)", value=v.get("py_rate", 150), min_value=50, max_value=400, step=10, key="edit_rate")
            st.write("Preview and apply changes below.")
            if st.button("Save changes to voice", key="save_voice"):
                new_name = st.session_state.get("edit_name", sel).strip() or sel
                new_templates = [line.strip() for line in templates_in.splitlines() if line.strip()]
                voices.pop(sel, None)  # remove old key if renaming
                voices[new_name] = {
                    "templates": new_templates,
                    "py_rate": int(py_rate_in),
                    "py_voice_index": v.get("py_voice_index")
                }
                save_voices(voices)
                st.success(f"Saved voice '{new_name}'.")
                # refresh lists
                voice_names = list(voices.keys())
                voice_names_sorted = sorted(voice_names)
            if st.button("Delete this voice", key="delete_voice"):
                voices.pop(sel, None)
                save_voices(voices)
                st.success(f"Deleted voice '{sel}'.")
                voice_names = list(voices.keys())
                voice_names_sorted = sorted(voice_names)
    with cols[1]:
        st.subheader("Create new voice")
        new_name = st.text_input("New voice name", key="new_name")
        new_templates = st.text_area("New voice templates (one per line). Use {label} where the guess goes.", height=150, key="new_templates")
        new_py_rate = st.number_input("pyttsx3 rate (optional)", value=150, min_value=50, max_value=400, step=10, key="new_rate")
        if st.button("Create voice", key="create_voice"):
            name = new_name.strip() or f"Voice_{len(voices)+1}"
            templates = [line.strip() for line in new_templates.splitlines() if line.strip()]
            if not templates:
                st.error("Add at least one template line.")
            else:
                if name in voices:
                    st.error("A voice with that name already exists. Choose a different name.")
                else:
                    voices[name] = {"templates": templates, "py_rate": int(new_py_rate), "py_voice_index": None}
                    save_voices(voices)
                    st.success(f"Created voice '{name}'.")
                    voice_names = list(voices.keys())
                    voice_names_sorted = sorted(voice_names)
    st.markdown("## Quick preview")
    preview_voice = st.selectbox("Pick voice to preview", options=sorted(list(voices.keys())), index=0)
    preview_label = st.text_input("Label to use in preview", value="cat", key="preview_label")
    if st.button("Play preview", key="play_preview"):
        pv = voices.get(preview_voice, None)
        if not pv:
            st.error("Voice not found.")
        else:
            top_label = preview_label
            tmpl = random.choice(pv.get("templates", [f"I think this is {top_label}."]))
            phrase = tmpl.format(label=top_label)
            audio_bytes = None
            audio_format = None
            if tts_backend.startswith("pyttsx3") and TTS_PYTTSX3:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", pv.get("py_rate", 150))
                    voices_list = engine.getProperty("voices")
                    if pv.get("py_voice_index") is not None:
                        idx = pv.get("py_voice_index")
                        if isinstance(idx, int) and 0 <= idx < len(voices_list):
                            try:
                                engine.setProperty("voice", voices_list[idx].id)
                            except Exception:
                                pass
                    else:
                        try:
                            idx = abs(hash(preview_voice)) % len(voices_list)
                            engine.setProperty("voice", voices_list[idx].id)
                        except Exception:
                            pass
                    tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmpf.close()
                    engine.save_to_file(phrase, tmpf.name)
                    engine.runAndWait()
                    with open(tmpf.name, "rb") as f:
                        audio_bytes = f.read()
                        audio_format = "audio/wav"
                    try:
                        os.remove(tmpf.name)
                    except Exception:
                        pass
                except Exception as e:
                    st.warning(f"pyttsx3 TTS failed: {e}")
                    audio_bytes = None
            if audio_bytes is None and tts_backend.startswith("gTTS") and TTS_GTTS:
                try:
                    tts = gTTS(text=phrase, lang="en")
                    buf = io.BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    audio_bytes = buf.read()
                    audio_format = "audio/mp3"
                except Exception as e:
                    st.warning(f"gTTS failed: {e}")
                    audio_bytes = None
            if audio_bytes:
                st.audio(audio_bytes, format=audio_format)
            else:
                st.info("No TTS backend available or generation failed.")

st.markdown("---")
st.header("Developer / Files")
st.write("Voices are stored in `voices.json`. User-labeled images are stored under `user_data/<label>/`. The fine-tuned model is saved to `user_model.h5` and class list to `classes.json`.")
st.write("Requirements: see requirements.txt. Run with `streamlit run streamlit_app.py`.")
