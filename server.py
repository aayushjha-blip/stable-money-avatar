"""
Stable Money Avatar Agent — Self-Hosted Backend
TTS: edge-tts (Microsoft Neural) | LLM: Groq | Avatar: Wav2Lip (GPU)
Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio, base64, io, os, sys, tempfile, time, uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Wav2Lip (optional — requires GPU server) ──
WAV2LIP_AVAILABLE = False
try:
    # Try multiple paths (local ./Wav2Lip or Colab /content/Wav2Lip)
    for wp in ["./Wav2Lip", "/content/Wav2Lip"]:
        if Path(wp).exists():
            sys.path.insert(0, wp)
    import audio as wav2lip_audio
    from models import Wav2Lip as Wav2LipModel
    import face_detection
    import cv2
    WAV2LIP_AVAILABLE = True
    print("[server] Wav2Lip available ✓")
except ImportError as e:
    print(f"[server] Wav2Lip not available — voice-only mode ({e})")
    try:
        import cv2
    except ImportError:
        pass

app = FastAPI(title="Stable Money Avatar Server")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Config ──
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
AVATAR_IMG   = "./avatar.jpg"          # uploaded face image

# Search multiple paths for Wav2Lip checkpoint
WAV2LIP_CKPT = None
for ckpt_path in [
    "./checkpoints/wav2lip_gan.pth",
    "/content/checkpoints/wav2lip_gan.pth",
    "/content/Wav2Lip/checkpoints/wav2lip_gan.pth",
]:
    if Path(ckpt_path).exists() and os.path.getsize(ckpt_path) > 1_000_000:
        WAV2LIP_CKPT = ckpt_path
        break

print(f"[server] Using device: {DEVICE}")
print(f"[server] Wav2Lip checkpoint: {WAV2LIP_CKPT or 'not found'}")

# ════════════════════════════════════════
# 1. LOAD MODELS AT STARTUP
# ════════════════════════════════════════

class Models:
    wav2lip: Optional[object] = None
    face_frames: Optional[list] = None   # pre-processed face frames
    detector = None

models = Models()

@app.on_event("startup")
async def load_models():
    # Patch torch.load for older checkpoints
    _torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    if WAV2LIP_AVAILABLE and WAV2LIP_CKPT:
        print("[startup] Loading Wav2Lip…")
        ckpt = torch.load(WAV2LIP_CKPT, map_location=DEVICE)
        s = ckpt["state_dict"]
        # strip 'module.' prefix if present
        s = {k.replace("module.", ""): v for k, v in s.items()}
        models.wav2lip = Wav2LipModel()
        models.wav2lip.load_state_dict(s)
        models.wav2lip = models.wav2lip.to(DEVICE).eval()
        print("[startup] Wav2Lip loaded ✓")

        models.detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False, device=DEVICE)

        if Path(AVATAR_IMG).exists():
            await preprocess_face(AVATAR_IMG)
    else:
        print("[startup] Wav2Lip not available — audio-only mode")

    print("[startup] All models ready ✓")


# ════════════════════════════════════════
# 2. TTS — YOUR OWN ELEVENLABS
# ════════════════════════════════════════

ACRONYMS = {
    "DICGC": "D I C G C",
    "NBFC": "N B F C",
    "NBFCs": "N B F Cs",
    "SEBI": "S E B I",
    "RBI": "R B I",
    # FD/FDs intentionally excluded — LLM already writes "Fixed Deposits (FDs)"
    # so replacing "FDs" would create "Fixed Deposits (fixed deposits)" double-read
    "SFB": "small finance bank",
    "SFBs": "small finance banks",
}

def preprocess_tts(text: str) -> str:
    import re
    # Step 1: "ACRONYM (Full Form)" → keep only Full Form, drop the redundant acronym
    # e.g. "RBI (Reserve Bank of India)" → "Reserve Bank of India"
    # e.g. "NBFCs (Non-Banking Financial Companies)" → "Non-Banking Financial Companies"
    text = re.sub(r'\b[A-Z]{2,7}s?\s*\(([^)]+)\)', r'\1', text)
    # Step 2: expand any remaining standalone acronyms
    for acronym, expansion in ACRONYMS.items():
        text = re.sub(r'\b' + re.escape(acronym) + r'\b', expansion, text)
    return text

def _detect_hindi(text: str) -> bool:
    """Return True if text contains Devanagari characters (Hindi)."""
    import re
    return bool(re.search(r'[\u0900-\u097F]', text))

# ── Knowledge Base ──
_KB_PATH = Path("./knowledge_base.txt")
def _load_kb() -> str:
    if _KB_PATH.exists():
        return _KB_PATH.read_text(encoding="utf-8")
    return ""

KNOWLEDGE_BASE = _load_kb()

async def synthesize_speech(text: str) -> tuple[bytes, str]:
    """Convert text → audio bytes. Returns (audio_bytes, format).
    Uses ElevenLabs if API key is set, falls back to edge-tts."""
    text = preprocess_tts(text)
    is_hindi = _detect_hindi(text)

    # Try ElevenLabs first
    eleven_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if eleven_key:
        try:
            from elevenlabs import ElevenLabs
            client = ElevenLabs(api_key=eleven_key)
            audio_iter = client.text_to_speech.convert(
                text=text,
                voice_id="pNInz6obpgDQGcFmaJgB",  # "Adam" — warm male voice
                model_id="eleven_turbo_v2_5",       # fastest model
                output_format="mp3_22050_32",       # small + fast
            )
            buf = io.BytesIO()
            for chunk in audio_iter:
                buf.write(chunk)
            buf.seek(0)
            print(f"[tts] ElevenLabs: {buf.getbuffer().nbytes} bytes")
            return buf.read(), "mp3"
        except Exception as e:
            print(f"[tts] ElevenLabs failed, falling back to edge-tts: {e}")

    # Fallback: edge-tts
    import edge_tts
    voice = "hi-IN-MadhurNeural" if is_hindi else "en-IN-PrabhatNeural"
    communicate = edge_tts.Communicate(text, voice=voice, rate="+5%", pitch="+0Hz")
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read(), "mp3"


# ════════════════════════════════════════
# 3. AVATAR — YOUR OWN HEYGEN
# ════════════════════════════════════════

IMG_SIZE = 96   # Wav2Lip input size

async def preprocess_face(img_path: str):
    """Pre-process face image for Wav2Lip — run once"""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (480, 360))
    # detect face bbox
    preds = models.detector.get_detections_for_batch(
        np.array([img[:, :, ::-1]])
    )
    if not preds or preds[0] is None:
        print("[avatar] No face detected in image!")
        return
    # store as repeated frames (Wav2Lip needs video-like input)
    models.face_frames = [img] * 25  # 25 still frames
    print(f"[avatar] Face pre-processed ✓")


def generate_lipsync_video(audio_bytes: bytes) -> Optional[bytes]:
    """
    Audio bytes → lip-synced MP4 bytes
    Core Wav2Lip inference pipeline
    """
    if models.wav2lip is None or models.face_frames is None:
        return None

    # Write audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    out_path = f"/tmp/{uuid.uuid4().hex}.mp4"
    raw_path = None
    final_path = None

    try:
        # Load & process mel spectrogram
        wav = wav2lip_audio.load_wav(audio_path, 16000)
        mel = wav2lip_audio.melspectrogram(wav)
        mel_chunks = []
        mel_step  = 16
        mel_idx   = 0
        fps       = 25
        mel_per_frame = mel.shape[1] / (len(wav) / 16000 * fps)

        for _ in models.face_frames:
            s = int(mel_idx)
            e = s + mel_step
            mel_chunks.append(mel[:, s:e])
            mel_idx += mel_per_frame

        # Extend face frames to match mel
        face_frames_extended = (models.face_frames * (
            (len(mel_chunks) // len(models.face_frames)) + 1
        ))[:len(mel_chunks)]

        # Wav2Lip inference in batches
        BATCH = 128
        gen_frames = []

        for i in range(0, len(mel_chunks), BATCH):
            img_batch  = np.array(face_frames_extended[i:i+BATCH])
            mel_batch  = np.array(mel_chunks[i:i+BATCH])

            # Prepare lower-half mask (Wav2Lip only animates mouth region)
            img_masked = img_batch.copy()
            img_masked[:, img_batch.shape[1]//2:] = 0

            img_batch_t = torch.FloatTensor(
                np.transpose(img_batch, (0,3,1,2))).to(DEVICE) / 255.
            img_masked_t = torch.FloatTensor(
                np.transpose(img_masked, (0,3,1,2))).to(DEVICE) / 255.
            mel_batch_t = torch.FloatTensor(
                np.transpose(mel_batch, (0,3,1,2))).unsqueeze(1).to(DEVICE)

            with torch.no_grad():
                pred = models.wav2lip(mel_batch_t, img_masked_t)

            pred = (pred.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)

            for p, orig in zip(pred, img_batch):
                # Blend generated mouth region back into original frame
                h, w = orig.shape[:2]
                p_resized = cv2.resize(p, (w, h))
                combined  = orig.copy()
                combined[h//2:] = p_resized[h//2:]
                gen_frames.append(combined)

        # Encode to browser-compatible H.264 MP4 using ffmpeg pipe
        h, w = gen_frames[0].shape[:2]
        raw_path = f"/tmp/{uuid.uuid4().hex}.raw"
        with open(raw_path, "wb") as rf:
            for frame in gen_frames:
                rf.write(frame.tobytes())

        final_path = out_path.replace(".mp4", "_h264.mp4")
        os.system(
            f"ffmpeg -y -f rawvideo -vcodec rawvideo -s {w}x{h} "
            f"-pix_fmt bgr24 -r {fps} -i {raw_path} -i {audio_path} "
            f"-c:v libx264 -preset ultrafast -pix_fmt yuv420p "
            f"-c:a aac -shortest {final_path} -loglevel quiet"
        )

        with open(final_path, "rb") as vf:
            return vf.read()

    finally:
        for p in [audio_path, out_path, raw_path, final_path]:
            if p:
                try: os.remove(p)
                except: pass


# ════════════════════════════════════════
# 4. WEBSOCKET — REAL-TIME PIPELINE
# ════════════════════════════════════════

@app.websocket("/ws/talk")
async def ws_talk(ws: WebSocket):
    """
    Minimal working WebSocket:
    receives text, returns text + audio
    """
    await ws.accept()
    print("[ws] Client connected")

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "config":
                print("[ws] config received")
                continue

            if msg_type != "speak":
                continue

            text = data.get("text", "").strip()
            if not text:
                continue

            # Use explicit lang sent by frontend; fallback to Devanagari detection
            lang = data.get("lang") or ("hi" if _detect_hindi(text) else "en")

            if lang == "hi":
                system_msg = f"""आप StableAI हैं — Stable Money के वित्तीय सलाहकार। आप एक असली इंसान की तरह बात करते हैं।

KNOWLEDGE BASE:
{KNOWLEDGE_BASE}

नियम:
- सिर्फ हिंदी में जवाब दें।
- जैसे कोई दोस्त समझा रहा हो, वैसे बात करें — गर्मजोशी से, आसान भाषा में।
- सवाल का सीधा, पूरा और भरोसेमंद जवाब दें। 3-5 वाक्य में।
- Knowledge base के exact नंबर इस्तेमाल करें (8% to 9.5%, Rs 5 लाख insurance)।
- Acronym और full form दोनों एक साथ मत लिखें। सिर्फ full form लिखें।
- जवाब हमेशा पूरा करें।
- अगर जवाब knowledge base में नहीं है: "इस बारे में मैं team से confirm करके बताऊंगा।\""""
            else:
                system_msg = f"""You are StableAI, a friendly and knowledgeable financial advisor at Stable Money. You speak like a real person — warm, confident, and helpful. Think of yourself as a trusted friend who happens to be great with money.

KNOWLEDGE BASE (use ONLY this — never make up facts):
{KNOWLEDGE_BASE}

RULES:
- Answer ONLY in English.
- Speak naturally, like you're having a conversation — not reading a script.
- Give complete, helpful answers in 3-5 sentences. Include specific numbers and details from the knowledge base.
- Use exact figures: "8% to 9.5%", "Rs 5 lakhs insurance cover", "25+ partner institutions".
- Never write acronym and full form together like "RBI (Reserve Bank of India)" — just use the full form.
- If someone seems new to investing, be encouraging and explain simply.
- Always complete your answer — never cut off mid-sentence.
- If not in knowledge base: "Let me check the latest details on that and get back to you.\""""

            try:
                from groq import Groq
                client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=300,
                    temperature=0.6,
                )
                answer = completion.choices[0].message.content.strip().strip('"') or "Sorry, I could not generate a response."
            except Exception as e:
                answer = f"Sorry, Groq failed: {str(e)}"

            print(f"[ws] answer ready: {answer[:120]}")
            await ws.send_json({"type": "text_chunk", "text": answer})
            await ws.send_json({"type": "status", "msg": "synthesizing"})

            try:
                audio_bytes, audio_fmt = await synthesize_speech(answer)
            except Exception as e:
                await ws.send_json({"type": "error", "msg": f"TTS failed: {str(e)}"})
                continue

            # Audio-first: send audio immediately so user hears response fast
            await ws.send_json({
                "type": "audio",
                "data": base64.b64encode(audio_bytes).decode("utf-8"),
                "format": audio_fmt
            })

            # If Wav2Lip available (GPU), generate lip-synced video in parallel
            if models.wav2lip is not None and models.face_frames is not None:
                try:
                    await ws.send_json({"type": "status", "msg": "animating"})
                    # Convert audio to WAV 16kHz mono for Wav2Lip
                    wav_path = f"/tmp/{uuid.uuid4().hex}.wav"
                    src_path = f"/tmp/{uuid.uuid4().hex}.{audio_fmt}"
                    with open(src_path, "wb") as f:
                        f.write(audio_bytes)
                    os.system(f"ffmpeg -y -i {src_path} -ar 16000 -ac 1 {wav_path} -loglevel quiet")
                    with open(wav_path, "rb") as f:
                        wav_bytes = f.read()
                    for p in [src_path, wav_path]:
                        try: os.remove(p)
                        except: pass
                    video_bytes = await asyncio.get_event_loop().run_in_executor(
                        None, generate_lipsync_video, wav_bytes)
                    if video_bytes:
                        await ws.send_json({
                            "type": "video",
                            "data": base64.b64encode(video_bytes).decode("utf-8"),
                            "format": "mp4"
                        })
                except Exception as e:
                    print(f"[ws] Wav2Lip failed: {e}")

            await ws.send_json({"type": "done"})
            print("[ws] Ready for next question")

    except WebSocketDisconnect:
        print("[ws] Client disconnected")
    except Exception as e:
        print("[ws] Error:", e)
        try:
            await ws.send_json({"type": "error", "msg": str(e)})
        except:
            pass


@app.post("/upload/avatar")
async def upload_avatar(file: UploadFile = File(...)):
    contents = await file.read()
    with open(AVATAR_IMG, "wb") as f:
        f.write(contents)
    return {"status": "ok", "message": "Avatar photo saved"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "tts": "edge-tts",
        "wav2lip_loaded": models.wav2lip is not None,
        "face_ready": models.face_frames is not None,
    }

# ── Static frontend ──
if Path("./static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")