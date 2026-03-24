"""
Stable Money Avatar Agent — Self-Hosted Backend
Replaces: ElevenLabs (Coqui XTTS v2) + HeyGen (Wav2Lip)
Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio, base64, io, os, sys, tempfile, time, uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import soundfile as sf

# ── Coqui TTS ──
from TTS.api import TTS

# ── Wav2Lip (optional — requires GPU server) ──
WAV2LIP_AVAILABLE = False
try:
    sys.path.insert(0, "./Wav2Lip")
    import audio as wav2lip_audio
    from models import Wav2Lip as Wav2LipModel
    import face_detection
    WAV2LIP_AVAILABLE = True
    print("[server] Wav2Lip available ✓")
except ImportError:
    print("[server] Wav2Lip not available — avatar disabled, voice-only mode")

app = FastAPI(title="Stable Money Avatar Server")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Config ──
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE  = 24000
AVATAR_IMG   = "./avatar.jpg"          # uploaded face image
WAV2LIP_CKPT = "./checkpoints/wav2lip_gan.pth"
VOICE_SAMPLE = "./voice_sample.wav"    # 6s sample for voice cloning

print(f"[server] Using device: {DEVICE}")

# ════════════════════════════════════════
# 1. LOAD MODELS AT STARTUP
# ════════════════════════════════════════

class Models:
    tts: Optional[TTS] = None
    wav2lip: Optional[object] = None
    face_frames: Optional[list] = None   # pre-processed face frames
    detector = None

models = Models()

@app.on_event("startup")
async def load_models():
    print("[startup] Loading Coqui XTTS v2…")

    _torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    models.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(DEVICE)
    print("[startup] XTTS v2 loaded ✓")

    if Path(WAV2LIP_CKPT).exists():
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
        print("[startup] Wav2Lip checkpoint not found — avatar disabled")

    print("[startup] All models ready ✓")


# ════════════════════════════════════════
# 2. TTS — YOUR OWN ELEVENLABS
# ════════════════════════════════════════

def synthesize_speech(text: str) -> bytes:
    """Convert text → WAV bytes using Coqui XTTS v2"""
    buf = io.BytesIO()
    models.tts.tts_to_file(text=text, file_path=buf)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════
# 3. AVATAR — YOUR OWN HEYGEN
# ════════════════════════════════════════

IMG_SIZE = 96   # Wav2Lip input size

async def preprocess_face(img_path: str):
    """Pre-process face image for Wav2Lip — run once"""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 480))
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

        # Write to MP4
        h, w = gen_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )
        for f in gen_frames:
            writer.write(f)
        writer.release()

        # Mux audio into video using ffmpeg
        muxed = out_path.replace(".mp4", "_final.mp4")
        os.system(
            f"ffmpeg -y -i {out_path} -i {audio_path} "
            f"-c:v copy -c:a aac -shortest {muxed} -loglevel quiet"
        )

        with open(muxed, "rb") as vf:
            return vf.read()

    finally:
        for p in [audio_path, out_path, out_path.replace(".mp4","_final.mp4")]:
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

            prompt = f"""You are Aayush, a warm friendly financial advisor for Stable Money — India's leading fixed-income investment platform.

Key facts:
- Stable Money offers FDs from 25+ NBFCs and small finance banks with 8-9.5% returns
- Much better than regular bank FDs which give 6.5-7%
- Sukoon: a no-lock-in idle money product earning 7-8% — better than savings accounts
- Minimum investment: Rs 1000
- DICGC insured up to Rs 5 lakhs for bank FDs
- Target: urban Indian professionals aged 25-45

Answer in MAX 2 short sentences only. No bullet points. No links. Be warm and conversational.
User question: {text}
"""

            try:
                import urllib.request, json
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=json.dumps({
                        "model": "llama3.2:3b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": 80, "temperature": 0.7}
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                answer = data.get("response", "").strip() or "Sorry, I could not generate a response."
            except Exception as e:
                answer = f"Sorry, Ollama failed: {str(e)}"

            print(f"[ws] answer ready: {answer[:120]}")
            await ws.send_json({"type": "text_chunk", "text": answer})
            await ws.send_json({"type": "status", "msg": "synthesizing"})

            try:
                audio_bytes = synthesize_speech(answer)
                await ws.send_json({
                    "type": "audio",
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    "format": "wav"
                })
            except Exception as e:
                await ws.send_json({"type": "error", "msg": f"TTS failed: {str(e)}"})
                continue

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

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "tts_loaded": models.tts is not None,
        "wav2lip_loaded": models.wav2lip is not None,
        "face_ready": models.face_frames is not None,
        "voice_clone": Path(VOICE_SAMPLE).exists()
    }

# ── Static frontend ──
if Path("./static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")