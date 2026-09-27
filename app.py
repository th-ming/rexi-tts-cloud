import io
import os
import tempfile
import threading
import wave

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from vieneu import Vieneu

app = FastAPI(title="VieNeu TTS API")

_tts = None
_lock = threading.Lock()


@app.on_event("startup")
def _load():
    global _tts
    _tts = Vieneu()


class SpeechRequest(BaseModel):
    model: str = "vieneu-v3-turbo"
    voice: str | None = None
    input: str
    response_format: str = "wav"
    sample_rate: int | None = None


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": v, "name": v} for _, v in _tts.list_preset_voices()]
    return {"object": "list", "data": voices}


@app.get("/health")
def health():
    return {"ok": _tts is not None}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="input is empty")
    voice = req.voice or None
    with _lock:
        try:
            audio = _tts.infer(req.input.strip(), voice=voice)
        except ValueError:
            audio = _tts.infer(req.input.strip(), voice=None)
    buf = _to_wav(audio, req.sample_rate)
    return Response(content=buf, media_type="audio/wav")


@app.post("/v1/clone")
async def clone(file: UploadFile = File(...), text: str = Form(...), ref_text: str = Form(None)):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="audio file is empty")
    suffix = os.path.splitext(file.filename or "ref.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        with _lock:
            audio = _tts.infer(text.strip(), ref_audio=tmp.name)
        return Response(content=_to_wav(audio, None), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"clone failed: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _to_wav(audio, sample_rate: int | None) -> bytes:
    import numpy as np

    data = audio if isinstance(audio, np.ndarray) else np.asarray(audio)
    if data.ndim > 1:
        data = data.reshape(-1)
    if sample_rate is None:
        sample_rate = int(getattr(_tts, "sample_rate", 48000) or 48000)
    if data.dtype != np.int16:
        peak = float(np.max(np.abs(data))) or 1.0
        data = (data / peak * 32000.0).astype(np.int16)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data.tobytes())
    return out.getvalue()
