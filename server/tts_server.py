import gc
import glob
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import psutil
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# ── config ─────────────────────────────────────────────────────────────────────

MISOTTS_DIR  = "/home/pineway/misotts"
VOICES_DIR   = os.path.join(MISOTTS_DIR, "voices")
OUTPUTS_DIR  = os.path.join(MISOTTS_DIR, "outputs")
QUEUE_FILE   = os.path.join(MISOTTS_DIR, "queue.json")
MODEL_REPO   = "MisoLabs/MisoTTS"
MODEL_FILE   = "model.safetensors"
EXPECTED_MODEL_BYTES = 16_100_000_000
IDLE_TIMEOUT = 30 * 60  # seconds

TONE_PRESETS = {
    "Conversational": "This is a warm, casual conversation between friends.",
    "Instructional":  "This is clear, patient instructional narration for a tutorial.",
    "Voiceover":      "This is a professional, authoritative voiceover presentation.",
    "Dramatic":       "This is an emotionally charged, dramatic and expressive reading.",
}

LIBGOMP = os.path.join(
    MISOTTS_DIR,
    ".venv/lib/python3.10/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0",
)
if os.path.exists(LIBGOMP) and "LD_PRELOAD" not in os.environ:
    os.environ["LD_PRELOAD"] = LIBGOMP

sys.path.insert(0, MISOTTS_DIR)
os.makedirs(VOICES_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── logging ────────────────────────────────────────────────────────────────────

LOG_FILE     = os.path.join(MISOTTS_DIR, "tts_server.log")
LOG_FMT      = "%(asctime)s %(levelname)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%dT%H:%M:%S"
_LOG_RE      = __import__("re").compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}) (DEBUG|INFO|WARNING|ERROR|CRITICAL) (.*)$"
)

log = logging.getLogger("misotts")
log.setLevel(logging.DEBUG)

# stdout → journalctl
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE_FMT))
log.addHandler(_sh)

# rotating file — 5 MB × 5 = 25 MB max
from logging.handlers import RotatingFileHandler as _RFH
_fh = _RFH(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
_fh.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE_FMT))
log.addHandler(_fh)

# ── state ──────────────────────────────────────────────────────────────────────

_mstate: dict = {
    "status":       "unloaded",   # unloaded | loading | ready | busy
    "error":        None,
    "progress":     0,
    "progress_msg": "",
    "idle_until":   None,         # epoch float: when auto-unload fires
    "generator":    None,
}

_queue: list          = []
_queue_lock           = threading.Lock()
_new_job_event        = threading.Event()
_idle_timer: Optional[threading.Timer] = None
_idle_timer_lock      = threading.Lock()
_executor             = ThreadPoolExecutor(max_workers=1)

app = FastAPI()

# ── queue persistence ──────────────────────────────────────────────────────────

def _save_queue() -> None:
    with open(QUEUE_FILE, "w") as f:
        json.dump(_queue, f, indent=2, default=str)


def _load_queue_from_disk() -> None:
    global _queue
    if not os.path.exists(QUEUE_FILE):
        return
    with open(QUEUE_FILE) as f:
        data = json.load(f)
    reset = 0
    for job in data:
        if job.get("status") == "processing":
            job["status"]     = "pending"
            job["started_at"] = None
            reset += 1
    _queue = data
    log.info(f"Loaded {len(data)} job(s) from disk ({reset} reset from processing→pending)")

# ── idle timer ─────────────────────────────────────────────────────────────────

def _start_idle_timer() -> None:
    global _idle_timer
    with _idle_timer_lock:
        if _idle_timer and _idle_timer.is_alive():
            return
        _mstate["idle_until"] = time.time() + IDLE_TIMEOUT
        _idle_timer = threading.Timer(IDLE_TIMEOUT, _auto_unload)
        _idle_timer.daemon = True
        _idle_timer.start()
        log.info(f"Idle timer started — auto-unload in {IDLE_TIMEOUT//60} min")


def _cancel_idle_timer() -> None:
    global _idle_timer
    with _idle_timer_lock:
        if _idle_timer:
            _idle_timer.cancel()
            _idle_timer = None
        _mstate["idle_until"] = None


def _auto_unload() -> None:
    log.info("Idle timeout reached — auto-unloading model")
    _cancel_idle_timer()
    _unload()

# ── model lifecycle ─────────────────────────────────────────────────────────────

def _load_blocking() -> None:
    _mstate.update(status="loading", error=None, progress=0, progress_msg="Starting…")
    t0 = time.time()
    log.info("Model load started")
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        # Phase 1 — download (0 → 50%)
        cached = try_to_load_from_cache(MODEL_REPO, MODEL_FILE)
        already_cached = cached is not None and os.path.exists(str(cached))

        if not already_cached:
            log.info("Model not in cache — downloading from HuggingFace (~16 GB)")
            _mstate["progress_msg"] = "Downloading model weights (~16 GB)…"
            dl_error: list = [None]
            dl_done = threading.Event()
            _last_dl_pct: list = [-1]

            def _do_dl():
                try:
                    hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
                except Exception as exc:
                    dl_error[0] = exc
                finally:
                    dl_done.set()

            threading.Thread(target=_do_dl, daemon=True).start()
            blob_dir = os.path.expanduser(
                "~/.cache/huggingface/hub/models--"
                + MODEL_REPO.replace("/", "--") + "/blobs"
            )
            while not dl_done.is_set():
                try:
                    if os.path.isdir(blob_dir):
                        total = sum(
                            os.path.getsize(f)
                            for f in glob.glob(os.path.join(blob_dir, "*"))
                            if os.path.isfile(f)
                        )
                        if total > 0:
                            pct = min(49, int(50 * total / EXPECTED_MODEL_BYTES))
                            _mstate["progress"] = pct
                            _mstate["progress_msg"] = (
                                f"Downloading: {total/1_048_576:,.0f} / ~16,100 MB"
                            )
                            milestone = (pct // 10) * 10
                            if milestone > _last_dl_pct[0]:
                                _last_dl_pct[0] = milestone
                                log.info(f"Download {milestone}% ({total/1_048_576:,.0f} MB)")
                except Exception:
                    pass
                dl_done.wait(timeout=0.5)
            if dl_error[0]:
                raise dl_error[0]
            log.info(f"Download complete ({(time.time()-t0)/60:.1f} min)")
        else:
            log.info("Model found in HuggingFace cache — skipping download")
            _mstate.update(progress=50, progress_msg="Found in cache, loading into memory…")

        # Phase 2 — memory load (50 → 100%)
        log.info("Loading model into memory (~16 GB)")
        _mstate.update(progress=50, progress_msg="Loading model into memory (~16 GB)…")
        start_ram = psutil.virtual_memory().used
        load_done = threading.Event()

        def _monitor_ram():
            while not load_done.is_set():
                try:
                    delta = max(0, psutil.virtual_memory().used - start_ram)
                    pct = 50 + min(45, int(45 * delta / EXPECTED_MODEL_BYTES))
                    _mstate["progress"] = max(_mstate["progress"], pct)
                    _mstate["progress_msg"] = (
                        f"Loading into memory: {delta/1024**3:.1f} / ~16 GB"
                    )
                except Exception:
                    pass
                load_done.wait(timeout=0.5)

        threading.Thread(target=_monitor_ram, daemon=True).start()
        from generator import load_miso_8b
        _mstate["generator"] = load_miso_8b(device="cpu")
        load_done.set()
        elapsed = time.time() - t0
        log.info(f"Model ready — total load time {elapsed/60:.1f} min")
        _mstate.update(status="ready", progress=100, progress_msg="Ready")

    except Exception as exc:
        log.error(f"Model load failed: {exc}")
        _mstate.update(status="unloaded", error=str(exc), progress=0, progress_msg="")


def _unload() -> None:
    import torch
    _mstate["generator"] = None
    _mstate.update(status="unloaded", error=None, progress=0, progress_msg="", idle_until=None)
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Model unloaded, memory freed")

# ── job processing ──────────────────────────────────────────────────────────────

def _process_job(job: dict) -> None:
    import torch
    from generator import Segment

    job["status"]     = "processing"
    job["started_at"] = time.time()
    _mstate["status"] = "busy"
    with _queue_lock:
        _save_queue()

    label = job.get("label") or job["id"]
    log.info(
        f"Job {job['id']} started — label={label!r} speaker={job.get('speaker',0)} "
        f"voice={job.get('voice_name') or 'none'} tone={job.get('tone_preset') or 'none'} "
        f"text_len={len(job['text'])} chars"
    )

    try:
        gen     = _mstate["generator"]
        context = []

        # Build preamble: tone preset text + optional custom preamble
        parts = []
        if job.get("tone_preset") and job["tone_preset"] in TONE_PRESETS:
            parts.append(TONE_PRESETS[job["tone_preset"]])
        if job.get("preamble"):
            parts.append(job["preamble"].strip())
        full_preamble = " ".join(parts).strip()

        if full_preamble:
            log.debug(f"Job {job['id']} preamble context: {full_preamble[:80]!r}")
            # 0.5 s of silence as placeholder audio — primes tone from text
            context.append(Segment(
                speaker=job.get("speaker", 0),
                text=full_preamble,
                audio=torch.zeros(12000),
            ))

        # Voice clone
        if job.get("voice_name"):
            wav_p = os.path.join(VOICES_DIR, f"{job['voice_name']}.wav")
            txt_p = os.path.join(VOICES_DIR, f"{job['voice_name']}.txt")
            if os.path.exists(wav_p):
                log.debug(f"Job {job['id']} loading voice clone: {job['voice_name']}")
                wf, sr = torchaudio.load(wav_p)
                if sr != gen.sample_rate:
                    wf = torchaudio.functional.resample(wf, sr, gen.sample_rate)
                transcript = open(txt_p).read().strip() if os.path.exists(txt_p) else ""
                context.append(Segment(
                    speaker=job.get("speaker", 0),
                    text=transcript,
                    audio=wf[0],
                ))
            else:
                log.warning(f"Job {job['id']} voice {job['voice_name']!r} WAV not found — skipping")

        log.info(f"Job {job['id']} synthesis starting ({len(context)} context segment(s))")
        audio = gen.generate(
            text=job["text"],
            speaker=job.get("speaker", 0),
            context=context,
            max_audio_length_ms=job.get("max_audio_length_ms", 30_000),
        )

        out_path = os.path.join(OUTPUTS_DIR, f"{job['id']}.wav")
        torchaudio.save(out_path, audio.unsqueeze(0).cpu(), gen.sample_rate, format="wav")
        out_size_kb = round(os.path.getsize(out_path) / 1024)

        duration = round(time.time() - job["started_at"], 1)
        job.update(
            status="completed",
            completed_at=time.time(),
            duration_sec=duration,
            output_file=f"{job['id']}.wav",
        )
        log.info(f"Job {job['id']} completed in {duration}s — output {out_size_kb} KB")

    except Exception as exc:
        job.update(status="failed", completed_at=time.time(), error=str(exc))
        log.error(f"Job {job['id']} failed: {exc}")

    finally:
        _mstate["status"] = "ready"
        with _queue_lock:
            _save_queue()

# ── worker thread ───────────────────────────────────────────────────────────────

def _worker() -> None:
    log.info("Worker thread started")
    while True:
        _new_job_event.wait(timeout=5)
        _new_job_event.clear()

        with _queue_lock:
            pending = [j for j in _queue if j["status"] == "pending"]

        if not pending:
            if _mstate["status"] == "ready":
                _start_idle_timer()
            continue

        log.info(f"Worker found {len(pending)} pending job(s)")
        _cancel_idle_timer()

        if _mstate["status"] == "unloaded":
            _load_blocking()

        if _mstate["status"] != "ready":
            continue

        with _queue_lock:
            pending = [j for j in _queue if j["status"] == "pending"]
        if pending:
            _process_job(pending[0])
            _new_job_event.set()   # check immediately for more

# ── startup ─────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    log.info("MisoTTS server starting up")
    _load_queue_from_disk()
    threading.Thread(target=_worker, daemon=True).start()
    _new_job_event.set()

# ── API ─────────────────────────────────────────────────────────────────────────

@app.get("/status")
async def get_status():
    idle = _mstate["idle_until"]
    return {
        "status":           _mstate["status"],
        "error":            _mstate["error"],
        "progress":         _mstate["progress"],
        "progress_msg":     _mstate["progress_msg"],
        "idle_remaining":   max(0, int(idle - time.time())) if idle else None,
    }


@app.get("/queue")
async def get_queue():
    with _queue_lock:
        return list(reversed(_queue))


class EnqueueReq(BaseModel):
    text:                str
    label:               Optional[str]   = None
    preamble:            Optional[str]   = None
    tone_preset:         Optional[str]   = None
    speaker:             int             = 0
    voice_name:          Optional[str]   = None
    max_audio_length_ms: float           = 30_000


@app.post("/queue")
async def enqueue(req: EnqueueReq):
    if not req.text.strip():
        raise HTTPException(400, "text is required")
    if req.tone_preset and req.tone_preset not in TONE_PRESETS:
        raise HTTPException(400, f"Unknown tone preset: {req.tone_preset}")
    job = {
        "id":                   uuid.uuid4().hex[:12],
        "label":                req.label or None,
        "text":                 req.text.strip(),
        "preamble":             req.preamble or None,
        "tone_preset":          req.tone_preset or None,
        "speaker":              req.speaker,
        "voice_name":           req.voice_name or None,
        "max_audio_length_ms":  req.max_audio_length_ms,
        "status":               "pending",
        "created_at":           time.time(),
        "started_at":           None,
        "completed_at":         None,
        "duration_sec":         None,
        "error":                None,
        "output_file":          None,
    }
    with _queue_lock:
        _queue.append(job)
        _save_queue()
    _cancel_idle_timer()
    _new_job_event.set()
    log.info(f"Job {job['id']} enqueued — label={req.label!r} text={req.text[:60]!r}")
    return job


@app.delete("/queue/{job_id}")
async def delete_job(job_id: str):
    with _queue_lock:
        job = next((j for j in _queue if j["id"] == job_id), None)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] == "processing":
            raise HTTPException(400, "Cannot delete a running job")
        if job.get("output_file"):
            p = os.path.join(OUTPUTS_DIR, job["output_file"])
            if os.path.exists(p):
                os.remove(p)
        _queue.remove(job)
        _save_queue()
    log.info(f"Job {job_id} cancelled")
    return {"deleted": job_id}


@app.get("/outputs/{job_id}")
async def get_output(job_id: str):
    with _queue_lock:
        job = next((j for j in _queue if j["id"] == job_id), None)
    if not job or not job.get("output_file"):
        raise HTTPException(404, "Output not found")
    path = os.path.join(OUTPUTS_DIR, job["output_file"])
    if not os.path.exists(path):
        raise HTTPException(404, "Audio file missing")
    fname = f"{job['label'] or job['id']}.wav"
    return FileResponse(path, media_type="audio/wav", filename=fname)


@app.get("/voices")
async def list_voices():
    voices = []
    for wp in glob.glob(os.path.join(VOICES_DIR, "*.wav")):
        name = os.path.splitext(os.path.basename(wp))[0]
        tp = os.path.join(VOICES_DIR, f"{name}.txt")
        transcript = open(tp).read().strip() if os.path.exists(tp) else ""
        try:
            info = torchaudio.info(wp)
            dur  = round(info.num_frames / info.sample_rate, 1)
        except Exception:
            dur = None
        voices.append({"name": name, "transcript": transcript, "duration_sec": dur})
    return sorted(voices, key=lambda v: v["name"])


@app.post("/voices")
async def add_voice(
    name:       str        = Form(...),
    transcript: str        = Form(...),
    audio:      UploadFile = File(...),
):
    safe = "".join(c for c in name if c.isalnum() or c in "-_").strip()
    if not safe:
        raise HTTPException(400, "Invalid voice name")
    raw = await audio.read()
    try:
        wf, sr = torchaudio.load(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Cannot read audio: {exc}")
    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)
    if sr != 24_000:
        wf = torchaudio.functional.resample(wf, sr, 24_000)
    torchaudio.save(os.path.join(VOICES_DIR, f"{safe}.wav"), wf, 24_000)
    with open(os.path.join(VOICES_DIR, f"{safe}.txt"), "w") as f:
        f.write(transcript.strip())
    dur = round(wf.shape[1] / 24_000, 1)
    log.info(f"Voice {safe!r} added ({dur}s)")
    return {"name": safe, "duration_sec": dur}


@app.delete("/voices/{name}")
async def delete_voice(name: str):
    wp = os.path.join(VOICES_DIR, f"{name}.wav")
    if not os.path.exists(wp):
        raise HTTPException(404, "Voice not found")
    for p in [wp, os.path.join(VOICES_DIR, f"{name}.txt")]:
        if os.path.exists(p):
            os.remove(p)
    log.info(f"Voice {name!r} deleted")
    return {"deleted": name}


@app.get("/logs")
async def get_logs(lines: int = 500):
    """Return recent log lines parsed from the log file."""
    entries = []
    try:
        # Collect lines from the active log file (newest lines = tail)
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            raw = f.readlines()
        for line in raw[-lines:]:
            m = _LOG_RE.match(line.rstrip())
            if m:
                ts_str, level, msg = m.group(1), m.group(2), m.group(3)
                entries.append({"ts": ts_str, "level": level, "msg": msg})
    except FileNotFoundError:
        pass
    except Exception as exc:
        log.warning(f"Could not read log file: {exc}")
    return {"entries": entries, "log_file": LOG_FILE}


@app.delete("/logs")
async def clear_logs():
    """Truncate the log file (keeps it in place so the handler still works)."""
    try:
        open(LOG_FILE, "w").close()
        log.info("Log file cleared via HTTP")
    except Exception as exc:
        raise HTTPException(500, f"Could not clear log: {exc}")
    return {"cleared": True}


# ── HTML ────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MisoTTS Work Queue</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0a0d14;color:#e2e8f0;min-height:100vh;padding:1.5rem 2rem}
h1{font-size:1.4rem;font-weight:700}
.subtitle{color:#4a5568;font-size:.8rem;margin-bottom:1.75rem}
/* Cards */
.card{background:#141824;border:1px solid #1e2535;border-radius:10px;padding:1.25rem;margin-bottom:1.25rem}
.card-title{font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;color:#4a5568;margin-bottom:.9rem}
/* Model banner */
.model-banner{display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
.dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.dot.unloaded{background:#2d3748}.dot.loading{background:#ecc94b;animation:pulse 1s infinite}
.dot.ready{background:#48bb78}.dot.busy{background:#4299e1;animation:pulse .8s infinite}
.dot.error{background:#f56565}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.model-status-text{font-weight:600;font-size:.9rem;text-transform:capitalize}
.idle-badge{font-size:.75rem;color:#4a5568;margin-left:.25rem}
.model-error{color:#fc8181;font-size:.78rem;margin-top:.4rem;font-family:monospace;word-break:break-word}
.progress-wrap{width:100%;margin-top:.75rem}
.progress-track{background:#0a0d14;border-radius:4px;height:6px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,#4299e1,#63b3ed);border-radius:4px;transition:width .4s}
.progress-label{font-size:.72rem;color:#4a5568;margin-top:.3rem}
/* Buttons */
button{border:none;border-radius:7px;font-size:.8rem;font-weight:600;cursor:pointer;transition:opacity .15s;padding:.4rem .9rem}
button:disabled{opacity:.35;cursor:not-allowed}
button:hover:not(:disabled){opacity:.82}
.btn-blue{background:#4299e1;color:#fff}
.btn-gray{background:#1e2535;color:#cbd5e0;border:1px solid #2d3748}
.btn-red{background:#f56565;color:#fff}
.btn-green{background:#48bb78;color:#000}
.btn-icon{background:transparent;border:1px solid #2d3748;color:#94a3b8;padding:.3rem .5rem;font-size:.78rem;border-radius:6px}
.btn-icon:hover:not(:disabled){background:#1e2535;color:#e2e8f0}
/* Form fields */
input[type=text],textarea,select{width:100%;background:#0a0d14;border:1px solid #1e2535;border-radius:7px;color:#e2e8f0;padding:.55rem .7rem;font-size:.85rem}
textarea{resize:vertical}
input[type=text]:focus,textarea:focus,select:focus{outline:none;border-color:#4299e1}
input[type=file]{color:#94a3b8;font-size:.82rem}
label{font-size:.75rem;color:#64748b;display:block;margin-top:.65rem;margin-bottom:.2rem}
label:first-child{margin-top:0}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}
.btn-row{display:flex;gap:.6rem;margin-top:.9rem;flex-wrap:wrap}
/* New job form */
#new-job-form{display:none;background:#0a0d14;border:1px solid #1e2535;border-radius:9px;padding:1.1rem;margin-bottom:1rem}
/* Queue */
.queue-empty{color:#4a5568;font-size:.85rem;padding:.5rem 0}
.job-card{border:1px solid #1e2535;border-radius:8px;padding:.85rem 1rem;margin-bottom:.6rem;transition:border-color .2s}
.job-card.processing{border-color:#4299e1}
.job-card.completed{border-color:#2d4a3e}
.job-card.failed{border-color:#4a2020}
.job-header{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
.status-pip{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.status-pip.pending{background:#4a5568}
.status-pip.processing{background:#4299e1;animation:pulse .8s infinite}
.status-pip.completed{background:#48bb78}
.status-pip.failed{background:#f56565}
.job-label{font-weight:600;font-size:.88rem;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.badge{font-size:.68rem;padding:.15rem .45rem;border-radius:4px;font-weight:600;white-space:nowrap}
.badge.tone{background:#2a3a5c;color:#90cdf4}
.badge.voice{background:#2a3a2a;color:#9ae6b4}
.badge.status-pending{background:#2a2a10;color:#ecc94b}
.badge.status-processing{background:#1a2a3a;color:#90cdf4}
.badge.status-completed{background:#0d2a1d;color:#9ae6b4}
.badge.status-failed{background:#2a1010;color:#feb2b2}
.job-meta{font-size:.72rem;color:#4a5568;white-space:nowrap}
.job-actions{display:flex;gap:.35rem;margin-left:auto;flex-shrink:0}
.job-text-preview{font-size:.78rem;color:#64748b;margin-top:.45rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.job-error{font-size:.75rem;color:#fc8181;margin-top:.35rem;font-family:monospace;word-break:break-word}
.job-audio{margin-top:.65rem}
audio{width:100%;border-radius:6px}
.spinner{display:inline-block;width:12px;height:12px;border:2px solid #4299e1;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:5px}
@keyframes spin{to{transform:rotate(360deg)}}
.section-toggle{cursor:pointer;user-select:none;display:flex;align-items:center;gap:.5rem}
.section-toggle::after{content:'▾';font-size:.8rem;color:#4a5568;transition:transform .2s}
.section-toggle.collapsed::after{transform:rotate(-90deg)}
/* Voices */
.voices-list{display:flex;flex-direction:column;gap:.5rem;margin-bottom:.75rem}
.voice-card{background:#0a0d14;border:1px solid #1e2535;border-radius:7px;padding:.65rem .85rem;display:flex;align-items:center;gap:.75rem}
.voice-name{font-weight:600;font-size:.85rem}
.voice-sub{font-size:.72rem;color:#4a5568;margin-top:.1rem}
.voice-transcript{font-size:.72rem;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:300px}
#add-voice-form{display:none;background:#0a0d14;border:1px solid #1e2535;border-radius:8px;padding:.9rem;margin-bottom:.75rem}
.note{font-size:.72rem;color:#4a5568;margin-top:.4rem}
hr{border:none;border-top:1px solid #1e2535;margin:.75rem 0}
</style>
</head>
<body>
<h1>MisoTTS Work Queue</h1>
<p class="subtitle">Jetson AGX Orin · CPU inference · model auto-loads and unloads</p>

<!-- Model Status -->
<div class="card">
  <div class="card-title">Model</div>
  <div class="model-banner">
    <div class="dot" id="mdot"></div>
    <span class="model-status-text" id="mstatus">—</span>
    <span class="idle-badge" id="idle-badge"></span>
  </div>
  <div class="model-error" id="merror"></div>
  <div class="progress-wrap" id="progress-wrap" style="display:none">
    <div class="progress-track"><div class="progress-fill" id="pfill" style="width:0%"></div></div>
    <div class="progress-label" id="plabel"></div>
  </div>
</div>

<!-- New Work Order -->
<div style="margin-bottom:1rem">
  <button class="btn-blue" id="btn-new" onclick="toggleForm(true)">+ New Work Order</button>
</div>

<div id="new-job-form">
  <div style="font-size:.85rem;font-weight:700;margin-bottom:.9rem;color:#90cdf4">New Work Order</div>
  <label>Label <span style="color:#4a5568">(optional — for your reference)</span></label>
  <input type="text" id="f-label" placeholder="e.g. Morning announcement">
  <label>Tone Preset</label>
  <select id="f-tone">
    <option value="">— None (infer from text) —</option>
    <option value="Conversational">Conversational — warm, casual, friendly</option>
    <option value="Instructional">Instructional — clear, patient, tutorial</option>
    <option value="Voiceover">Voiceover — professional, authoritative</option>
    <option value="Dramatic">Dramatic — emotionally charged, expressive</option>
  </select>
  <label>Context / Preamble <span style="color:#4a5568">(optional — scene description to steer delivery)</span></label>
  <textarea id="f-preamble" rows="2" placeholder='e.g. "This is an excited sports announcer describing a last-second goal."'></textarea>
  <label>Text <span style="color:#e53e3e">*</span></label>
  <textarea id="f-text" rows="4" placeholder="Text to synthesize…"></textarea>
  <div class="row2">
    <div>
      <label>Speaker Style</label>
      <select id="f-speaker">
        <option value="0">Speaker 0</option><option value="1">Speaker 1</option>
        <option value="2">Speaker 2</option><option value="3">Speaker 3</option>
        <option value="4">Speaker 4</option><option value="5">Speaker 5</option>
      </select>
    </div>
    <div>
      <label>Voice Clone</label>
      <select id="f-voice"><option value="">— None —</option></select>
    </div>
  </div>
  <div class="btn-row">
    <button class="btn-blue" id="btn-enqueue" onclick="enqueue()">Enqueue</button>
    <button class="btn-gray" onclick="toggleForm(false)">Cancel</button>
  </div>
  <p class="note" id="enqueue-note"></p>
</div>

<!-- Queue -->
<div class="card">
  <div class="card-title">Queue</div>
  <div id="queue-list"><p class="queue-empty">No work orders yet.</p></div>
</div>

<!-- Voices -->
<div class="card" id="voices-card">
  <div class="card-title section-toggle" id="voices-toggle" onclick="toggleVoices()">Voices</div>
  <div id="voices-section">
    <div class="voices-list" id="voices-list"><p style="color:#4a5568;font-size:.82rem">No saved voices.</p></div>
    <div id="add-voice-form">
      <label>Voice Name</label>
      <input type="text" id="vname" placeholder="e.g. alice">
      <label>Transcript <span style="color:#4a5568">(exactly what is said in the clip)</span></label>
      <textarea id="vtranscript" rows="2"></textarea>
      <label>Audio File</label>
      <input type="file" id="vaudio" accept="audio/*">
      <div class="btn-row">
        <button class="btn-blue" onclick="saveVoice()">Save Voice</button>
        <button class="btn-gray" onclick="toggleAddVoice(false)">Cancel</button>
      </div>
      <p class="note" id="voice-note"></p>
    </div>
    <button class="btn-gray" id="btn-add-voice" onclick="toggleAddVoice(true)">+ Add Voice</button>
  </div>
</div>

<!-- Logs -->
<div class="card">
  <div class="card-title section-toggle" id="logs-toggle" onclick="toggleLogs()">Logs</div>
  <div id="logs-section">
    <div id="log-panel" style="background:#060810;font-family:monospace;font-size:.74rem;height:280px;overflow-y:auto;border-radius:6px;padding:.6rem .8rem;display:flex;flex-direction:column;gap:.1rem;line-height:1.5"></div>
    <div style="margin-top:.5rem;display:flex;align-items:center;gap:.75rem;flex-wrap:wrap">
      <button class="btn-gray" onclick="clearLogs()">Clear log</button>
      <span id="log-meta" style="font-size:.72rem;color:#4a5568"></span>
    </div>
  </div>
</div>

<script>
// ── state ─────────────────────────────────────────────────────────────────────
let pollTimer = null;
let _enqueueing = false;
let queuePollTimer = null;
let logPollTimer = null;
let voicesCollapsed = false;
let logsCollapsed = false;
const openAudioIds = new Set();  // job IDs with expanded audio player

// ── polling ───────────────────────────────────────────────────────────────────
function startPolling() {
  stopPolling();
  pollTimer      = setInterval(pollStatus, 2000);
  queuePollTimer = setInterval(loadQueue,  2500);
  logPollTimer   = setInterval(loadLogs,   3000);
}
function stopPolling() {
  clearInterval(pollTimer); clearInterval(queuePollTimer); clearInterval(logPollTimer);
}

async function pollStatus() {
  try {
    const d = await (await fetch('/status')).json();
    applyStatus(d);
  } catch(e) {}
}

function applyStatus(d) {
  document.getElementById('mdot').className = 'dot ' + (d.error ? 'error' : d.status);
  document.getElementById('mstatus').textContent = d.status;
  document.getElementById('merror').textContent = d.error || '';

  const idle = document.getElementById('idle-badge');
  if (d.idle_remaining != null && d.status === 'ready') {
    idle.textContent = `· unloads in ${fmtSec(d.idle_remaining)}`;
  } else {
    idle.textContent = '';
  }

  const pw = document.getElementById('progress-wrap');
  if (d.status === 'loading') {
    pw.style.display = 'block';
    document.getElementById('pfill').style.width = (d.progress || 0) + '%';
    document.getElementById('plabel').textContent = d.progress_msg || '';
  } else {
    pw.style.display = 'none';
  }
}

function fmtSec(s) {
  const m = Math.floor(s / 60), sec = s % 60;
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function fmtTs(epoch) {
  if (!epoch) return '';
  return new Date(epoch * 1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
}

// ── queue ─────────────────────────────────────────────────────────────────────
async function loadQueue() {
  try {
    const jobs = await (await fetch('/queue')).json();
    renderQueue(jobs);
  } catch(e) {}
}

function renderQueue(jobs) {
  const el = document.getElementById('queue-list');
  if (!jobs.length) { el.innerHTML = '<p class="queue-empty">No work orders yet.</p>'; return; }

  el.innerHTML = jobs.map(j => {
    const label = esc(j.label || j.text.slice(0, 60) + (j.text.length > 60 ? '…' : ''));
    const badges = [
      j.tone_preset ? `<span class="badge tone">${esc(j.tone_preset)}</span>` : '',
      j.voice_name  ? `<span class="badge voice">🎙 ${esc(j.voice_name)}</span>` : '',
      `<span class="badge status-${j.status}">${statusIcon(j.status)} ${j.status}</span>`,
    ].join('');

    const meta = j.status === 'completed'
      ? `${j.duration_sec}s · ${fmtTs(j.created_at)}`
      : j.status === 'processing'
      ? `started ${fmtTs(j.started_at)}`
      : fmtTs(j.created_at);

    const actions = [
      j.status === 'completed'
        ? `<button class="btn-icon" onclick="toggleAudio('${j.id}')" title="Play">▶</button>` : '',
      j.status === 'completed'
        ? `<a href="/outputs/${j.id}" download><button class="btn-icon" title="Download">⬇</button></a>` : '',
      j.status !== 'processing'
        ? `<button class="btn-icon" onclick="deleteJob('${j.id}')" title="Delete">✕</button>` : '',
    ].join('');

    const audioEl = j.status === 'completed' && openAudioIds.has(j.id)
      ? `<div class="job-audio"><audio controls src="/outputs/${j.id}"></audio></div>` : '';

    const errorEl = j.status === 'failed' && j.error
      ? `<div class="job-error">Error: ${esc(j.error)}</div>` : '';

    const spinner = j.status === 'processing'
      ? '<span class="spinner"></span>' : '';

    return `<div class="job-card ${j.status}" id="jcard-${j.id}">
      <div class="job-header">
        <div class="status-pip ${j.status}"></div>
        <div class="job-label">${spinner}${label}</div>
        <div style="display:flex;gap:.3rem;flex-wrap:wrap">${badges}</div>
        <div class="job-meta">${meta}</div>
        <div class="job-actions">${actions}</div>
      </div>
      <div class="job-text-preview">${esc(j.text)}</div>
      ${errorEl}${audioEl}
    </div>`;
  }).join('');
}

function statusIcon(s) {
  return {pending:'○', processing:'◉', completed:'✓', failed:'✗'}[s] || '';
}

function toggleAudio(id) {
  if (openAudioIds.has(id)) openAudioIds.delete(id);
  else openAudioIds.add(id);
  loadQueue();
}

async function deleteJob(id) {
  if (!confirm('Delete this work order?')) return;
  try {
    const r = await fetch(`/queue/${id}`, {method:'DELETE'});
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    openAudioIds.delete(id);
    await loadQueue();
  } catch(e) { alert(e.message); }
}

// ── enqueue ───────────────────────────────────────────────────────────────────
function toggleForm(show) {
  document.getElementById('new-job-form').style.display = show ? 'block' : 'none';
  document.getElementById('btn-new').style.display = show ? 'none' : 'inline-block';
  if (!show) document.getElementById('enqueue-note').textContent = '';
}

async function enqueue() {
  if (_enqueueing) return;
  const btn  = document.getElementById('btn-enqueue');
  const note = document.getElementById('enqueue-note');
  const text = document.getElementById('f-text').value.trim();
  if (!text) { note.textContent = 'Text is required.'; return; }
  _enqueueing = true;
  if (btn) btn.disabled = true;

  const body = {
    text,
    label:      document.getElementById('f-label').value.trim() || null,
    preamble:   document.getElementById('f-preamble').value.trim() || null,
    tone_preset:document.getElementById('f-tone').value || null,
    speaker:    parseInt(document.getElementById('f-speaker').value),
    voice_name: document.getElementById('f-voice').value || null,
  };

  try {
    const r = await fetch('/queue', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body),
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    document.getElementById('f-text').value = '';
    document.getElementById('f-label').value = '';
    document.getElementById('f-preamble').value = '';
    document.getElementById('f-tone').value = '';
    note.textContent = 'Work order queued.';
    setTimeout(() => toggleForm(false), 800);
    await loadQueue();
  } catch(e) { note.textContent = 'Error: ' + e.message; }
  finally { _enqueueing = false; if (btn) btn.disabled = false; }
}

// ── voices ────────────────────────────────────────────────────────────────────
function toggleVoices() {
  voicesCollapsed = !voicesCollapsed;
  document.getElementById('voices-section').style.display = voicesCollapsed ? 'none' : 'block';
  document.getElementById('voices-toggle').classList.toggle('collapsed', voicesCollapsed);
}

async function loadVoices() {
  const voices = await (await fetch('/voices')).json();
  const list = document.getElementById('voices-list');
  const sel  = document.getElementById('f-voice');
  const prev = sel.value;
  sel.innerHTML = '<option value="">— None —</option>';
  voices.forEach(v => {
    const o = document.createElement('option');
    o.value = v.name; o.textContent = v.name;
    sel.appendChild(o);
  });
  if (prev) sel.value = prev;

  list.innerHTML = voices.length
    ? voices.map(v => `
      <div class="voice-card">
        <div style="flex:1;min-width:0">
          <div class="voice-name">${esc(v.name)}</div>
          <div class="voice-sub">${v.duration_sec != null ? v.duration_sec+'s' : ''}</div>
          <div class="voice-transcript">${esc(v.transcript)}</div>
        </div>
        <button class="btn-icon" onclick="deleteVoice('${esc(v.name)}')">✕</button>
      </div>`).join('')
    : '<p style="color:#4a5568;font-size:.82rem">No saved voices.</p>';
}

function toggleAddVoice(show) {
  document.getElementById('add-voice-form').style.display = show ? 'block' : 'none';
  document.getElementById('btn-add-voice').style.display  = show ? 'none' : 'inline-block';
}

async function saveVoice() {
  const note = document.getElementById('voice-note');
  const name = document.getElementById('vname').value.trim();
  const transcript = document.getElementById('vtranscript').value.trim();
  const file = document.getElementById('vaudio').files[0];
  if (!name) { note.textContent = 'Enter a name.'; return; }
  if (!transcript) { note.textContent = 'Enter the transcript.'; return; }
  if (!file) { note.textContent = 'Select an audio file.'; return; }
  note.textContent = 'Saving…';
  const fd = new FormData();
  fd.append('name', name); fd.append('transcript', transcript); fd.append('audio', file);
  try {
    const r = await fetch('/voices', {method:'POST', body:fd});
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    const v = await r.json();
    note.textContent = `Saved "${v.name}" (${v.duration_sec}s)`;
    document.getElementById('vname').value = '';
    document.getElementById('vtranscript').value = '';
    document.getElementById('vaudio').value = '';
    await loadVoices();
    setTimeout(() => toggleAddVoice(false), 1000);
  } catch(e) { note.textContent = 'Error: ' + e.message; }
}

async function deleteVoice(name) {
  if (!confirm(`Delete voice "${name}"?`)) return;
  try {
    await fetch(`/voices/${encodeURIComponent(name)}`, {method:'DELETE'});
    await loadVoices();
  } catch(e) { alert(e.message); }
}

// ── logs ──────────────────────────────────────────────────────────────────────
const LOG_COLORS = {DEBUG:'#4a5568', INFO:'#cbd5e0', WARNING:'#ecc94b', ERROR:'#fc8181', CRITICAL:'#f56565'};

async function loadLogs() {
  if (logsCollapsed) return;
  try {
    const d = await (await fetch('/logs')).json();
    renderLogs(d.entries, d.log_file);
  } catch(e) {}
}

function renderLogs(entries, logFile) {
  const panel = document.getElementById('log-panel');
  const atBottom = panel.scrollHeight - panel.scrollTop <= panel.clientHeight + 8;
  panel.innerHTML = entries.length === 0
    ? '<span style="color:#4a5568">No log entries yet.</span>'
    : entries.map(e => {
        const color = LOG_COLORS[e.level] || '#cbd5e0';
        const ts = e.ts.replace('T', ' ');
        return `<div><span style="color:#4a5568">${esc(ts)}</span> `
             + `<span style="color:${color};font-weight:700">${e.level}</span> `
             + `<span style="color:${color}">${esc(e.msg)}</span></div>`;
      }).join('');
  if (atBottom) panel.scrollTop = panel.scrollHeight;
  const meta = document.getElementById('log-meta');
  meta.textContent = entries.length + ' entries' + (logFile ? ' · ' + logFile : '');
}

async function clearLogs() {
  if (!confirm('Clear the log file?')) return;
  try {
    await fetch('/logs', {method:'DELETE'});
    await loadLogs();
  } catch(e) { alert('Could not clear logs: ' + e.message); }
}

function toggleLogs() {
  logsCollapsed = !logsCollapsed;
  document.getElementById('logs-section').style.display = logsCollapsed ? 'none' : 'block';
  document.getElementById('logs-toggle').classList.toggle('collapsed', logsCollapsed);
  if (!logsCollapsed) loadLogs();
}

// ── utils ─────────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── init ──────────────────────────────────────────────────────────────────────
pollStatus();
loadQueue();
loadVoices();
loadLogs();
startPolling();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_server:app", host="0.0.0.0", port=8080, reload=False)
