# misotts-jetson

End-to-end deployment of [MisoTTS](https://github.com/MisoLabsAI/MisoTTS) on a Jetson AGX Orin, with a queue-based HTTP control panel for managing synthesis work orders.

---

## Overview

MisoTTS is an 8B-parameter emotive text-to-speech model released by Miso Labs in June 2026. It produces highly expressive, contextually appropriate speech — not just "read this text", but "read this text the way it would feel in this scene". The model's key innovation is emotional contextual awareness driven by a dual-transformer architecture that reasons about delivery before generating audio tokens.

This project packages:

- `server/tts_server.py` — A FastAPI HTTP server with embedded web UI for managing and running synthesis work orders
- `scripts/install.sh` — Idempotent one-shot installer targeting a freshly flashed Jetson (handles all known dependency conflicts on this platform)
- `scripts/uninstall.sh` — Full teardown
- `systemd/misotts.service` — Systemd unit template (paths substituted by install.sh)

---

## Hardware Target

| Property | Value |
|---|---|
| Device | NVIDIA Jetson AGX Orin |
| Architecture | aarch64 |
| JetPack | R35.6.4 (JetPack 5.x) |
| RAM | 30 GB unified (CPU + GPU share the same pool) |
| OS | Ubuntu 20.04 (focal) |
| CUDA | 11.4 runtime (no toolkit) |

The 16 GB model comfortably fits in the 30 GB unified memory pool.

---

## Model Details

| Property | Value |
|---|---|
| Name | MisoTTS 8B |
| Parameters | 8B total (7.7B backbone + 300M decoder) |
| Architecture | RVQ Transformer (Sesame CSM-inspired) |
| Weights format | bfloat16 safetensors (~16 GB) |
| Audio codec | Mimi (24 kHz, 32 RVQ codebooks) |
| Tokenizer | Llama 3.2 |
| HuggingFace repo | `MisoLabs/MisoTTS` |
| License | Modified MIT (free ≤ 50M MAU or ≤ $10M/month revenue) |

### What makes it special

MisoTTS generates speech that is emotionally aware of the surrounding text context — not flat recitation. The dual-transformer design:

1. **Context transformer (7.7B)** — reads the full text and decides *how* to say it: pace, affect, register
2. **Decoder (300M)** — converts that delivery plan into 32 RVQ audio token streams simultaneously

The result is natural prosody variation, appropriate emotional register, and voice character consistency that smaller/simpler TTS systems miss entirely. It also ships with SilentCipher watermarking baked in.

---

## The CPU-Only Constraint

MisoTTS requires Python 3.10. NVIDIA's official PyTorch wheels for JetPack 5 (CUDA 11.4) only support Python 3.8. The standard PyPI `aarch64` manylinux wheel for `torch==2.4.0` is CPU-only (89.8 MB — no CUDA kernels). The Jetson AI Lab pip index (`pypi.jetson-ai-lab.dev`) does not resolve.

**Result:** Inference runs on the 12-core Cortex-A78AE CPU. Typical synthesis time is 1–3 minutes per request, which is acceptable for an asynchronous work order queue.

**To get GPU acceleration:** upgrade to JetPack 6 (which ships official Python 3.10 CUDA wheels) and re-run `install.sh`. No other changes are needed.

---

## Prerequisites

- A Jetson AGX Orin with JetPack 5.x flashed and booted
- The Jetson is on your local network and reachable by hostname (e.g., `pineway-desktop`)
- SSH access from your laptop (key-based preferred)
- ~20 GB free disk space (16 GB model + venv + outputs)
- Internet access from the Jetson (for downloading packages and model weights)

---

## Installation

Clone this repo on your **laptop**:

```bash
git clone <this-repo> ~/misotts-jetson
cd ~/misotts-jetson
```

Copy the project to the Jetson and run the installer:

```bash
scp -r . pineway@pineway-desktop:~/misotts-jetson
ssh pineway@pineway-desktop "bash ~/misotts-jetson/scripts/install.sh"
```

The installer is fully idempotent — safe to re-run after partial failures. It will:

1. Install system packages (`ffmpeg`, `libsndfile1`, build tools, etc.)
2. Install pyenv (skipped if already present)
3. Add pyenv to `~/.bashrc` (skipped if already there)
4. Compile Python 3.10.14 from source via pyenv (skipped if already built; takes ~20 min first time — deadsnakes PPA has no aarch64 focal packages)
5. Clone MisoTTS from GitHub (skipped if already present)
6. Create a Python venv at `~/misotts/.venv`
7. Install all Python dependencies in the correct order with known version overrides (see [Dependency Notes](#dependency-notes))
8. Copy `server/tts_server.py` to `~/misotts/tts_server.py`
9. Install and enable the systemd service (with `LD_PRELOAD` fix applied — see [libgomp Fix](#libgomp-tls-fix))

The model weights (~16 GB) are **not** downloaded at install time. They download on first model load from the web UI.

---

## Usage

Open `http://pineway-desktop:8080` in your browser.

### Work Order Queue

The UI presents a queue of synthesis work orders. The model lifecycle is fully automatic:

- When a work order is enqueued, the worker thread wakes up, loads the model (if not already loaded), and processes the queue
- When the queue is empty and the model has been idle for **30 minutes**, it unloads automatically
- The queue persists across server restarts (stored in `~/misotts/queue.json`)

**To create a work order:**

1. Click **New Work Order**
2. Fill in:
   - **Label** (optional) — human-readable name for the job
   - **Text** — the text to synthesise (required)
   - **Preamble / context** (optional) — extra scene-setting text the model reads before generating, e.g. *"This is an internal monologue during a tense confrontation."*
   - **Tone preset** (optional) — quick register picker:
     - *Conversational* — warm, casual, between friends
     - *Instructional* — clear, patient tutorial narration
     - *Voiceover* — professional, authoritative
     - *Dramatic* — emotionally charged, expressive
   - **Speaker** — numeric speaker ID (0–9; different voice timbres)
   - **Voice clone** — select a saved voice to clone its vocal characteristics
3. Click **Enqueue**

Completed jobs show a download link to the output WAV file.

### Voice System

Voices are stored as audio sample + transcript pairs at `~/misotts/voices/`. The transcript must exactly match what is spoken in the clip — accuracy matters for quality.

**To add a voice:**

1. Open the **Voices** section
2. Enter a name (e.g. `alice`)
3. Paste the transcript of what is said in the clip
4. Upload any audio file (WAV, MP3, FLAC, OGG — anything torchaudio reads)
5. Click **Upload**

The server converts the file to mono 24 kHz WAV. The voice is immediately available in the **Voice clone** dropdown.

---

## API Reference

All endpoints return JSON unless noted.

### Status & Model

| Method | Path | Description |
|---|---|---|
| `GET` | `/status` | `{status, error, progress, progress_msg, idle_remaining}` |

`status` values: `idle` / `loading` / `ready` / `unloading` / `error`

`idle_remaining` is seconds until auto-unload (null if model is not loaded).

### Queue

| Method | Path | Body / Notes |
|---|---|---|
| `GET` | `/queue` | Returns all jobs, newest first |
| `POST` | `/queue` | Enqueue a work order (see schema below) |
| `DELETE` | `/queue/{id}` | Cancel a pending job |

**Enqueue body (JSON):**

```json
{
  "text":        "The text to synthesise.",
  "label":       "optional human-readable name",
  "preamble":    "optional extra scene-setting context",
  "tone_preset": "Conversational | Instructional | Voiceover | Dramatic | null",
  "speaker":     0,
  "voice_name":  "alice | null"
}
```

**Job object:**

```json
{
  "id":          "uuid4",
  "status":      "pending | processing | done | error | cancelled",
  "label":       "...",
  "text":        "...",
  "preamble":    "...",
  "tone_preset": "...",
  "speaker":     0,
  "voice_name":  "alice",
  "created_at":  "ISO-8601",
  "started_at":  "ISO-8601 | null",
  "finished_at": "ISO-8601 | null",
  "error":       "error message | null",
  "output_path": "/outputs/uuid4.wav | null"
}
```

### Outputs

| Method | Path | Description |
|---|---|---|
| `GET` | `/outputs/{id}` | Download completed WAV file (FileResponse) |

### Voices

| Method | Path | Description |
|---|---|---|
| `GET` | `/voices` | List saved voices: `["alice", "bob", ...]` |
| `POST` | `/voices` | Upload a voice (multipart: `name`, `transcript`, `audio`) |
| `DELETE` | `/voices/{name}` | Delete a voice |

---

## Architecture

### Server (`tts_server.py`)

Single-file FastAPI app with embedded HTML/CSS/JS (~900 lines). Key design decisions:

- **Module-level state** (`_mstate` dict + `_queue` list) — simple enough for a single-server single-user tool; no database needed
- **Queue persistence** — serialised to `queue.json` on every mutation, loaded at startup
- **Worker thread** (`_worker()`) — a long-lived daemon thread that blocks on `_new_job_event`, processes jobs serially, and starts an idle timer when the queue empties
- **Idle auto-unload** — `threading.Timer(IDLE_TIMEOUT, _auto_unload)` started when queue empties; cancelled when a new job arrives. IDLE_TIMEOUT defaults to 30 minutes
- **Synthesis lock** — `asyncio.Lock` ensures only one synthesis runs at a time (the worker thread is the only caller, so this is belt-and-suspenders)
- **Tone steering** — implemented by prepending a text-only `Segment` (with 0.5 s of silence as placeholder audio: `torch.zeros(12000)` at 24 kHz) that delivers scene-setting prose before the actual text. This conditions the context transformer's delivery plan.
- **Voice cloning** — MisoTTS's native `Segment(speaker, transcript, audio)` context injection. The saved WAV is loaded, resampled to 24 kHz, and passed as a context segment.

### Model Loading Progress

Loading is split into two phases, each tracked independently:

- **Phase 1 — Download (0–50%):** polls `~/.cache/huggingface/hub/models--MisoLabs--MisoTTS/blobs/` cumulative size vs 16.1 GB expected. Skipped if already cached.
- **Phase 2 — Memory load (50–100%):** polls `psutil.virtual_memory().used` delta from a baseline taken just before `load_miso_8b()`. Expected delta is ~16 GB.

The frontend polls `/status` every 1.5 seconds and animates a progress bar.

---

## Configuration

All tuneable constants are at the top of `server/tts_server.py`:

```python
IDLE_TIMEOUT = 30 * 60   # seconds until auto-unload after queue empties

TONE_PRESETS = {
    "Conversational": "This is a warm, casual conversation between friends.",
    "Instructional":  "This is clear, patient instructional narration for a tutorial.",
    "Voiceover":      "This is a professional, authoritative voiceover presentation.",
    "Dramatic":       "This is an emotionally charged, dramatic and expressive reading.",
}

MISOTTS_DIR  = "/home/pineway/misotts"
VOICES_DIR   = os.path.join(MISOTTS_DIR, "voices")
OUTPUTS_DIR  = os.path.join(MISOTTS_DIR, "outputs")
QUEUE_FILE   = os.path.join(MISOTTS_DIR, "queue.json")
MODEL_REPO   = "MisoLabs/MisoTTS"
EXPECTED_MODEL_BYTES = 16_100_000_000
```

---

## Dependency Notes

These version overrides are baked into `install.sh` to handle known conflicts on aarch64 focal:

| Package | Version | Reason |
|---|---|---|
| `torch` / `torchaudio` | 2.4.0 | Highest version with a manylinux aarch64 wheel |
| `bitsandbytes` | 0.46.0 | First version with an aarch64 binary wheel |
| `moshi` | 0.2.2 `--no-deps` | Pins `bitsandbytes<0.46` which conflicts; skip its deps |
| `sentencepiece` | 0.2.0 | moshi requires `==0.2`; 0.2.1 breaks it |
| `safetensors` | 0.5.3 | moshi requires `<0.6`; latest is 0.8+ |
| `torchao` | 0.9.0 | Required by torchtune; pure-Python wheel works on any arch |

---

## libgomp TLS Fix

Python built by pyenv on aarch64 loads shared libraries late in the process startup, which means the scikit-learn-bundled `libgomp` misses the TLS (Thread Local Storage) registration window. Symptom:

```
ImportError: cannot allocate memory in static TLS block
```

Fix: preload the library before Python starts, so the OS registers it in TLS early.

The `install.sh` script finds the library path automatically and writes it into the systemd `Environment=LD_PRELOAD=...` directive. The server also sets it at the top of `tts_server.py` as a fallback for running outside systemd:

```python
import ctypes, glob as _glob
_gomp = _glob.glob(".venv/lib/*/site-packages/scikit_learn.libs/libgomp-*.so*")
if _gomp:
    ctypes.CDLL(_gomp[0], mode=ctypes.RTLD_GLOBAL)
```

---

## Troubleshooting

**Service won't start / crashes immediately:**

```bash
journalctl -u misotts -n 50
```

**Model download stalls:**

HuggingFace Hub resumes interrupted downloads. Just restart the service; the progress bar will pick up from where it left off.

**`cannot allocate memory in static TLS block`:**

The libgomp preload is missing. Check that `LD_PRELOAD` is set in `/etc/systemd/system/misotts.service` and that the `.so` path exists inside the venv. Re-run `install.sh` to regenerate it.

**`sentencepiece` version error at startup:**

```bash
~/misotts/.venv/bin/pip install sentencepiece==0.2.0 --force-reinstall
sudo systemctl restart misotts
```

**`bitsandbytes` `aarch64` error:**

```bash
~/misotts/.venv/bin/pip install bitsandbytes==0.46.0 --force-reinstall
sudo systemctl restart misotts
```

**Synthesis is very slow (1–3 min):**

This is expected on JetPack 5. The model runs on CPU. See [The CPU-Only Constraint](#the-cpu-only-constraint).

---

## File Layout on the Jetson

```
/home/pineway/
├── .pyenv/versions/3.10.14/    Python 3.10 compiled from source
└── misotts/                    MisoTTS repo
    ├── .venv/                  Python venv
    ├── generator.py            MisoTTS inference (upstream)
    ├── models.py               Model definition (upstream)
    ├── moshi_compat.py         bitsandbytes bypass patch (upstream)
    ├── watermarking.py         SilentCipher integration (upstream)
    ├── tts_server.py           HTTP control panel (this repo)
    ├── queue.json              Persistent work order queue
    ├── voices/                 Voice clone samples
    │   ├── alice.wav
    │   ├── alice.txt
    │   └── ...
    └── outputs/                Completed synthesis WAVs
        ├── <uuid>.wav
        └── ...

~/.cache/huggingface/hub/
└── models--MisoLabs--MisoTTS/ Downloaded weights (~16 GB)

/etc/systemd/system/
└── misotts.service             Systemd unit
```

---

## Known Limitations & Upgrade Path

| Limitation | Path to fix |
|---|---|
| CPU-only inference (~1–3 min/request) | Upgrade to JetPack 6 and re-run `install.sh` |
| No authentication | Add HTTP Basic Auth or API key middleware to FastAPI |
| No HTTPS | Put nginx in front as a TLS terminator |
| Single synthesis at a time | Already serialised by design; add a job priority field if needed |
| Voice cloning quality depends on transcript accuracy | Transcript must exactly match the spoken words in the clip |
| Watermarking always on | SilentCipher is baked into `gen.generate()` — disable upstream if needed |
