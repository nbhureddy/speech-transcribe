# live-transcriber

Real-time system-audio transcription for macOS/Linux, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Captures audio from any input device (incl. BlackHole for system audio), transcribes it locally using Whisper, and appends results to a timestamped text file — all with zero cloud dependency.

---

## Requirements

| Tool | Version |
|------|---------|
| Python | ≥ 3.9 |
| [BlackHole](https://existential.audio/blackhole/) *(macOS system audio)* | 2ch or 16ch |

---

## Installation

```bash
# 1. Clone / navigate to the project
cd live_speech_to_text

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install the package in editable mode
pip install -e .
```

This registers the `live-transcriber` CLI command inside the active virtualenv.

---

## Quick Start

```bash
# Auto-detect MacBook mic and start transcribing
live-transcriber

# Add session context (written as header in transcript file)
live-transcriber --context "Speaker: John Doe, Topic: Q4 earnings"
live-transcriber --context-file ~/notes/session.txt

# Specify a device by index
live-transcriber --device 2

# Enable LLM refinement (set llm.enabled: true in config.yaml first)
live-transcriber

# Use a custom config file
live-transcriber --config /path/to/my_config.yaml

# Enable verbose debug logging
live-transcriber --debug
```

Press **CTRL+C** to stop. If LLM refinement is enabled, the cleaned transcript is saved automatically on exit.

---

## Configuration

Copy and edit `config.yaml` (already at the project root):

```yaml
model:
  size: small          # tiny | base | small | medium | large-v3
  language: en         # BCP-47 code; null = auto-detect
  compute_type: int8

audio:
  sample_rate: 16000
  chunk_seconds: 6     # transcription window length
  overlap_seconds: 1   # overlap between windows
  block_size: 4096

transcription:
  vad_filter: true
  beam_size: 5
  temperature: 0.0

output:
  transcript_dir: transcripts  # folder where per-run transcript files are written

logging:
  level: INFO
  log_file: logs/app.log   # null = stdout only
  suppress_third_party: true

# LLM refinement — runs at end of session on CTRL+C
llm:
  enabled: false           # set to true to enable
  provider: openai         # openai | ollama | anthropic
  model: gpt-4o-mini
  api_key: null            # or set OPENAI_API_KEY env var
  base_url: null           # e.g. http://localhost:11434/v1 for Ollama
  temperature: 0.1
  max_tokens: 4096

debug: false
```

---

## macOS System Audio Setup (BlackHole)

1. Install [BlackHole 2ch](https://existential.audio/blackhole/)
2. Open **Audio MIDI Setup** → create a **Multi-Output Device** combining BlackHole + your speakers
3. Set that Multi-Output Device as your system output
4. `live-transcriber` will auto-detect BlackHole as the input

---

## Project Structure

```
live_speech_to_text/
├── src/
│   └── live_transcriber/
│       ├── audio/
│       │   ├── capture.py          # AudioCapture (InputStream + queue)
│       │   └── devices.py          # Device detection & selection
│       ├── io/
│       │   ├── wav_writer.py       # Float32 → 16-bit PCM WAV
│       │   └── transcript_logger.py# Timestamped file appender
│       ├── transcription/
│       │   └── transcriber.py      # LiveTranscriber (faster-whisper)
│       ├── config.py               # AppConfig dataclass + YAML loader
│       └── cli.py                  # argparse entry point
├── config.yaml                     # Default configuration
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Development

```bash
# Install in editable mode with all deps
pip install -e .

# Run directly without installing
python -m live_transcriber.cli
```

---

## License

MIT
