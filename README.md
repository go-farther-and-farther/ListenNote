# ListenNote - System Audio Speech to Text

Real-time speech recognition from system audio (videos, music, meetings, etc.) using Whisper. Captures any sound playing on your computer and transcribes it to text files.

## Features

- Captures all system audio via WASAPI loopback (Windows)
- Real-time speech recognition using faster-whisper
- Supports Chinese and other languages
- Auto-saves each session to a timestamped file in `output/`
- Skips silent segments to avoid garbage output

## Requirements

- Windows 10/11
- Python 3.8+
- Speaker or headphone connected (for loopback capture)

## Quick Start

1. Double-click `install.bat` to create venv and install dependencies
2. Play any video/audio on your computer
3. Double-click `run.bat` to start recognition
4. Press `Ctrl+C` to stop

Results are saved to `output/captions_YYYYMMDD_HHMMSS.txt`

## Configuration

Edit the top of `ListenNote.py`:

| Option | Default | Description |
|--------|---------|-------------|
| `CHUNK_SECONDS` | 15 | Seconds between each recognition (shorter = more real-time, longer = more accurate) |
| `MODEL_SIZE` | `medium` | Whisper model: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `DEVICE` | `cpu` | `cpu` or `cuda` (for NVIDIA GPU acceleration) |
| `LANGUAGE` | `zh` | Language code (`zh` for Chinese, `en` for English, etc.) |

## Model Size Reference

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| tiny | ~75MB | Fastest | Lower |
| base | ~150MB | Fast | Basic |
| small | ~500MB | Medium | Good |
| medium | ~1.5GB | Slower | Great (recommended for Chinese) |
| large-v3 | ~3GB | Slowest | Best |

## Project Structure

```
ListenNote/
├── ListenNote.py       # Main script
├── install.bat         # Install dependencies
├── run.bat             # Start recognition
├── requirements.txt    # Python dependencies
├── output/             # Generated caption files
└── README.md
```

## License

MIT
