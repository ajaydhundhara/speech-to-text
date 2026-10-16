# Speech-to-Text Prototype

Transcribes an audio file into a timestamped transcript using OpenAI Whisper. The audio is split into 30-second chunks, so memory use stays flat on long recordings, and an interrupted run resumes where it stopped.

## How it works

```
audio file ──► audio_chunker.py ──► chunk_transcriber.py ──► transcript_assembler.py
  (mp3/wav)    30s WAV chunks +      text_XXXX.json per      transcript.txt
               chunks.json           chunk (Whisper)         (global timestamps)
```

| File | Role |
| --- | --- |
| `transcribe.py` | Entry point. Runs all three steps in order. |
| `audio_chunker.py` | Uses `ffmpeg` to convert to 16 kHz mono and split into 30 s WAV chunks. Writes `chunks.json` with each chunk's start and end time. |
| `chunk_transcriber.py` | Runs Whisper (`base` model) on each chunk and saves phrases with timestamps relative to the chunk. |
| `transcript_assembler.py` | Shifts each phrase by its chunk's start time and writes `transcript.txt`. |

Design notes:

- **Low memory.** ffmpeg writes chunks straight to disk, and only one chunk is loaded at a time.
- **Resumable.** Each chunk's result is saved as soon as it finishes (temp file, then rename). Re-running skips finished chunks, and a finished split is skipped because `chunks.json` exists.
- **Accurate timestamps.** Chunk start times are built from each file's real duration, not `chunk_id * 30`, because ffmpeg doesn't cut at exactly 30 s.

## Requirements

- Python 3.11 (see `.python-version`)
- [`ffmpeg`](https://ffmpeg.org/) on your `PATH` (`brew install ffmpeg` on macOS)
- [`openai-whisper`](https://github.com/openai/whisper) (pulls in `torch`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai-whisper
```

The first run downloads the Whisper `base` model.

## Usage

Run the full pipeline:

```bash
python transcribe.py audio/12.mp3
```

Output goes to a folder named after the input, with the dot replaced by an underscore (`audio/12.mp3` → `audio/12_mp3_output/`):

```
audio/12_mp3_output/
├── chunk_0000.wav ...     # 30 s audio chunks
├── chunks.json            # chunk ids, file names, start/end times
├── text_0000.json ...     # Whisper phrases per chunk
└── transcript.txt         # final transcript
```

Example `transcript.txt` line:

```
[00:02.0 - 00:04.0] I think I broke it.
```

Hours appear only when needed (`1:02:03.5`).

### Running the steps separately

```bash
python audio_chunker.py audio/12.mp3                 # split only
python chunk_transcriber.py audio/12_mp3_output      # transcribe chunks
python transcript_assembler.py audio/12_mp3_output   # build transcript.txt
```

### Re-running

- **Resume after interruption:** run the same command again.
- **Start over from the split:** delete the output folder.
- **Different model:** change `MODEL_NAME` in `chunk_transcriber.py`. Delete the `text_*.json` files first, or the old results will be reused.

## Quick start

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate
pip install openai-whisper

python transcribe.py path/to/your_audio.mp3
```

The transcript is written to `<your_audio>_mp3_output/transcript.txt` and also printed to the terminal.

## Limitations

- Chunks are cut at fixed 30 s boundaries, so a word at a boundary can be split or mis-transcribed.
- It runs on CPU (`fp16=False`). The `base` model takes about 1 s per chunk.
- There is no speaker labelling, and the language is auto-detected per chunk.
