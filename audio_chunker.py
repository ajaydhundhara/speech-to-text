"""
Split an audio file into 30 second WAV chunks for Whisper.

    python audio_chunker.py 12.mp3

Output goes to 12_mp3_output/: the chunk files plus chunks.json with the
start and end time of each chunk. If chunks.json already exists this
step is skipped, so delete the folder if you want to split again.
"""

import json
import subprocess
import sys
import wave
from pathlib import Path


SAMPLE_RATE = 16000
CHUNK_SECONDS = 30  # Whisper works on 30s windows, shorter chunks don't run any faster


def output_folder_for(audio_path):
    # Keep the extension in the name so 12.mp3 and 12.wav don't share a folder.
    return audio_path.with_name(audio_path.name.replace(".", "_") + "_output")


def split_audio(audio_path, output_folder):
    # ffmpeg converts to 16kHz mono (the format Whisper expects) and writes
    # the chunks straight to disk, so the full recording is never in memory.
    command = [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "segment",
        "-segment_time", str(CHUNK_SECONDS),
        str(output_folder / "chunk_%04d.wav"),
    ]
    result = subprocess.run(command, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())


def list_chunks(output_folder):
    # ffmpeg doesn't cut at exactly 30s, so the start times are built up
    # from each file's real length rather than chunk_id * 30.
    chunks = []
    start = 0.0

    for chunk_id, path in enumerate(sorted(output_folder.glob("chunk_*.wav"))):
        with wave.open(str(path)) as wav_file:
            duration = wav_file.getnframes() / wav_file.getframerate()

        chunks.append({
            "chunk_id": chunk_id,
            "audio_file": path.name,
            "text_file": f"text_{chunk_id:04d}.json",
            "start_seconds": round(start, 3),
            "end_seconds": round(start + duration, 3),
        })
        start += duration

    return chunks


def chunk_audio(audio_path, output_folder):
    chunks_path = output_folder / "chunks.json"

    if chunks_path.exists():
        print("Chunks already exist, skipping")
        return

    output_folder.mkdir(exist_ok=True)
    split_audio(audio_path, output_folder)
    chunks = list_chunks(output_folder)

    # Written last so an interrupted split isn't mistaken for a finished one.
    chunks_path.write_text(json.dumps(chunks, indent=2))
    print(f"Created {len(chunks)} chunks in {output_folder}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python audio_chunker.py <audio file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    chunk_audio(audio_path, output_folder_for(audio_path))


if __name__ == "__main__":
    main()
