"""
Combine the chunk results into one timestamped transcript.

    python transcript_assembler.py 12_mp3_output

Whisper's phrase times are relative to the chunk, so each one is
shifted by the chunk's start time. The result goes to transcript.txt.
"""

import json
import sys
from pathlib import Path


def format_time(seconds):
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)

    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{seconds:04.1f}"
    return f"{int(minutes):02d}:{seconds:04.1f}"


def build_transcript(output_folder):
    chunks = json.loads((output_folder / "chunks.json").read_text())
    lines = []

    for chunk in chunks:
        text_path = output_folder / chunk["text_file"]
        if not text_path.exists():
            raise FileNotFoundError(f"{text_path} not found, run chunk_transcriber.py first")

        phrases = json.loads(text_path.read_text())["phrases"]

        for phrase in phrases:
            start = chunk["start_seconds"] + phrase["start"]
            end = chunk["start_seconds"] + phrase["end"]
            lines.append(f"[{format_time(start)} - {format_time(end)}] {phrase['text']}")

    return lines


def save_transcript(output_folder, lines):
    transcript_path = output_folder / "transcript.txt"
    transcript_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {transcript_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python transcript_assembler.py <output folder>")
        sys.exit(1)

    output_folder = Path(sys.argv[1])
    lines = build_transcript(output_folder)
    print("\n".join(lines))
    save_transcript(output_folder, lines)


if __name__ == "__main__":
    main()
