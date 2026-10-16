"""
Run the whole pipeline: split -> transcribe -> build transcript.

    python transcribe.py 12.mp3

Everything is written to 12_mp3_output/. If a run is interrupted, running
it again picks up from the last finished chunk.
"""

import sys
from pathlib import Path

import whisper

from audio_chunker import chunk_audio, output_folder_for
from chunk_transcriber import MODEL_NAME, transcribe_all
from transcript_assembler import build_transcript, save_transcript


def main():
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <audio file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    output_folder = output_folder_for(audio_path)

    print("Splitting audio...")
    chunk_audio(audio_path, output_folder)

    print("Transcribing...")
    model = whisper.load_model(MODEL_NAME)
    transcribe_all(model, output_folder)

    print("Building transcript...")
    lines = build_transcript(output_folder)
    print("\n".join(lines))
    save_transcript(output_folder, lines)


if __name__ == "__main__":
    main()
