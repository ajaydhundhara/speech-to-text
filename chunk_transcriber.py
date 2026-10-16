"""
Transcribe each chunk with Whisper.

    python chunk_transcriber.py 12_mp3_output

Each chunk's result is saved to its own text_XXXX.json as soon as it's
done. Chunks that already have a result are skipped, so an interrupted
run can just be started again.
"""

import json
import re
import sys
from pathlib import Path

import whisper

from audio_chunker import SAMPLE_RATE


MODEL_NAME = "base"


def transcribe_chunk(model, audio):
    """Return the phrases in a chunk, with times relative to the chunk start."""
    chunk_seconds = len(audio) / SAMPLE_RATE

    # Whisper takes exactly 30s of audio as a log-mel spectrogram,
    # so the last (shorter) chunk gets padded with silence.
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # Run the encoder ourselves and pass its output to decode(),
    # which then only runs the decoder.
    audio_features = model.embed_audio(mel.unsqueeze(0))
    options = whisper.DecodingOptions(fp16=False)  # fp16 is GPU only
    result = whisper.decode(model, audio_features, options)[0]

    return parse_phrases(model, result.tokens, chunk_seconds)


def parse_phrases(model, tokens, chunk_seconds):
    # The decoder puts a timestamp before and after each phrase, e.g.
    #   <|0.00|> Hello?<|2.00|><|2.00|> I broke it.<|4.00|>
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )
    text = tokenizer.decode_with_timestamps(tokens)

    # The closing timestamp is missing when the chunk ends mid-sentence.
    pattern = r"<\|([\d.]+)\|>([^<]+)(?:<\|([\d.]+)\|>)?"

    phrases = []
    for start, words, end in re.findall(pattern, text):
        end = float(end) if end else chunk_seconds
        phrases.append({
            "start": float(start),
            "end": min(end, chunk_seconds),
            "text": words.strip(),
        })

    return phrases


def save_json(path, data):
    # Write to a temp file and rename it, so a run killed mid-write
    # can't leave a partial file that would be skipped next time.
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temp_path.replace(path)


def transcribe_all(model, output_folder):
    chunks = json.loads((output_folder / "chunks.json").read_text())

    for chunk in chunks:
        progress = f"[{chunk['chunk_id'] + 1}/{len(chunks)}]"
        text_path = output_folder / chunk["text_file"]

        if text_path.exists():
            print(f"{progress} already done")
            continue

        audio = whisper.load_audio(str(output_folder / chunk["audio_file"]))
        phrases = transcribe_chunk(model, audio)
        save_json(text_path, {"chunk_id": chunk["chunk_id"], "phrases": phrases})

        print(progress, " ".join(phrase["text"] for phrase in phrases))


def main():
    if len(sys.argv) != 2:
        print("Usage: python chunk_transcriber.py <output folder>")
        sys.exit(1)

    output_folder = Path(sys.argv[1])
    if not (output_folder / "chunks.json").exists():
        print(f"No chunks.json in {output_folder}, run audio_chunker.py first")
        sys.exit(1)

    model = whisper.load_model(MODEL_NAME)
    transcribe_all(model, output_folder)


if __name__ == "__main__":
    main()
