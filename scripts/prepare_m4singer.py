#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import soundfile as sf
from lyrics_whisper.manifest import file_sha256, write_jsonl
from lyrics_whisper.text import normalize_text

def find_root(path: Path) -> Path:
    matches = list(path.rglob("meta.json"))
    valid = [p.parent for p in matches if any(p.parent.glob("*#*/*.wav"))]
    if len(valid) != 1: raise SystemExit(f"Expected exactly one M4Singer root, found {valid}")
    return valid[0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rejected", required=True)
    args = p.parse_args()
    root = find_root(Path(args.raw_root))
    metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    rows, rejected = [], []
    for item in metadata:
        try:
            singer, song, segment = item["item_name"].split("#")
            audio = (root / f"{singer}#{song}" / f"{segment}.wav").resolve()
            text_raw = str(item["txt"]).strip(); text = normalize_text(text_raw)
            if not audio.is_file(): raise ValueError("missing_audio")
            info = sf.info(audio)
            duration = info.frames / info.samplerate
            if not text: raise ValueError("empty_text")
            if duration < 0.5: raise ValueError("duration_below_0.5s")
            if duration > 30: raise ValueError("duration_above_30s_requires_segmentation")
            rows.append({"id":f"m4singer_{singer}_{song}_{segment}", "dataset":"m4singer", "audio_path":str(audio),
                "audio_sha256":file_sha256(audio), "text_raw":text_raw, "text_normalized":text, "speaker_id":singer,
                "song_id":song, "song_title":song, "recording_id":f"{singer}#{song}", "segment_id":segment,
                "duration":round(duration, 6), "sample_rate":info.samplerate, "channels":info.channels, "split":None})
        except Exception as exc:
            rejected.append({"item_name":item.get("item_name"), "reason":str(exc)})
    write_jsonl(rows, args.output); write_jsonl(rejected, args.rejected)
    print(json.dumps({"root":str(root), "metadata":len(metadata), "accepted":len(rows), "rejected":len(rejected)}, ensure_ascii=False))

if __name__ == "__main__": main()
