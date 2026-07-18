import hashlib
import json
from pathlib import Path

REQUIRED = {"id", "dataset", "audio_path", "text_raw", "text_normalized", "speaker_id", "song_id", "segment_id", "duration", "sample_rate"}

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]

def write_jsonl(rows, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def validate_rows(rows, *, check_files=True):
    errors, ids = [], set()
    for index, row in enumerate(rows, 1):
        missing = REQUIRED - row.keys()
        if missing: errors.append(f"row {index}: missing {sorted(missing)}")
        sample_id = row.get("id")
        if sample_id in ids: errors.append(f"row {index}: duplicate id {sample_id}")
        ids.add(sample_id)
        if not str(row.get("text_normalized", "")).strip(): errors.append(f"row {index}: empty text")
        if float(row.get("duration", 0)) < 0.5: errors.append(f"row {index}: duration below 0.5s")
        if float(row.get("duration", 0)) > 30: errors.append(f"row {index}: duration above 30s")
        if check_files and not Path(row.get("audio_path", "")).is_file(): errors.append(f"row {index}: missing audio")
    return errors

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()
