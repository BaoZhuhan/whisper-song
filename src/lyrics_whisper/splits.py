import random
from collections import defaultdict

def grouped_song_split(rows, seed=42, ratios=(0.8, 0.1, 0.1)):
    by_song = defaultdict(list)
    for row in rows: by_song[row["song_id"]].append(row)
    songs = sorted(by_song); random.Random(seed).shuffle(songs)
    total = sum(len(by_song[s]) for s in songs); targets = [total * ratios[0], total * (ratios[0] + ratios[1])]
    assigned, count = {}, 0
    for song in songs:
        assigned[song] = "train" if count < targets[0] else "validation" if count < targets[1] else "test"
        count += len(by_song[song])
    return [{**row, "split": assigned[row["song_id"]]} for row in rows]

def leakage_errors(rows):
    errors = []
    for field in ("song_id", "audio_path", "audio_sha256"):
        owners = defaultdict(set)
        for row in rows:
            if row.get(field): owners[row[field]].add(row.get("split"))
        leaked = [key for key, splits in owners.items() if len(splits) > 1]
        if leaked: errors.append(f"{field} overlaps across splits: {len(leaked)}")
    return errors
