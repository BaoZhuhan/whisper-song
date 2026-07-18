from lyrics_whisper.splits import grouped_song_split, leakage_errors

def test_song_groups_never_cross_splits():
    rows = [{"id":f"{song}-{i}", "song_id":song, "audio_path":f"{song}-{i}.wav"} for song in "abcdefghij" for i in range(3)]
    split = grouped_song_split(rows)
    assert leakage_errors(split) == []
    assert all(len({r["split"] for r in split if r["song_id"] == song}) == 1 for song in "abcdefghij")
