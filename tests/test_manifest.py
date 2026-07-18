from lyrics_whisper.manifest import validate_rows

def test_duplicate_and_empty_are_rejected():
    base = {"id":"x","dataset":"d","audio_path":"missing","text_raw":"","text_normalized":"","speaker_id":"s","song_id":"q","segment_id":"1","duration":1,"sample_rate":16000}
    errors = validate_rows([base, base], check_files=False)
    assert any("empty text" in x for x in errors)
    assert any("duplicate id" in x for x in errors)
