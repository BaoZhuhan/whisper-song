from lyrics_whisper.metrics import error_counts, normalized_error_counts

def test_cer_operations():
    counts = error_counts("你好世界", "你号世界啊")
    assert (counts.substitutions, counts.insertions, counts.deletions) == (1, 1, 0)
    assert counts.cer == 0.5

def test_normalized_cer_ignores_punctuation():
    assert normalized_error_counts("你好！", "你好").cer == 0
