from lyrics_whisper.text import normalize_text

def test_nfkc_punctuation_case_and_space():
    assert normalize_text("ＡＢＣ，  你\n好！") == "abc 你好"

def test_mixed_language_and_digits_are_kept():
    assert normalize_text("爱 U ３０００") == "爱 u 3000"
