import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE = re.compile(r"\s+")

def normalize_text(text: str, *, simplified: bool = True, keep_english: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _CONTROL.sub("", text).lower()
    if simplified:
        try:
            from opencc import OpenCC
            text = OpenCC("t2s").convert(text)
        except ImportError:
            pass
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if "\u4e00" <= char <= "\u9fff" or char.isdigit(): kept.append(char)
        elif keep_english and char.isascii() and char.isalpha(): kept.append(char)
        elif char.isspace(): kept.append(" ")
        elif category.startswith(("P", "S")): continue
    normalized = _SPACE.sub(" ", "".join(kept)).strip()
    return re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", normalized)

def cer_units(text: str) -> list[str]:
    return list(text.replace(" ", ""))
