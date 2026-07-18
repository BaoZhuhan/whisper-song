from dataclasses import dataclass
from .text import cer_units, normalize_text

@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    insertions: int
    deletions: int
    reference_units: int
    @property
    def cer(self) -> float:
        errors = self.substitutions + self.insertions + self.deletions
        return errors / self.reference_units if self.reference_units else float(errors > 0)

def error_counts(reference: str, hypothesis: str) -> ErrorCounts:
    ref, hyp = cer_units(reference), cer_units(hypothesis)
    rows = [[(j, 0, j, 0) for j in range(len(hyp) + 1)]]
    for i in range(1, len(ref) + 1):
        row = [(i, 0, 0, i)]
        for j in range(1, len(hyp) + 1):
            candidates = []
            c, s, ins, d = rows[i - 1][j - 1]
            candidates.append((c, s, ins, d) if ref[i - 1] == hyp[j - 1] else (c + 1, s + 1, ins, d))
            c, s, ins, d = row[j - 1]; candidates.append((c + 1, s, ins + 1, d))
            c, s, ins, d = rows[i - 1][j]; candidates.append((c + 1, s, ins, d + 1))
            row.append(min(candidates))
        rows.append(row)
    _, s, ins, d = rows[-1][-1]
    return ErrorCounts(s, ins, d, len(ref))

def normalized_error_counts(reference: str, hypothesis: str) -> ErrorCounts:
    return error_counts(normalize_text(reference), normalize_text(hypothesis))
