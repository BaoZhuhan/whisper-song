from dataclasses import dataclass
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset
from .manifest import read_jsonl

class ManifestDataset(Dataset):
    def __init__(self, manifest, limit=None): self.rows = read_jsonl(manifest)[:limit]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]

@dataclass
class WhisperCollator:
    processor: object
    sampling_rate: int = 16000
    def __call__(self, rows):
        features, label_features = [], []
        for row in rows:
            audio, sr = sf.read(row["audio_path"], dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) > 1: audio = audio.mean(axis=1)
            if sr != self.sampling_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
            values = self.processor.feature_extractor(audio, sampling_rate=self.sampling_rate).input_features[0]
            features.append({"input_features": values})
            label_features.append({"input_ids": self.processor.tokenizer(row["text_normalized"]).input_ids})
        batch = self.processor.feature_extractor.pad(features, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (ids[:, 0] == self.processor.tokenizer.bos_token_id).all().item(): ids = ids[:, 1:]
        batch["labels"] = ids
        return batch
