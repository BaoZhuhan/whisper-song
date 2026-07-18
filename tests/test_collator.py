from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch
from lyrics_whisper.data import WhisperCollator

class FeatureExtractor:
    def __call__(self, audio, sampling_rate): return SimpleNamespace(input_features=[np.asarray(audio[:4])])
    def pad(self, rows, return_tensors): return {"input_features":torch.tensor(np.stack([r["input_features"] for r in rows]))}
class Tokenizer:
    bos_token_id=1; pad_token_id=0
    def __call__(self,text): return SimpleNamespace(input_ids=[1]+[ord(c)%20+2 for c in text])
    def pad(self,rows,return_tensors):
        width=max(len(r['input_ids']) for r in rows); ids=[]; masks=[]
        for r in rows:
            n=len(r['input_ids']); ids.append(r['input_ids']+[0]*(width-n)); masks.append([1]*n+[0]*(width-n))
        return {'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(masks)}
class Processor:
    feature_extractor=FeatureExtractor(); tokenizer=Tokenizer()

def test_collator_masks_label_padding(tmp_path):
    wav=tmp_path/'x.wav'; sf.write(wav,np.zeros(16000,dtype=np.float32),16000)
    batch=WhisperCollator(Processor())([{'audio_path':str(wav),'text_normalized':'你好'},{'audio_path':str(wav),'text_normalized':'你'}])
    assert batch['labels'].shape == (2,2)
    assert batch['labels'][1,-1].item() == -100
    assert torch.isfinite(batch['input_features']).all()
