#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path
import librosa
import soundfile as sf
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from lyrics_whisper.manifest import read_jsonl
from lyrics_whisper.metrics import error_counts, normalized_error_counts

ROOT=Path('/hpc/group/honglab/zb78')
def load_audio(path):
    x,sr=sf.read(path,dtype='float32',always_2d=False)
    if getattr(x,'ndim',1)>1: x=x.mean(axis=1)
    return librosa.resample(x,orig_sr=sr,target_sr=16000) if sr!=16000 else x
def evaluate_model(model_name, splits, output_root, batch_size=4, max_samples=None):
    local=ROOT/'model/whisper-song/base'/model_name.replace('/','--')
    processor=AutoProcessor.from_pretrained(local,local_files_only=True)
    model=AutoModelForSpeechSeq2Seq.from_pretrained(local,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    model.generation_config.language='zh'; model.generation_config.task='transcribe'
    for split in splits:
        rows=read_jsonl(ROOT/'dataset/whisper-song/manifests'/f'{split}.jsonl')[:max_samples]
        target=Path(output_root)/model_name.split('/')[-1]/split; target.mkdir(parents=True,exist_ok=True)
        predictions=[]; start=time.time(); torch.cuda.reset_peak_memory_stats()
        for pos in tqdm(range(0,len(rows),batch_size),desc=f'{model_name}:{split}'):
            chunk=rows[pos:pos+batch_size]; audio=[load_audio(r['audio_path']) for r in chunk]
            inputs=processor(audio,sampling_rate=16000,return_tensors='pt',padding=True).input_features.to('cuda',dtype=torch.bfloat16)
            with torch.inference_mode(): ids=model.generate(inputs,max_new_tokens=448)
            texts=processor.batch_decode(ids,skip_special_tokens=True)
            for row,hyp in zip(chunk,texts):
                raw=error_counts(row['text_raw'],hyp); norm=normalized_error_counts(row['text_normalized'],hyp)
                predictions.append({"id":row['id'],"speaker_id":row['speaker_id'],"song_id":row['song_id'],"duration":row['duration'],"reference":row['text_raw'],"hypothesis":hyp,
                  "raw_cer":raw.cer,"normalized_cer":norm.cer,"substitutions":norm.substitutions,"insertions":norm.insertions,"deletions":norm.deletions,"reference_units":norm.reference_units})
        elapsed=time.time()-start; refs=sum(x['reference_units'] for x in predictions); errors=sum(x['substitutions']+x['insertions']+x['deletions'] for x in predictions)
        metrics={"model":model_name,"split":split,"samples":len(predictions),"normalized_cer":errors/max(1,refs),"exact_match_rate":sum(x['normalized_cer']==0 for x in predictions)/max(1,len(predictions)),
          "elapsed_seconds":elapsed,"audio_hours":sum(x['duration'] for x in predictions)/3600,"realtime_factor":elapsed/max(1,sum(x['duration'] for x in predictions)),"peak_memory_gib":torch.cuda.max_memory_allocated()/1024**3}
        with (target/'predictions.jsonl').open('w',encoding='utf-8') as f:
            for row in predictions: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        (target/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
        with (target/'per_sample_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=predictions[0].keys()); writer.writeheader(); writer.writerows(predictions)
        print(json.dumps(metrics,ensure_ascii=False),flush=True)
    del model; torch.cuda.empty_cache()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output-root'); p.add_argument('--max-samples',type=int)
    a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); out=a.output_root or str(ROOT/'model/whisper-song/baseline')
    for model in cfg['models']: evaluate_model(model,cfg['splits'],out,int(cfg.get('batch_size',4)),a.max_samples)
if __name__=='__main__': main()
