#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def read(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def aggregate(rows):
    refs=sum(x['reference_units'] for x in rows); s=sum(x['substitutions'] for x in rows); i=sum(x['insertions'] for x in rows); d=sum(x['deletions'] for x in rows)
    return {'samples':len(rows),'reference_units':refs,'micro_cer':(s+i+d)/max(1,refs),'substitution_rate':s/max(1,refs),'insertion_rate':i/max(1,refs),'deletion_rate':d/max(1,refs),'exact_match_rate':sum(x['normalized_cer']==0 for x in rows)/max(1,len(rows))}
def grouped(rows,key):
    groups=defaultdict(list)
    for x in rows: groups[key(x)].append(x)
    return {str(k):aggregate(v) for k,v in sorted(groups.items(),key=lambda z:str(z[0]))}
def song_stats(rows): return grouped(rows,lambda x:x['song_id'])
def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--finetuned',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--bootstrap',type=int,default=10000); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); base=read(a.baseline); fine=read(a.finetuned); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    bmap={x['id']:x for x in base}; fmap={x['id']:x for x in fine}; ids=sorted(set(bmap)&set(fmap))
    if len(ids)!=len(base) or len(ids)!=len(fine): raise SystemExit('Prediction IDs do not match exactly')
    base=[bmap[x] for x in ids]; fine=[fmap[x] for x in ids]; bs=song_stats(base); fs=song_stats(fine); songs=sorted(set(bs)&set(fs))
    macro_b=float(np.mean([bs[s]['micro_cer'] for s in songs])); macro_f=float(np.mean([fs[s]['micro_cer'] for s in songs]))
    rng=np.random.default_rng(a.seed); deltas=[]
    for _ in range(a.bootstrap):
        sample=rng.choice(songs,size=len(songs),replace=True)
        deltas.append(float(np.mean([bs[s]['micro_cer']-fs[s]['micro_cer'] for s in sample])))
    ci=np.percentile(deltas,[2.5,97.5]).tolist()
    def duration(x): return '<=3s' if x['duration']<=3 else '3-6s' if x['duration']<=6 else '6-9s' if x['duration']<=9 else '>9s'
    def textlen(x):
        n=x['reference_units']; return '<=8' if n<=8 else '9-16' if n<=16 else '17-24' if n<=24 else '>24'
    result={'paired_samples':len(ids),'songs':len(songs),'baseline':aggregate(base),'finetuned':aggregate(fine),
      'macro_song_cer_baseline':macro_b,'macro_song_cer_finetuned':macro_f,'paired_macro_song_cer_improvement':macro_b-macro_f,
      'paired_song_bootstrap_95_ci':ci,'bootstrap_replicates':a.bootstrap,
      'by_speaker':{'baseline':grouped(base,lambda x:x['speaker_id']),'finetuned':grouped(fine,lambda x:x['speaker_id'])},
      'by_duration':{'baseline':grouped(base,duration),'finetuned':grouped(fine,duration)},
      'by_lyrics_length':{'baseline':grouped(base,textlen),'finetuned':grouped(fine,textlen)}}
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'per_song.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['song_id','baseline_cer','finetuned_cer','improvement'])
        for s in songs: w.writerow([s,bs[s]['micro_cer'],fs[s]['micro_cer'],bs[s]['micro_cer']-fs[s]['micro_cer']])
    print(json.dumps({k:v for k,v in result.items() if k not in ('by_speaker','by_duration','by_lyrics_length')},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
