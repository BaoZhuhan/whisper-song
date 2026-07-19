#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, Seq2SeqTrainingArguments

p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--output',required=True)
a=p.parse_args(); model=AutoModelForSpeechSeq2Seq.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16)
total=sum(x.numel() for x in model.parameters()); trainable=sum(x.numel() for x in model.parameters() if x.requires_grad)
args=Seq2SeqTrainingArguments(output_dir='/tmp/model-audit',bf16=True,optim='adamw_torch_fused')
optimizer=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=5e-6,weight_decay=0.01)
result={"total_parameters":total,"trainable_parameters":trainable,"trainable_ratio":trainable/total,
 "encoder_trainable":sum(x.numel() for x in model.model.encoder.parameters() if x.requires_grad),
 "decoder_trainable":sum(x.numel() for x in model.model.decoder.parameters() if x.requires_grad),
 "configured_trainer_optimizer":str(args.optim),"audit_optimizer_class":type(optimizer).__name__,
 "cuda_allocated_gib":torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0,
 "cuda_reserved_gib":torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_available() else 0}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
