#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
from lyrics_whisper.data import ManifestDataset, WhisperCollator
from lyrics_whisper.metrics import normalized_error_counts

STORAGE = Path('/hpc/group/honglab/zb78')
def local_model(name):
    path = STORAGE/'model/whisper-song/base'/name.replace('/','--')
    if not path.is_dir(): raise SystemExit(f'Model not downloaded: {path}')
    return str(path)
def latest_checkpoint(path):
    candidates=[]
    for p in Path(path).glob('checkpoint-*'):
        try: candidates.append((int(p.name.rsplit('-',1)[1]),p))
        except ValueError: pass
    return str(max(candidates)[1]) if candidates else None
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--resume_from_checkpoint',default=None)
    a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); set_seed(cfg.get('seed',42))
    model_path=local_model(cfg['model_name_or_path']); processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForSpeechSeq2Seq.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False; model.generation_config.language='zh'; model.generation_config.task='transcribe'
    if cfg.get('training_method')=='lora':
        from peft import LoraConfig, get_peft_model
        lc=LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],target_modules=cfg['target_modules'],bias='none')
        model=get_peft_model(model,lc); model.print_trainable_parameters()
    manifest_root=STORAGE/'dataset/whisper-song/manifests'
    limit=cfg.get('train_limit'); train=ManifestDataset(manifest_root/'train.jsonl',limit=limit); valid=ManifestDataset(manifest_root/'validation.jsonl',limit=cfg.get('eval_limit'))
    def metrics(pred):
        pred_ids=pred.predictions; label_ids=pred.label_ids.copy(); label_ids[label_ids==-100]=processor.tokenizer.pad_token_id
        hypotheses=processor.batch_decode(pred_ids,skip_special_tokens=True); references=processor.batch_decode(label_ids,skip_special_tokens=True)
        counts=[normalized_error_counts(r,h) for r,h in zip(references,hypotheses)]; refs=sum(x.reference_units for x in counts)
        return {'normalized_cer':sum(x.substitutions+x.insertions+x.deletions for x in counts)/max(1,refs)}
    args=Seq2SeqTrainingArguments(output_dir=a.output_dir,do_train=True,do_eval=True,bf16=cfg.get('bf16',True),tf32=cfg.get('tf32',True),
      gradient_checkpointing=cfg.get('gradient_checkpointing',True),use_cache=False,learning_rate=float(cfg.get('learning_rate',5e-6)),weight_decay=float(cfg.get('weight_decay',0.01)),
      warmup_ratio=float(cfg.get('warmup_ratio',0.05)),max_grad_norm=float(cfg.get('max_grad_norm',1.0)),num_train_epochs=float(cfg.get('num_train_epochs',1)),max_steps=int(cfg.get('max_steps',-1)),
      per_device_train_batch_size=int(cfg.get('per_device_train_batch_size',1)),gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps',1)),per_device_eval_batch_size=int(cfg.get('per_device_eval_batch_size',1)),
      eval_strategy=cfg.get('eval_strategy','steps'),eval_steps=int(cfg.get('eval_steps',50)),save_strategy='steps',save_steps=int(cfg.get('save_steps',50)),logging_steps=int(cfg.get('logging_steps',10)),
      save_total_limit=int(cfg.get('save_total_limit',3)),predict_with_generate=cfg.get('predict_with_generate',True),generation_max_length=int(cfg.get('generation_max_length',448)),
      load_best_model_at_end=cfg.get('load_best_model_at_end',False),metric_for_best_model='normalized_cer',greater_is_better=False,dataloader_num_workers=int(cfg.get('dataloader_num_workers',4)),
      remove_unused_columns=False,report_to=['tensorboard'],seed=cfg.get('seed',42),data_seed=cfg.get('seed',42))
    trainer=Seq2SeqTrainer(model=model,args=args,train_dataset=train,eval_dataset=valid,data_collator=WhisperCollator(processor),processing_class=processor,compute_metrics=metrics)
    resume=latest_checkpoint(a.output_dir) if a.resume_from_checkpoint=='auto' else a.resume_from_checkpoint
    result=trainer.train(resume_from_checkpoint=resume); trainer.save_model(); processor.save_pretrained(a.output_dir); trainer.save_state()
    Path(a.output_dir,'run_summary.json').write_text(json.dumps(result.metrics,indent=2),encoding='utf-8')
if __name__=='__main__': main()
