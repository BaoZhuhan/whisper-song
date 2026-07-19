#!/usr/bin/env python3
import argparse
from pathlib import Path
from huggingface_hub import HfApi

p=argparse.ArgumentParser()
p.add_argument('--repo-id',required=True)
p.add_argument('--model-dir',required=True)
p.add_argument('--model-card',required=True)
p.add_argument('--private',action='store_true')
a=p.parse_args(); api=HfApi()
api.create_repo(repo_id=a.repo_id,repo_type='model',private=a.private,exist_ok=True)
allowed=['model.safetensors','config.json','generation_config.json','processor_config.json','tokenizer.json','tokenizer_config.json','run_summary.json']
api.upload_folder(repo_id=a.repo_id,repo_type='model',folder_path=a.model_dir,allow_patterns=allowed,commit_message='Upload fine-tuned Whisper medium model')
api.upload_file(repo_id=a.repo_id,repo_type='model',path_or_fileobj=a.model_card,path_in_repo='README.md',commit_message='Add model card')
print(f'https://huggingface.co/{a.repo_id}')
