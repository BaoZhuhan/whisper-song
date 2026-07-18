#!/usr/bin/env python3
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument('--models', nargs='+', required=True)
p.add_argument('--model-root', required=True)
args = p.parse_args()
root = Path(args.model_root)
root.mkdir(parents=True, exist_ok=True)
for repo in args.models:
    target = root / repo.replace('/', '--')
    print(f'downloading {repo} -> {target}', flush=True)
    snapshot_download(repo_id=repo, local_dir=target)
