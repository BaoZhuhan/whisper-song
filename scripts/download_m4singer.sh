#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?DATA_ROOT must be set}"
download_dir="$DATA_ROOT/downloads"
archive="$download_dir/m4singer.zip"
raw_dir="$DATA_ROOT/raw/m4singer"
mkdir -p "$download_dir" "$raw_dir"
if [[ ! -s "$archive" ]]; then
  python -m gdown --fuzzy \
    'https://drive.google.com/file/d/1xC37E59EWRRFFLdG3aJkVqwtLDgtFNqW/view?usp=share_link' \
    -O "$archive.part"
  mv "$archive.part" "$archive"
fi
file "$archive"
unzip -t "$archive"
if [[ ! -f "$raw_dir/.extract-complete" ]]; then
  unzip -q "$archive" -d "$raw_dir"
  touch "$raw_dir/.extract-complete"
fi
find "$raw_dir" -type f | sort > "$DATA_ROOT/m4singer_files.txt"
du -sh "$archive" "$raw_dir"
