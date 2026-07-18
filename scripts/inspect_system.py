#!/usr/bin/env python3
import json
import os
import platform
import socket

print(json.dumps({
    "hostname": socket.gethostname(),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
}, indent=2))

