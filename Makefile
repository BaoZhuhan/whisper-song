.PHONY: inspect dry-run test
inspect:
	bash cluster/inspect_cluster.sh

dry-run:
	bash cluster/submit_pipeline.sh --dry-run

test:
	/hpc/group/honglab/zb78/env/whisper-song/bin/python -m pytest -q
