#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--dry-run" ]]; then
  echo 'First-run safety: this script does not auto-submit the pipeline.' >&2
  echo 'Run with --dry-run, then submit and review each gated stage manually.' >&2
  exit 2
fi
cat <<'EOF'
1. sbatch cluster/preprocess.sbatch
2. Review data/reports/data_quality_report.md and leakage report.
3. Create data_validation.PASS only via the future validator.
4. sbatch cluster/baseline.sbatch
5. sbatch cluster/smoke_test.sbatch
6. Review overfit, recovery, memory and loss reports.
7. Submit formal training only after its model-specific gate exists.
EOF

