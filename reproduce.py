"""Reproduce the manuscript figures and numerical tables."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    'catalog': 'catalog.py',
    'lead-ratios': 'lead_ratios.py',
    'figure-1': 'figure_01.py',
    'forecast': 'forecast.py',
    'tables': 'tables.py',
    'figures': 'figures.py',
    'sensitivity': 'sensitivity.py',
    'timing-stratification': 'timing_stratification.py',
    'coupling': 'coupling_results.py',
    'pilots': 'pilots.py',
    'verify-results': 'verify_results.py',
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=ROOT / 'data/current.parquet')
parser.add_argument('--stage', choices=[*STAGES, 'shoulders'])
args = parser.parse_args()
if not args.data.is_file():
    parser.error('Catalog file does not exist')
env = os.environ.copy()
env.update(APTO_DATA=str(args.data.resolve()), MPLBACKEND='Agg',
           MPLCONFIGDIR=str(ROOT / 'outputs/.mpl'), PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH=str(ROOT / 'analysis'), PYTHONUNBUFFERED='1')
for key in ('WK3_Y0', 'WK3_Y1', 'WK3_HORIZON', 'WK3_JUMP_FLOOR'):
    env.pop(key, None)
for name in ['data', 'figures', 'diagnostics', 'logs']:
    (ROOT / 'outputs' / name).mkdir(parents=True, exist_ok=True)
stages = [args.stage] if args.stage else list(STAGES)
for stage in stages:
    script = 'shoulders.py' if stage == 'shoulders' else STAGES[stage]
    print(f'Running {stage}', flush=True)
    log_path = ROOT / 'outputs/logs' / (stage + '.log')
    with log_path.open('w') as log:
        result = subprocess.run([sys.executable, str(ROOT / 'pipeline' / script)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log_path.read_text()[-8000:])
        sys.exit(result.returncode)
    print(f'Completed {stage}', flush=True)
