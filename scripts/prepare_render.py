"""Install the bundled inference assets into runtime directories."""
import shutil
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for name in ('trained_models/full_v3', 'reports/full_v3', 'reports/web_analyses', 'temp/uploads'):
    (root / name).mkdir(parents=True, exist_ok=True)
for name in ('best.pt', 'config.json'):
    shutil.copy2(root / 'deploy/full_v3' / name, root / 'trained_models/full_v3' / name)
shutil.copy2(root / 'deploy/full_v3/test_metrics.json', root / 'reports/full_v3/test_metrics.json')
print('full_v3 model and evaluation threshold installed')
