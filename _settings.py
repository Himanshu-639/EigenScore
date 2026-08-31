import os

# Use repository-local data directory by default so paths resolve on this machine.
_BASE_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join(_BASE_DIR, 'weights')
DATA_FOLDER = os.path.join(_BASE_DIR, 'datasets')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)

