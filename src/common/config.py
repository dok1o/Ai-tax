from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
SCHEMA_PATH = DATA_DIR / "SCHEMA.md"
PROMPTS_PATH = DATA_DIR / "prompts.csv"
RESPONSES_PATH = DATA_DIR / "responses.csv"
TOKENS_PATH = DATA_DIR / "tokens.csv"
LABELS_PATH = DATA_DIR / "labels.csv"
METRICS_PATH = DATA_DIR / "metrics.csv"
