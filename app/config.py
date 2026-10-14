
from pathlib import Path

# Project root: the directory containing the app/ folder
BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "production_artifacts"

WORD2VEC_MODEL_PATH = (
    ARTIFACTS_DIR / "word2vec_skipgram_full.model"
)
FASTTEXT_MODEL_PATH = ARTIFACTS_DIR / "fasttext_full.model"

WORD2VEC_VECTORS_PATH = ARTIFACTS_DIR / "product_vectors.npy"
FASTTEXT_VECTORS_PATH = (
    ARTIFACTS_DIR / "fasttext_product_vectors.npy"
)

CATALOGUE_PATH = ARTIFACTS_DIR / "production_catalogue.csv"

MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"
HYBRID_CONFIG_PATH = ARTIFACTS_DIR / "hybrid_config.json"
