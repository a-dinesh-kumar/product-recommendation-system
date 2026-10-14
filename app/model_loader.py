
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText

from app.config import (
    WORD2VEC_MODEL_PATH,
    FASTTEXT_MODEL_PATH,
    WORD2VEC_VECTORS_PATH,
    FASTTEXT_VECTORS_PATH,
    CATALOGUE_PATH,
    MODEL_CONFIG_PATH,
    HYBRID_CONFIG_PATH,
)


@dataclass
class ModelArtifacts:
    word2vec_model: Word2Vec
    fasttext_model: FastText
    word2vec_vectors: np.ndarray
    fasttext_vectors: np.ndarray
    catalogue: pd.DataFrame
    model_config: dict
    hybrid_config: dict


def validate_artifact_paths() -> None:
    """Fail clearly if any required production artifact is missing."""

    paths = [
        WORD2VEC_MODEL_PATH,
        FASTTEXT_MODEL_PATH,
        WORD2VEC_VECTORS_PATH,
        FASTTEXT_VECTORS_PATH,
        CATALOGUE_PATH,
        MODEL_CONFIG_PATH,
        HYBRID_CONFIG_PATH,
    ]

    missing = [str(path) for path in paths if not path.is_file()]

    if missing:
        formatted = "\n".join(f" - {path}" for path in missing)
        raise FileNotFoundError(
            "Required production artifacts are missing:\n"
            f"{formatted}"
        )

    print("All seven artifact files exist.")


def _load_json(path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")

    return data


def _normalize_rows(vectors: np.ndarray, label: str) -> np.ndarray:
    """Ensure each non-zero vector has unit length."""

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(
            f"{label} must be a 2D matrix; got shape {vectors.shape}."
        )

    if not np.isfinite(vectors).all():
        raise ValueError(f"{label} contains NaN or infinite values.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    if np.any(norms == 0):
        bad_rows = int(np.sum(norms[:, 0] == 0))
        raise ValueError(
            f"{label} contains {bad_rows} zero-vector rows. "
            "Check the exported production vectors."
        )

    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def load_artifacts() -> ModelArtifacts:
    """Load all artifacts and validate their compatibility."""

    validate_artifact_paths()

    print("Loading Word2Vec Skip-Gram model...")
    word2vec_model = Word2Vec.load(str(WORD2VEC_MODEL_PATH))

    print("Loading FastText model...")
    fasttext_model = FastText.load(str(FASTTEXT_MODEL_PATH))

    print("Loading product vector matrices...")
    word2vec_vectors = np.load(
        WORD2VEC_VECTORS_PATH, allow_pickle=False
    )
    fasttext_vectors = np.load(
        FASTTEXT_VECTORS_PATH, allow_pickle=False
    )

    print("Loading production catalogue...")
    catalogue = pd.read_csv(
        CATALOGUE_PATH,
        dtype={"product_id": "string"},
    )

    model_config = _load_json(MODEL_CONFIG_PATH)
    hybrid_config = _load_json(HYBRID_CONFIG_PATH)

    # Validate required catalogue columns.
    required_columns = {"product_id"}

    missing_columns = required_columns - set(catalogue.columns)

    if missing_columns:
        raise ValueError(
            "Production catalogue is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if catalogue.empty:
        raise ValueError("Production catalogue is empty.")

    if catalogue["product_id"].isna().any():
        raise ValueError("Catalogue contains missing product IDs.")

    catalogue["product_id"] = catalogue["product_id"].astype(str)

    if catalogue["product_id"].duplicated().any():
        raise ValueError("Catalogue product IDs are not unique.")

    # Validate row alignment: row i must describe vector i.
    row_count = len(catalogue)

    if word2vec_vectors.ndim != 2:
        raise ValueError("Word2Vec product vectors must be a 2D matrix.")

    if fasttext_vectors.ndim != 2:
        raise ValueError("FastText product vectors must be a 2D matrix.")

    if word2vec_vectors.shape[0] != row_count:
        raise ValueError(
            "Word2Vec vector rows do not match catalogue rows: "
            f"{word2vec_vectors.shape[0]} vs {row_count}."
        )

    if fasttext_vectors.shape[0] != row_count:
        raise ValueError(
            "FastText vector rows do not match catalogue rows: "
            f"{fasttext_vectors.shape[0]} vs {row_count}."
        )

    # Validate embedding dimensions against their trained models.
    if word2vec_vectors.shape[1] != word2vec_model.wv.vector_size:
        raise ValueError(
            "Word2Vec vector dimension mismatch: matrix has "
            f"{word2vec_vectors.shape[1]}, model expects "
            f"{word2vec_model.wv.vector_size}."
        )

    if fasttext_vectors.shape[1] != fasttext_model.wv.vector_size:
        raise ValueError(
            "FastText vector dimension mismatch: matrix has "
            f"{fasttext_vectors.shape[1]}, model expects "
            f"{fasttext_model.wv.vector_size}."
        )

    word2vec_vectors = _normalize_rows(
        word2vec_vectors, "Word2Vec product vectors"
    )
    fasttext_vectors = _normalize_rows(
        fasttext_vectors, "FastText product vectors"
    )

    print("Artifact validation successful.")
    print(f"Catalogue rows: {row_count:,}")
    print(f"Word2Vec matrix: {word2vec_vectors.shape}")
    print(f"FastText matrix: {fasttext_vectors.shape}")
    print(f"Word2Vec dimensions: {word2vec_model.wv.vector_size}")
    print(f"FastText dimensions: {fasttext_model.wv.vector_size}")

    return ModelArtifacts(
        word2vec_model=word2vec_model,
        fasttext_model=fasttext_model,
        word2vec_vectors=word2vec_vectors,
        fasttext_vectors=fasttext_vectors,
        catalogue=catalogue,
        model_config=model_config,
        hybrid_config=hybrid_config,
    )
