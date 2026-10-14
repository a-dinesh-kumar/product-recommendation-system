
import re

import numpy as np
import pandas as pd

from app.model_loader import ModelArtifacts


TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


class ProductRecommender:
    def __init__(self, artifacts: ModelArtifacts):
        self.artifacts = artifacts

        self.word2vec_model = artifacts.word2vec_model
        self.fasttext_model = artifacts.fasttext_model

        self.word2vec_vectors = artifacts.word2vec_vectors
        self.fasttext_vectors = artifacts.fasttext_vectors
        self.catalogue = artifacts.catalogue

        self.product_id_to_row = {
            product_id: row_index
            for row_index, product_id in enumerate(
                self.catalogue["product_id"].tolist()
            )
        }

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Apply lightweight tokenization to an incoming query."""
        return TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))

        if not np.isfinite(norm) or norm == 0:
            raise ValueError(
                "The query could not be converted into a valid vector."
            )

        return vector / norm

    def _word2vec_query_vector(self, tokens: list[str]) -> np.ndarray:
        vectors = [
            self.word2vec_model.wv[token]
            for token in tokens
            if token in self.word2vec_model.wv.key_to_index
        ]

        if not vectors:
            raise ValueError(
                "The query contains no Word2Vec vocabulary words."
            )

        return self.normalize_vector(np.mean(vectors, axis=0))

    def _fasttext_query_vector(self, tokens: list[str]) -> np.ndarray:
        # Gensim FastText can construct vectors for unseen words
        # using learned character subword information.
        vectors = [self.fasttext_model.wv[token] for token in tokens]

        return self.normalize_vector(np.mean(vectors, axis=0))

    @staticmethod
    def _clean_value(value):
        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            return value.item()

        return value

    def _product_metadata(self, row_index: int) -> dict:
        row = self.catalogue.iloc[row_index]

        # Keep useful catalogue fields without returning image URLs,
        # links, or every internal training column by default.
        preferred_columns = [
            "product_id",
            "name",
            "name_clean",
            "main_category",
            "main_category_clean",
            "sub_category",
            "sub_category_clean",
            "ratings",
            "no_of_ratings",
            "discount_price",
            "actual_price",
            "link",
            "image",
        ]

        metadata = {}

        for column in preferred_columns:
            if column in self.catalogue.columns:
                value = self._clean_value(row[column])

                if value is not None:
                    metadata[column] = value

        # Ensure a usable display name even if the source column
        # has a different name from the preferred columns.
        if not any(
            metadata.get(column)
            for column in ("name", "name_clean")
        ):
            for column in self.catalogue.columns:
                if "name" in column.lower():
                    value = self._clean_value(row[column])
                    if value is not None:
                        metadata["product_name"] = value
                        break

        return metadata

    def _rank(
        self,
        query_vector: np.ndarray,
        matrix: np.ndarray,
        top_n: int,
        excluded_row: int | None = None,
    ) -> list[dict]:
        # The query and catalogue rows are normalized, so dot product
        # gives cosine similarity.
        scores = matrix @ query_vector

        if excluded_row is not None:
            scores[excluded_row] = -np.inf

        # Return only positive-similarity recommendations.
        eligible = np.flatnonzero(scores > 0)

        if eligible.size == 0:
            return []

        ranked_rows = eligible[
            np.argsort(scores[eligible])[::-1][:top_n]
        ]

        results = []

        for rank, row_index in enumerate(ranked_rows, start=1):
            results.append(
                {
                    "rank": rank,
                    "similarity_score": round(
                        float(scores[row_index]), 6
                    ),
                    "product": self._product_metadata(
                        int(row_index)
                    ),
                }
            )

        return results

    def recommend_by_text(
        self,
        query: str,
        top_n: int = 5,
    ) -> dict:
        tokens = self.tokenize(query)

        if not tokens:
            raise ValueError(
                "Enter a query containing at least one word."
            )

        known_tokens = [
            token
            for token in tokens
            if token in self.word2vec_model.wv.key_to_index
        ]

        # Use Word2Vec only if every query token is known.
        if len(known_tokens) == len(tokens):
            model_used = "word2vec_skipgram"
            query_vector = self._word2vec_query_vector(tokens)
            matrix = self.word2vec_vectors
        else:
            model_used = "fasttext"
            query_vector = self._fasttext_query_vector(tokens)
            matrix = self.fasttext_vectors

        recommendations = self._rank(
            query_vector=query_vector,
            matrix=matrix,
            top_n=top_n,
        )

        return {
            "input_type": "text",
            "query": query,
            "model_used": model_used,
            "unknown_tokens": [
                token for token in tokens
                if token not in self.word2vec_model.wv.key_to_index
            ],
            "recommendations": recommendations,
        }

    def recommend_by_product_id(
        self,
        product_id: str,
        top_n: int = 5,
    ) -> dict:
        row_index = self.product_id_to_row.get(product_id)

        if row_index is None:
            raise KeyError(
                f"Product ID '{product_id}' was not found in the catalogue."
            )

        query_vector = self.word2vec_vectors[row_index]

        recommendations = self._rank(
            query_vector=query_vector,
            matrix=self.word2vec_vectors,
            top_n=top_n,
            excluded_row=row_index,
        )

        return {
            "input_type": "product_id",
            "product_id": product_id,
            "model_used": "word2vec_skipgram",
            "recommendations": recommendations,
        }
