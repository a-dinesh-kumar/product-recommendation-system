
from typing import Any

from pydantic import BaseModel, Field, model_validator


class RecommendationRequest(BaseModel):
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description="Free-text product description or search query.",
    )
    product_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Existing product ID from the catalogue.",
    )
    top_n: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_input(self):
        has_query = bool(self.query and self.query.strip())
        has_product_id = bool(
            self.product_id and self.product_id.strip()
        )

        if has_query == has_product_id:
            raise ValueError(
                "Provide exactly one of 'query' or 'product_id'."
            )

        return self


class TextRecommendationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_n: int = Field(default=5, ge=1, le=20)


class ProductIdRequest(BaseModel):
    product_id: str = Field(min_length=1, max_length=200)
    top_n: int = Field(default=5, ge=1, le=20)


class RecommendationResponse(BaseModel):
    input_type: str
    model_used: str
    recommendations: list[dict[str, Any]]
