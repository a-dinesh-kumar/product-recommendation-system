
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.model_loader import load_artifacts
from app.recommender import ProductRecommender
from app.schemas import (
    RecommendationRequest,
    TextRecommendationRequest,
    ProductIdRequest,
)

import re
from pathlib import Path

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models and catalogue once, when the API starts.
    print("Starting Product Recommendation API...")

    artifacts = load_artifacts()
    app.state.recommender = ProductRecommender(artifacts)

    print("All production artifacts loaded successfully.")

    yield

    # Release references when the application shuts down.
    app.state.recommender = None
    print("Product Recommendation API stopped.")


app = FastAPI(
    title="Product Recommendation Similarity System",
    description=(
        "Hybrid product similarity search using Word2Vec Skip-Gram "
        "and FastText fallback for out-of-vocabulary text."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static"
)


def get_recommender(request: Request) -> ProductRecommender:
    recommender = getattr(request.app.state, "recommender", None)

    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation models are not loaded.",
        )

    return recommender


def handle_product_id_error(error: KeyError):
    raise HTTPException(
        status_code=404,
        detail=str(error).strip("'"),
    )


def handle_query_error(error: ValueError):
    raise HTTPException(
        status_code=422,
        detail=str(error),
    )


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    recommender = getattr(request.app.state, "recommender", None)

    return {
        "status": "healthy" if recommender is not None else "not_ready",
        "models_loaded": recommender is not None,
    }


@app.post("/recommend")
def recommend(
    payload: RecommendationRequest,
    request: Request,
):
    """Recommend products using either a product ID or free text."""

    recommender = get_recommender(request)

    if payload.product_id is not None:
        try:
            return recommender.recommend_by_product_id(
                product_id=payload.product_id.strip(),
                top_n=payload.top_n,
            )
        except KeyError as error:
            handle_product_id_error(error)

    try:
        return recommender.recommend_by_text(
            query=payload.query.strip(),
            top_n=payload.top_n,
        )
    except ValueError as error:
        handle_query_error(error)


@app.post("/recommend/by-text")
def recommend_by_text(
    payload: TextRecommendationRequest,
    request: Request,
):
    """Search for similar products using a free-text query."""

    recommender = get_recommender(request)

    try:
        return recommender.recommend_by_text(
            query=payload.query.strip(),
            top_n=payload.top_n,
        )
    except ValueError as error:
        handle_query_error(error)


@app.post("/recommend/by-id")
def recommend_by_id(
    payload: ProductIdRequest,
    request: Request,
):
    """Search for similar products using an existing product ID."""

    recommender = get_recommender(request)

    try:
        return recommender.recommend_by_product_id(
            product_id=payload.product_id.strip(),
            top_n=payload.top_n,
        )
    except KeyError as error:
        handle_product_id_error(error)

@app.get("/catalogue/featured")
def get_featured_products(
    limit: int = Query(default=8, ge=1, le=12)
):
    """
    Return a small catalogue preview for the frontend.
    This endpoint does not run the recommendation model.
    """

    recommender = getattr(app.state, "recommender", None)

    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="The recommendation system is not ready."
        )

    catalogue = getattr(recommender, "catalogue", None)

    if catalogue is None or catalogue.empty:
        raise HTTPException(
            status_code=503,
            detail="The product catalogue is unavailable."
        )

    # Match common column names without assuming a particular naming style.
    normalized_columns = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in catalogue.columns
    }

    def find_column(aliases):
        for alias in aliases:
            actual_column = normalized_columns.get(
                re.sub(r"[^a-z0-9]", "", alias.lower())
            )
            if actual_column is not None:
                return actual_column
        return None

    columns = {
        "id": find_column(["product_id", "id", "catalogue_id"]),
        "title": find_column([
            "product_name", "product_title", "title", "name"
        ]),
        "category": find_column([
            "category", "main_category", "product_category"
        ]),
        "brand": find_column(["brand", "brand_name"]),
        "price": find_column([
            "price", "discounted_price", "selling_price", "actual_price"
        ]),
        "image_url": find_column([
            "image_url", "image", "image_link", "product_image",
            "product_image_url", "thumbnail", "img_url"
        ]),
    }

    if columns["title"] is None:
        raise HTTPException(
            status_code=500,
            detail="No recognised product title column was found."
        )

    # Remove rows without a title and repeated titles for a cleaner preview.
    frame = catalogue[
        catalogue[columns["title"]].notna()
    ].copy()

    frame = frame.drop_duplicates(
        subset=[columns["title"]]
    )

    # Try to show different categories first, when category data exists.
    if columns["category"] is not None:
        diverse = frame.groupby(
            columns["category"],
            sort=False,
            dropna=False
        ).head(1)

        remaining = frame.drop(index=diverse.index)
        frame = __import__("pandas").concat([diverse, remaining])

    selected = frame.head(limit)
    products = []

    def clean_value(value):
        if value is None:
            return None

        try:
            if bool(__import__("pandas").isna(value)):
                return None
        except (TypeError, ValueError):
            pass

        # Convert NumPy scalar values into ordinary Python values.
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, TypeError):
                pass

        return value

    for _, row in selected.iterrows():
        product = {}

        for output_key, column in columns.items():
            if column is not None:
                product[output_key] = clean_value(row[column])

        products.append(product)

    return {
        "products": products,
        "total_catalogue_products": len(catalogue)
    }