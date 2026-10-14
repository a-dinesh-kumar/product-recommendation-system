# Product Similarity Search

A semantic product discovery application that finds products similar to a user's natural-language query. It uses Word2Vec Skip-Gram embeddings for in-vocabulary queries and FastText embeddings when a query contains out-of-vocabulary words, then ranks catalogue items using cosine similarity.

> **Project status:** Local API and frontend integration are in progress. Confirm endpoint response fields, dataset columns, tests, and deployment status before describing the project as production-ready.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Project structure](#project-structure)
- [Dataset and artifacts](#dataset-and-artifacts)
- [Setup](#setup)
- [Run the application](#run-the-application)
- [API reference](#api-reference)
- [Frontend](#frontend)
- [Testing checklist](#testing-checklist)
- [Known limitations](#known-limitations)
- [Future enhancements](#future-enhancements)
- [Author](#author)

## Overview

Exact keyword search can miss relevant products when users describe their needs differently from the words in a product title. This project explores semantic similarity to retrieve products whose text representations are close to a natural-language query.

The application supports searching by text, requesting a configurable number of recommendations, looking up similar items using an internal catalogue ID, and displaying ranked results with available product details.

The project uses a retail catalogue of approximately 70,000 products. The generated `product_id` is an internal identifier for this catalogue; it is **not** an external marketplace product ID.

## Features

- Semantic text search using Word2Vec Skip-Gram embeddings.
- FastText fallback when a query contains out-of-vocabulary words.
- Cosine similarity for ranking candidate products.
- Configurable top-N recommendations.
- Product-ID recommendations for items in the local catalogue.
- FastAPI backend with interactive API documentation.
- Minimalist, responsive HTML/CSS/JavaScript frontend.
- Small catalogue preview before searching.
- Available product metadata such as title, category, brand, price, image, and similarity score.
- GitHub and LinkedIn profile links in the frontend.

## How it works

1. A user enters a product name or description.
2. The API tokenizes the query and creates a vector representation.
3. If every query token exists in the Word2Vec vocabulary, the query is represented using Word2Vec Skip-Gram.
4. If any token is out of vocabulary, FastText represents the query using character-subword information.
5. The query vector is compared with compatible catalogue vectors using cosine similarity.
6. The highest-scoring eligible products are returned. Product-ID searches exclude the source item.

**Important:** Word2Vec and FastText vectors belong to different embedding spaces. A FastText query must be compared with FastText catalogue vectors, not Word2Vec vectors.

## Architecture

```text
Browser
  |
  v
FastAPI
  +--> Catalogue preview endpoint
  +--> Text recommendation endpoint
  |      +--> Tokenize query
  |      +--> Select Word2Vec or FastText
  |      +--> Build query vector
  |      +--> Rank by cosine similarity
  +--> Product-ID recommendation endpoint
         +--> Resolve internal product_id
         +--> Retrieve product vector
         +--> Rank similar products
  |
  v
Ranked JSON response
  |
  v
Frontend product cards
```

## Technology stack

| Area | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| ASGI server | Uvicorn |
| Data processing | pandas, NumPy |
| Embeddings | Gensim Word2Vec and FastText |
| Similarity | Cosine similarity / normalized vector dot product |
| Frontend | HTML, CSS, JavaScript |
| API exploration | FastAPI Swagger UI |

Use the versions pinned in `requirements.txt` as the installation baseline. Verify that they match the environment used to create the model artifacts.

## Project structure

```text
product_recommendation_system/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── model_loader.py
│   └── recommender.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── icons/
│       ├── github.png
│       └── linkedin.png
├── production_artifacts/
│   ├── word2vec_skipgram_full.model
│   ├── fasttext_full.model
│   ├── product_vectors.npy
│   ├── fasttext_product_vectors.npy
│   ├── production_catalogue.csv
│   ├── model_config.json
│   └── hybrid_config.json
├── tests/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

This is the intended structure; update it if your actual filenames differ. Never commit `.env` secrets, virtual environments, caches, or logs.

## Dataset and artifacts

The project was developed using a retail catalogue containing approximately 70,000 rows.

| Artifact | Purpose |
|---|---|
| `word2vec_skipgram_full.model` | Trained Word2Vec Skip-Gram model |
| `fasttext_full.model` | FastText model for OOV queries |
| `product_vectors.npy` | Catalogue vectors for Word2Vec |
| `fasttext_product_vectors.npy` | Catalogue vectors for FastText |
| `production_catalogue.csv` | Catalogue metadata and internal IDs |
| `model_config.json` | Word2Vec configuration metadata |
| `hybrid_config.json` | Hybrid-model configuration metadata |

**Artifact alignment matters:** each vector row must correspond to the same product row in `production_catalogue.csv`. Do not independently sort, filter, or reorder one file after exporting the artifacts.

### Handling large or restricted files

Model binaries and NumPy arrays can make a Git repository large. Inspect file sizes and consider Git LFS or an artifact-hosting service if needed. Do not publish the raw catalogue if you do not have permission to redistribute it. Document a secure download/setup process instead.

## Setup

### Prerequisites

- Python compatible with the pinned dependencies and saved model artifacts.
- `pip`
- Git
- The required production artifacts.

### 1. Clone the repository

Replace the URL with your actual GitHub repository URL:

```bash
git clone https://github.com/a-dinesh-kumar/product-recommendation-system.git
cd product-recommendation-system
```

Use the exact repository name you create on GitHub.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, confirm that the pinned versions match the model-artifact environment. Do not copy a full Google Colab `pip freeze` output into local requirements if it contains Colab-only packages.

### 4. Verify artifacts

Make sure all files expected by `app/config.py` exist in `production_artifacts/`. Startup may fail if a model, vector array, configuration file, or catalogue is missing or incompatible.

## Run the application

From the project root:

```bash
uvicorn app.main:app --reload
```

Open:

- Frontend: <http://127.0.0.1:8000/>
- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- Health check: <http://127.0.0.1:8000/health>
- Catalogue preview: <http://127.0.0.1:8000/catalogue/featured?limit=8>

The catalogue-preview URL is available only if the endpoint has been added to `app/main.py`.

## API reference

These are the intended routes. Use `/docs` to verify the actual request and response schemas in the running application.

### `GET /health`

Checks whether the service is available and its artifacts have loaded.

### `POST /recommend/by-text`

Finds similar products from a natural-language query.

Example request:

```json
{
  "query": "lightweight running shoes for daily use",
  "top_n": 5
}
```

Expected behavior: accept a product description and return ranked recommendations, similarity scores, and model information where implemented. FastText is used for queries containing out-of-vocabulary words.

### `POST /recommend/by-id`

Finds products similar to a catalogue item.

```json
{
  "product_id": "PROD_REPLACE_WITH_AN_EXISTING_ID",
  "top_n": 5
}
```

Replace the example with an ID that exists in the local catalogue. This internal ID is not a marketplace ID.

### `POST /recommend`

Combined endpoint accepting either a text query or internal product ID, according to the implemented request schema.

### `GET /catalogue/featured?limit=8`

Returns a small selection of products for the pre-search frontend view. This endpoint must be implemented in `app/main.py` for the catalogue preview to work.

## Frontend

- `frontend/index.html` — page structure.
- `frontend/style.css` — visual styling and responsive layout.
- `frontend/app.js` — API calls, search state, and product-card rendering.
- `frontend/icons/github.png` and `frontend/icons/linkedin.png` — social profile icons.

The UI should display available metadata such as title, category, brand, price, image, and similarity score. Missing fields should be handled gracefully.

## Testing checklist

- [ ] The application starts with the expected artifacts loaded.
- [ ] `GET /health` reports a healthy status.
- [ ] The home page loads without JavaScript or CSS errors.
- [ ] GitHub and LinkedIn icons open the intended profiles.
- [ ] Featured catalogue items appear before searching.
- [ ] Clearing the query resets recommendations and restores the catalogue preview.
- [ ] Text search returns results for a normal description.
- [ ] Text search works when a query includes an out-of-vocabulary word.
- [ ] The top-N setting controls the number of recommendations.
- [ ] Result cards display titles and available metadata correctly.
- [ ] Similarity scores use the expected score scale.
- [ ] Unknown product IDs return a clear error.
- [ ] Existing API endpoints continue working after frontend changes.
- [ ] Vector rows remain aligned with catalogue rows.

## Known limitations

- Product titles are the primary text source if a dedicated description column is unavailable.
- Similarity scores are not probabilities or guarantees of relevance.
- Category-based evaluation is only a proxy for relevance; stronger evaluation requires labelled or human-reviewed query-product relevance judgments.
- Results depend on the vocabulary and quality of the training corpus.
- Images may not load if URLs are missing, invalid, inaccessible, or blocked by the remote host.
- Internal catalogue IDs are not external marketplace IDs.
- Model sizes may affect startup time and hosting limits.

## Future enhancements

- Add automated tests for API contracts, OOV queries, ID lookup, and ranking.
- Build a human-reviewed relevance test set and report metrics such as Precision@K and Recall@K.
- Add caching and performance measurements.
- Add category, brand, and price filters where the data supports them.
- Consider external product URL lookup as a separate future feature.
- Deploy the application and add the live URL after verifying deployment.

---
## Author

**Dinesh Kumar Alagarsamy**  
*Software Quality Engineer / SDET*
-   GitHub: https://github.com/a-dinesh-kumar
-   LinkedIn: https://www.linkedin.com/in/dinesh-kumar-alagarsamy

### Areas of Interest:
* Quality Engineering
* Test Automation
* API Testing
* Artificial Intelligence
* Machine Learning
* AI-powered Testing
* Software Architecture

------------------------------------------------------------------------

## License

This project is intended for educational, portfolio, and demonstration
purposes.
