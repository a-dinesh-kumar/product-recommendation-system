
const API = {
    featured: "/catalogue/featured",
    search: "/recommend/by-text"
};

const elements = {
    searchForm: document.getElementById("searchForm"),
    query: document.getElementById("query"),
    topN: document.getElementById("topN"),
    searchBtn: document.getElementById("searchBtn"),
    status: document.getElementById("statusMessage"),
    featuredSection: document.getElementById("featuredSection"),
    featuredGrid: document.getElementById("featuredGrid"),
    resultsSection: document.getElementById("resultsSection"),
    resultsGrid: document.getElementById("resultsGrid"),
    resultsDescription: document.getElementById("resultsDescription"),
    modelBadge: document.getElementById("modelBadge")
};


function firstValue(object, keys) {
    if (!object || typeof object !== "object") return null;

    for (const key of keys) {
        const value = object[key];

        if (value === undefined || value === null || value === "") {
            continue;
        }

        // Some API responses wrap a field inside an object.
        if (typeof value === "object" && !Array.isArray(value)) {
            const nested = firstValue(value, [
                "name", "title", "value", "label", "text",
                "url", "src", "amount", "display"
            ]);

            if (nested !== null) return nested;
            continue;
        }

        if (typeof value !== "object" && String(value).trim() !== "") {
            return value;
        }
    }

    return null;
}

function normalizeProduct(item) {
    // Recommendations may wrap the actual catalogue record inside
    // product, item, details, metadata, or a similar property.
    const nestedProduct =
        item?.product ??
        item?.product_details ??
        item?.productDetails ??
        item?.details ??
        item?.metadata ??
        item?.item ??
        item;

    const product = (
        nestedProduct &&
        typeof nestedProduct === "object" &&
        !Array.isArray(nestedProduct)
    ) ? nestedProduct : item;

    const read = (keys) =>
        firstValue(product, keys) ?? firstValue(item, keys);

    return {
        id: read(["product_id", "id", "catalogue_id", "catalog_id"]),
        title: read([
            "product_name", "product_title", "title", "name",
            "product", "description"
        ]) || "Untitled product",
        category: read([
            "category", "main_category", "product_category"
        ]),
        brand: read(["brand", "brand_name"]),
        price: read([
            "price", "discounted_price", "selling_price",
            "actual_price", "discount_price"
        ]),
        image: read([
            "image_url", "image_link", "product_image_url",
            "product_image", "image", "thumbnail", "img_url"
        ]),
        score: firstValue(item, [
            "similarity_score", "similarity", "score",
            "cosine_similarity"
        ]) ?? read([
            "similarity_score", "similarity", "score",
            "cosine_similarity"
        ])
    };
}


function extractProducts(data) {
    if (Array.isArray(data)) {
        return data;
    }

    const possibleLists = [
        data?.recommendations,
        data?.results,
        data?.products,
        data?.items,
        data?.matches
    ];

    return possibleLists.find(Array.isArray) || [];
}

function setStatus(message = "", isError = false) {
    elements.status.textContent = message;
    elements.status.hidden = !message;
    elements.status.classList.toggle("error", isError);
}

function formatPrice(price) {
    if (price === null || price === undefined || String(price).trim() === "") {
        return "";
    }

    return String(price);
}

function formatSimilarity(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return null;
    }

    return `${(number * 100).toFixed(1)}%`;
}

function createImagePlaceholder() {
    const placeholder = document.createElement("div");
    placeholder.className = "image-placeholder";
    placeholder.setAttribute("aria-label", "Product image unavailable");

    // This is static SVG markup, not data from the API.
    placeholder.innerHTML = `
        <svg viewBox="0 0 48 48" fill="none"
             stroke="currentColor" stroke-width="1.5"
             stroke-linecap="round" stroke-linejoin="round"
             aria-hidden="true">
            <rect x="7" y="9" width="34" height="30" rx="4"></rect>
            <circle cx="18" cy="19" r="3"></circle>
            <path d="M8 33l10-9 7 6 6-5 9 8"></path>
        </svg>
    `;

    return placeholder;
}


function createProductCard(rawProduct, options = {}) {
    const product = normalizeProduct(rawProduct);
    const card = document.createElement("article");
    card.className = "product-card";

    const imageWrap = document.createElement("div");
    imageWrap.className = "product-image-wrap";

    if (options.rank) {
        const rank = document.createElement("span");
        rank.className = "product-rank";
        rank.textContent = `#${options.rank}`;
        imageWrap.appendChild(rank);
    }

    const imageUrl = typeof product.image === "string"
        ? product.image.trim()
        : "";

    let imageAdded = false;

    if (imageUrl) {
        try {
            const parsedUrl = new URL(imageUrl, window.location.origin);

            if (["http:", "https:"].includes(parsedUrl.protocol)) {
                const image = document.createElement("img");
                image.className = "product-image";
                image.src = parsedUrl.href;
                image.alt = product.title;
                image.loading = "lazy";
                image.referrerPolicy = "no-referrer";

                image.addEventListener("error", () => {
                    image.replaceWith(createImagePlaceholder());
                }, { once: true });

                imageWrap.appendChild(image);
                imageAdded = true;
            }
        } catch {
            // Fall back to the image placeholder below.
        }
    }

    if (!imageAdded) {
        imageWrap.appendChild(createImagePlaceholder());
    }

    const info = document.createElement("div");
    info.className = "product-info";

    if (product.category) {
        const category = document.createElement("span");
        category.className = "product-category";
        category.textContent = String(product.category);
        info.appendChild(category);
    }

    const title = document.createElement("h3");
    title.className = "product-title";
    title.textContent = String(product.title);
    info.appendChild(title);

    if (product.brand) {
        const brand = document.createElement("p");
        brand.className = "product-brand";
        brand.textContent = String(product.brand);
        info.appendChild(brand);
    }

    const meta = document.createElement("div");
    meta.className = "product-meta";

    if (product.price !== null && product.price !== undefined) {
        const price = document.createElement("span");
        price.className = "product-price";

        const rawPrice = String(product.price).trim();

        // Preserve prices that already contain a currency symbol.
        price.textContent = /[₹$€£]/.test(rawPrice)
            ? rawPrice
            : `₹${rawPrice}`;

        meta.appendChild(price);
    }

    // Only recommendation cards display similarity scores.
    if (options.showScore) {
        const scoreValue = Number(product.score);

        if (product.score !== null && Number.isFinite(scoreValue)) {
            const score = document.createElement("span");
            score.className = "similarity";

            // Similarity APIs may return either 0.834 or 83.4.
            const percentage = Math.abs(scoreValue) <= 1
                ? scoreValue * 100
                : scoreValue;

            score.textContent = `Match ${percentage.toFixed(1)}%`;
            score.title = "Cosine similarity score";
            meta.appendChild(score);
        }
    }

    if (meta.childElementCount > 0) {
        info.appendChild(meta);
    }

    card.append(imageWrap, info);
    return card;
}



function renderProducts(container, products, options = {}) {
    container.replaceChildren();

    if (!products.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = options.emptyMessage ||
            "No products are available to display right now.";
        container.appendChild(empty);
        return;
    }

    products.forEach((product, index) => {
        container.appendChild(
            createProductCard(product, {
                rank: options.showRank ? index + 1 : null,
                showScore: options.showScore === true
            })
        );
    });
}


async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: {
            Accept: "application/json",
            ...(options.body ? { "Content-Type": "application/json" } : {}),
            ...options.headers
        }
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        const detail = data.detail || data.message || data.error;
        throw new Error(
            typeof detail === "string"
                ? detail
                : `Request failed with status ${response.status}.`
        );
    }

    return data;
}

async function loadFeaturedProducts() {
    elements.featuredGrid.replaceChildren();

    const loading = document.createElement("div");
    loading.className = "loading-card";
    loading.textContent = "Loading products from the catalogue…";
    elements.featuredGrid.appendChild(loading);

    try {
        const data = await fetchJson(`${API.featured}?limit=8`);
        const products = extractProducts(data);

        renderProducts(elements.featuredGrid, products, {
            showScore: false,
            emptyMessage: "No featured products are available."
        });
    } catch (error) {
        renderProducts(elements.featuredGrid, [], {
            emptyMessage:
                "The catalogue preview could not be loaded. You can still try a product search."
        });
        console.error("Featured catalogue request failed:", error);
    }
}

async function searchProducts(event) {
    event.preventDefault();

    const query = elements.query.value.trim();
    const topN = Number(elements.topN.value);

    if (query.length < 2) {
        setStatus("Please enter a product name or a short description.", true);
        elements.query.focus();
        return;
    }

    elements.searchBtn.disabled = true;
    elements.searchBtn.querySelector("span").textContent = "Searching…";
    elements.resultsSection.hidden = false;
    elements.resultsGrid.replaceChildren();

    const loading = document.createElement("div");
    loading.className = "loading-card";
    loading.textContent = "Finding the most similar products…";
    elements.resultsGrid.appendChild(loading);

    setStatus("Comparing your query with the product catalogue.");

    try {
        // Keep the request aligned with the existing TextRecommendationRequest.
        const data = await fetchJson(API.search, {
            method: "POST",
            body: JSON.stringify({
                query,
                top_n: topN
            })
        });

        const products = extractProducts(data);
        const modelUsed = firstValue(data, [
            "model_used", "model", "selected_model"
        ]);

        elements.resultsDescription.textContent =
            `${products.length} recommendation${products.length === 1 ? "" : "s"} for “${query}”.`;

        elements.modelBadge.textContent = modelUsed
            ? String(modelUsed).replaceAll("_", " ")
            : "Similarity search";

        renderProducts(elements.resultsGrid, products, {
            showRank: true,
            showScore: true,
            emptyMessage:
                "No matching products were returned. Try a broader description."
        });

        elements.featuredSection.hidden = true;
        setStatus(products.length
            ? "Recommendations are ranked by their similarity scores."
            : "No recommendations were found. Try another product description.");
        elements.resultsSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    } catch (error) {
        console.error("Product search failed:", error);
        renderProducts(elements.resultsGrid, [], {
            emptyMessage: "Search could not be completed."
        });
        setStatus(
            `${error.message} Check that the FastAPI service is running and the request schema matches.`,
            true
        );
    } finally {
        elements.searchBtn.disabled = false;
        elements.searchBtn.querySelector("span").textContent = "Find matches";
    }
}


function resetSearchResults() {
    elements.resultsSection.hidden = true;
    elements.resultsGrid.replaceChildren();
    elements.resultsDescription.textContent = "";
    elements.modelBadge.textContent = "Similarity search";

    setStatus("");
    elements.featuredSection.hidden = false;

    elements.searchBtn.disabled = false;
    elements.searchBtn.querySelector("span").textContent = "Find matches";
}

elements.query.addEventListener("input", () => {
    if (!elements.query.value.trim()) {
        resetSearchResults();
    }
});

elements.searchForm.addEventListener("submit", searchProducts);

loadFeaturedProducts();
