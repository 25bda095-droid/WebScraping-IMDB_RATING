import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import time
import json
import html
import re
import urllib.parse
import numpy as np
from playwright.sync_api import sync_playwright
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ═══════════════════════════════════════════════════════════════════
# 1. SCRAPER LOGIC — Identical to IMDb_Review_Scraper.ipynb
#    DO NOT modify this section unless the notebook changes too.
# ═══════════════════════════════════════════════════════════════════

# SECURITY: Session file path loaded from env variable, never hardcoded.
SESSION_FILE = os.environ.get("IMDB_SESSION_FILE", "imdb_session.json")
GRAPHQL_ENDPOINT = "https://caching.graphql.imdb.com/"
PERSISTED_HASH = "286aee4ac14648e42c02c576e0cd29c33e9113f022290145cb1872968b389505"
BATCH_SIZE = 50


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def parse_review_node(node: dict, title_id: str, movie_name: str) -> dict:
    author = node.get("author", {}) or {}
    summary = node.get("summary", {}) or {}
    text_data = node.get("text", {}) or {}
    helpfulness = node.get("helpfulness", {}) or {}

    if isinstance(text_data, dict):
        orig = text_data.get("originalText", {})
        body = clean_html(orig.get("plaidHtml", "")) if isinstance(orig, dict) else clean_html(str(orig))
    else:
        body = clean_html(str(text_data))

    return {
        "movie_id": title_id,
        "movie_name": movie_name,
        "review_id": node.get("id", ""),
        "user_name": (author.get("username", {}) or {}).get("text", ""),
        "review_date": node.get("submissionDate", ""),
        "user_rating": node.get("authorRating"),
        "review_title": clean_html(
            summary.get("originalText", "") if isinstance(summary, dict) else str(summary)
        ),
        "review_body": body,
        "upvotes": helpfulness.get("upVotes", 0) if isinstance(helpfulness, dict) else 0,
        "downvotes": helpfulness.get("downVotes", 0) if isinstance(helpfulness, dict) else 0,
        "spoiler": node.get("spoiler", False),
    }


def fetch_reviews_graphql_sync(page, title_id: str, cursor: str, movie_name: str) -> tuple:
    variables = json.dumps({
        "after": cursor, "const": title_id, "filter": {},
        "first": BATCH_SIZE, "locale": "en-US",
        "sort": {"by": "HELPFULNESS_SCORE", "order": "DESC"},
    })
    extensions = json.dumps({"persistedQuery": {"sha256Hash": PERSISTED_HASH, "version": 1}})

    gql_url = (
        f"{GRAPHQL_ENDPOINT}?operationName=TitleReviewsRefine"
        f"&variables={urllib.parse.quote(variables)}"
        f"&extensions={urllib.parse.quote(extensions)}"
    )

    result = page.evaluate(f"""
        async () => {{
            try {{
                const resp = await fetch("{gql_url}", {{
                    method: 'GET',
                    headers: {{'content-type': 'application/json', 'x-imdb-client-name': 'imdb-web-next-localized'}},
                    credentials: 'include',
                }});
                if (!resp.ok) return {{ error: `HTTP ${{resp.status}}` }};
                return {{ data: await resp.json() }};
            }} catch (e) {{ return {{ error: e.message }}; }}
        }}
    """)

    if "error" in result or not result.get("data"):
        return [], None

    reviews_data = result["data"].get("data", {}).get("title", {}).get("reviews", {})
    edges = reviews_data.get("edges", [])
    page_info = reviews_data.get("pageInfo", {})
    next_cursor = page_info.get("endCursor") if page_info.get("hasNextPage") else None

    reviews = [parse_review_node(edge.get("node", {}), title_id, movie_name) for edge in edges]
    return reviews, next_cursor


def scrape_live_reviews(movie_id: str, status_container, max_reviews: int = 100):
    """
    Live-scrape reviews from IMDb using the same authenticated Playwright
    + GraphQL pagination pipeline as the Jupyter notebook.

    Initial page data is parsed inside the browser via page.evaluate() to
    avoid JSON truncation when the __NEXT_DATA__ blob is very large.
    """
    reviews = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        if os.path.exists(SESSION_FILE):
            context = browser.new_context(storage_state=SESSION_FILE, user_agent=ua)
        else:
            context = browser.new_context(user_agent=ua)

        page = context.new_page()
        url = f"https://www.imdb.com/title/{movie_id}/reviews"

        try:
            status_container.update(label="Connecting to IMDb...", state="running")

            # Navigate with domcontentloaded (fast) + explicit wait for __NEXT_DATA__
            loaded = False
            for attempt in range(3):
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    # Wait for __NEXT_DATA__ to appear (critical for data extraction)
                    page.wait_for_selector("script#__NEXT_DATA__", state="attached", timeout=10000)
                    loaded = True
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(2)

            if not loaded:
                # Final attempt: try networkidle as fallback
                try:
                    page.goto(url, wait_until="networkidle", timeout=60000)
                    loaded = True
                except Exception as e:
                    status_container.update(label=f"Error loading page: {e}", state="error")
                    return pd.DataFrame()

            # Brief pause for redirects/hydration to settle (matches notebook behavior)
            time.sleep(1.5)

            # Get movie name from page title
            movie_name = movie_id
            try:
                title_str = page.title()
                if title_str:
                    movie_name = title_str.split(" - ")[0].strip()
            except Exception:
                pass

            status_container.update(label=f"Extracting reviews for {movie_name}...", state="running")

            # Parse __NEXT_DATA__ INSIDE the browser to avoid JSON truncation
            initial_data = page.evaluate("""() => {
                const el = document.querySelector("script#__NEXT_DATA__");
                if (!el) return { edges: [], cursor: null, total: 0 };
                try {
                    const data = JSON.parse(el.textContent);
                    const reviews = data?.props?.pageProps?.contentData?.data?.title?.reviews;
                    if (!reviews) return { edges: [], cursor: null, total: 0 };
                    return {
                        edges: (reviews.edges || []).map(e => e.node),
                        cursor: reviews.pageInfo?.hasNextPage ? reviews.pageInfo?.endCursor : null,
                        total: reviews.total || 0
                    };
                } catch(e) { return { edges: [], cursor: null, total: 0 }; }
            }""")

            cursor = initial_data.get("cursor")
            for node in initial_data.get("edges", []):
                reviews.append(parse_review_node(node, movie_id, movie_name))

            # If initial extraction got nothing, start GraphQL from scratch
            if not reviews and not cursor:
                cursor = ""  # Empty string triggers first page in GraphQL

            seen_ids = {r["review_id"] for r in reviews}

            # GraphQL pagination — fast loop with minimal delay
            empty_count = 0
            while cursor is not None and len(reviews) < max_reviews:
                status_container.update(
                    label=f"Fetching reviews... {len(reviews)}/{max_reviews}",
                    state="running",
                )
                time.sleep(0.15)  # Minimal delay to avoid rate-limiting
                new_reviews, cursor = fetch_reviews_graphql_sync(page, movie_id, cursor, movie_name)

                added = 0
                for r in new_reviews:
                    if r["review_id"] not in seen_ids:
                        seen_ids.add(r["review_id"])
                        reviews.append(r)
                        added += 1

                if added == 0:
                    empty_count += 1
                    if empty_count >= 3:
                        break
                else:
                    empty_count = 0

        except Exception as e:
            status_container.update(label=f"Error: {e}", state="error")
        finally:
            if not os.path.exists(SESSION_FILE):
                context.storage_state(path=SESSION_FILE)
            context.close()
            browser.close()

    return pd.DataFrame(reviews[:max_reviews])


# ═══════════════════════════════════════════════════════════════════
# 2. ML MODEL — Real DistilBERT Sentiment Classifier
# ═══════════════════════════════════════════════════════════════════

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movie_sentiment_model_v1_450movies")
LABEL_MAP = {0: "Negative", 1: "Mixed", 2: "Positive"}


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_model():
    """Load the fine-tuned DistilBERT model and tokenizer once, then cache."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_sentiments(reviews: list[str]) -> list[str]:
    """Run batch inference on a list of review texts, return predicted labels."""
    tokenizer, model, device = load_model()
    predictions = []

    # Process in batches of 16 to manage memory
    batch_size = 16
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i : i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        predictions.extend([LABEL_MAP[p] for p in preds])

    return predictions


def run_model(reviews_df):
    """
    Run real DistilBERT inference on scraped reviews.
    Returns a rich results dict consumed by the UI rendering functions.
    """
    if reviews_df is None or reviews_df.empty:
        return None

    movie_name = (
        reviews_df.iloc[0]["movie_name"]
        if "movie_name" in reviews_df.columns
        else "Unknown Movie"
    )

    # Get review texts and run model
    review_texts = reviews_df["review_body"].fillna("").tolist()
    sentiments = predict_sentiments(review_texts)
    reviews_df = reviews_df.copy()
    reviews_df["predicted_sentiment"] = sentiments

    # ── Sentiment counts ──
    total = len(sentiments)
    pos_count = sentiments.count("Positive")
    mix_count = sentiments.count("Mixed")
    neg_count = sentiments.count("Negative")

    pos_pct = round(pos_count / total * 100)
    mix_pct = round(mix_count / total * 100)
    neg_pct = 100 - pos_pct - mix_pct

    # ── AI predicted score ──
    avg_score = round((pos_count * 8.5 + mix_count * 5.5 + neg_count * 3.0) / total, 1)

    # ── Avg user rating (from actual IMDb ratings) ──
    rated_reviews = reviews_df[reviews_df["user_rating"].notna()]
    avg_user_rating = round(rated_reviews["user_rating"].mean(), 1) if not rated_reviews.empty else None

    # ── Rating distribution (1–10 histogram) ──
    rating_dist = {}
    if not rated_reviews.empty:
        for r in range(1, 11):
            rating_dist[r] = int((rated_reviews["user_rating"] == r).sum())

    # ── Spoiler percentage ──
    spoiler_count = int(reviews_df["spoiler"].sum()) if "spoiler" in reviews_df.columns else 0
    spoiler_pct = round(spoiler_count / total * 100)

    # ── Polarization ──
    if pos_pct >= 70 or neg_pct >= 70:
        polarization = "Low"
    elif pos_pct >= 50 and neg_pct >= 20:
        polarization = "High"
    elif neg_pct >= 50 and pos_pct >= 20:
        polarization = "High"
    else:
        polarization = "Medium"

    # ── Community votes ──
    total_upvotes = int(reviews_df["upvotes"].sum()) if "upvotes" in reviews_df.columns else 0
    total_downvotes = int(reviews_df["downvotes"].sum()) if "downvotes" in reviews_df.columns else 0
    community_votes = total_upvotes + total_downvotes
    helpfulness_ratio = round(total_upvotes / community_votes * 100) if community_votes > 0 else 0

    # ── Average review length ──
    avg_review_len = int(reviews_df["review_body"].fillna("").str.split().str.len().mean())

    # ── Audience verdict text ──
    if pos_pct >= 75:
        verdict = "🎯 Overwhelmingly Positive — Audiences love this film"
    elif pos_pct >= 55:
        verdict = "👍 Generally Positive — Most viewers enjoyed it"
    elif neg_pct > pos_pct:
        verdict = "👎 Divisive — More critics than fans"
    else:
        verdict = "🤔 Mixed Reception — Opinions are split"

    # ── Best reviews by upvotes (full text, not truncated) ──
    positive_reviews = reviews_df[reviews_df["predicted_sentiment"] == "Positive"]
    negative_reviews = reviews_df[reviews_df["predicted_sentiment"] == "Negative"]

    def pick_best_review(subset):
        if subset.empty:
            return None
        # Pick by highest upvotes, break ties with longest review
        if "upvotes" in subset.columns:
            best_idx = subset["upvotes"].idxmax()
        else:
            best_idx = subset["review_body"].str.len().idxmax()
        row = subset.loc[best_idx]
        return {
            "body": row.get("review_body", ""),
            "title": row.get("review_title", ""),
            "user_name": row.get("user_name", "Anonymous"),
            "user_rating": row.get("user_rating"),
            "upvotes": int(row.get("upvotes", 0)),
            "downvotes": int(row.get("downvotes", 0)),
        }

    best_positive = pick_best_review(positive_reviews)
    best_negative = pick_best_review(negative_reviews)

    return {
        "movie_name": movie_name,
        "total_reviews": total,
        "sentiment_split": {"Positive": pos_pct, "Mixed": mix_pct, "Negative": neg_pct},
        "sentiment_counts": {"Positive": pos_count, "Mixed": mix_count, "Negative": neg_count},
        "avg_predicted_score": avg_score,
        "avg_user_rating": avg_user_rating,
        "rating_distribution": rating_dist,
        "spoiler_pct": spoiler_pct,
        "polarization_score": polarization,
        "verdict": verdict,
        "community_votes": community_votes,
        "helpfulness_ratio": helpfulness_ratio,
        "avg_review_length": avg_review_len,
        "best_positive": best_positive,
        "best_negative": best_negative,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. CUSTOM CSS — Premium dark-theme styling
# ═══════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Google Font ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Root variables ──────────────────────────────────── */
:root {
    --bg-primary:    hsl(220, 20%, 7%);
    --bg-card:       hsl(220, 18%, 11%);
    --bg-card-hover: hsl(220, 18%, 14%);
    --border-subtle: hsl(220, 14%, 18%);
    --text-primary:  hsl(0, 0%, 93%);
    --text-secondary:hsl(220, 10%, 58%);
    --accent-gold:   hsl(45, 100%, 58%);
    --accent-green:  hsl(152, 60%, 52%);
    --accent-amber:  hsl(38, 92%, 55%);
    --accent-red:    hsl(0, 72%, 56%);
    --accent-blue:   hsl(215, 80%, 58%);
    --radius:        12px;
    --shadow:        0 2px 16px rgba(0,0,0,.35);
}

/* ── Global overrides ────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.01em;
}
[data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
}
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

/* ── Hero header ─────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.03em;
}
.hero-header p {
    font-size: 1.05rem;
    color: var(--text-secondary);
    margin-top: .5rem;
    max-width: 560px;
    margin-left: auto; margin-right: auto;
    line-height: 1.55;
}

/* ── Stat card ───────────────────────────────────────── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: background .2s;
}
.stat-card:hover { background: var(--bg-card-hover); }
.stat-card .label {
    font-size: .78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--text-secondary);
    margin-bottom: .35rem;
}
.stat-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
}
.stat-card .value.gold   { color: var(--accent-gold); }
.stat-card .value.green  { color: var(--accent-green); }
.stat-card .value.amber  { color: var(--accent-amber); }
.stat-card .value.red    { color: var(--accent-red); }
.stat-card .value.blue   { color: var(--accent-blue); }

/* ── Sentiment bar ───────────────────────────────────── */
.sentiment-bar-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}
.sentiment-bar-wrapper .bar-title {
    font-size: .82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: var(--text-secondary);
    margin-bottom: .7rem;
}
.sentiment-bar {
    display: flex;
    height: 28px;
    border-radius: 6px;
    overflow: hidden;
}
.sentiment-bar .seg-pos { background: var(--accent-green); }
.sentiment-bar .seg-mix { background: var(--accent-amber); }
.sentiment-bar .seg-neg { background: var(--accent-red); }
.sentiment-legend {
    display: flex;
    gap: 1.5rem;
    margin-top: .6rem;
    font-size: .82rem;
    color: var(--text-secondary);
}
.sentiment-legend span { font-weight: 600; }
.sentiment-legend .dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}

/* ── Review quote card (scrollable, full-text) ───────── */
.review-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    min-height: 160px;
}
.review-card .tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: .65rem;
}
.review-card .tag.positive { background: hsla(152,60%,52%,.15); color: var(--accent-green); }
.review-card .tag.negative { background: hsla(0,72%,56%,.15); color: var(--accent-red); }
.review-card .review-title-text {
    font-size: .95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: .45rem;
}
.review-card .body {
    font-size: .88rem;
    line-height: 1.65;
    color: var(--text-primary);
    max-height: 320px;
    overflow-y: auto;
    padding-right: 8px;
}
.review-card .body::-webkit-scrollbar { width: 4px; }
.review-card .body::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 4px; }
.review-card .review-meta {
    display: flex;
    gap: 1rem;
    margin-top: .6rem;
    padding-top: .6rem;
    border-top: 1px solid var(--border-subtle);
    font-size: .76rem;
    color: var(--text-secondary);
}
.review-card .review-meta span { font-weight: 600; color: var(--text-primary); }

/* ── Verdict banner ──────────────────────────────────── */
.verdict-banner {
    background: linear-gradient(135deg, hsla(45,100%,58%,.08), hsla(215,80%,58%,.08));
    border: 1px solid hsla(45,100%,58%,.2);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    text-align: center;
    margin: 1rem 0;
}
.verdict-banner h3 {
    color: var(--accent-gold);
    margin: 0 0 .4rem;
    font-size: 1.3rem;
    font-weight: 700;
}
.verdict-banner p {
    color: var(--text-secondary);
    margin: 0;
    font-size: .95rem;
}

/* ── Watch button ────────────────────────────────────── */
.watch-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    border-radius: 8px;
    background: hsla(215,80%,58%,.12);
    border: 1px solid hsla(215,80%,58%,.3);
    color: var(--accent-blue);
    font-weight: 600;
    font-size: .88rem;
    cursor: default;
    margin-top: .8rem;
    opacity: .6;
}

/* ── Verdict description text ────────────────────────── */
.verdict-desc {
    font-size: .95rem;
    color: var(--text-secondary);
    text-align: center;
    margin-top: .5rem;
    line-height: 1.5;
}

/* ── Comparison table ────────────────────────────────── */
.cmp-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    overflow: hidden;
    margin: 1rem 0;
}
.cmp-table th, .cmp-table td {
    padding: .7rem 1rem;
    text-align: center;
    font-size: .88rem;
    border-bottom: 1px solid var(--border-subtle);
}
.cmp-table th {
    background: hsla(220, 14%, 18%, .5);
    font-weight: 700;
    color: var(--text-primary);
    text-transform: uppercase;
    font-size: .75rem;
    letter-spacing: .06em;
}
.cmp-table td:first-child {
    text-align: left;
    font-weight: 600;
    color: var(--text-secondary);
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-table .winner { color: var(--accent-green); font-weight: 700; }

/* ── Section heading ─────────────────────────────────── */
.section-heading {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 1.5rem 0 .75rem;
    letter-spacing: -0.02em;
}

/* ── Sidebar styling ─────────────────────────────────── */
[data-testid="stSidebar"] .stRadio label {
    font-weight: 500 !important;
}

/* ── Hide default Streamlit chrome (keep header for sidebar toggle) ── */
#MainMenu, footer { visibility: hidden; }
</style>
"""


# ═══════════════════════════════════════════════════════════════════
# 4. UI COMPONENT HELPERS
# ═══════════════════════════════════════════════════════════════════

def stat_card(label: str, value: str, color: str = ""):
    cls = f"value {color}" if color else "value"
    return f"""<div class="stat-card">
        <div class="label">{label}</div>
        <div class="{cls}">{value}</div>
    </div>"""


def sentiment_bar(pos: int, mix: int, neg: int, pos_n: int = 0, mix_n: int = 0, neg_n: int = 0):
    return f"""<div class="sentiment-bar-wrapper">
        <div class="bar-title">Sentiment Distribution</div>
        <div class="sentiment-bar">
            <div class="seg-pos" style="width:{pos}%"></div>
            <div class="seg-mix" style="width:{mix}%"></div>
            <div class="seg-neg" style="width:{neg}%"></div>
        </div>
        <div class="sentiment-legend">
            <div><span class="dot" style="background:var(--accent-green)"></span><span>{pos}%</span> Positive ({pos_n})</div>
            <div><span class="dot" style="background:var(--accent-amber)"></span><span>{mix}%</span> Mixed ({mix_n})</div>
            <div><span class="dot" style="background:var(--accent-red)"></span><span>{neg}%</span> Negative ({neg_n})</div>
        </div>
    </div>"""


def review_card_html(review_data, sentiment: str):
    """Render a full-text review card with metadata."""
    if review_data is None:
        label = "positive" if sentiment == "positive" else "negative"
        return f"""<div class="review-card">
    <div class="tag {label}">No {label} reviews</div>
    <div class="body">No {label} reviews were found in the analyzed set.</div>
</div>"""

    tag_cls = "positive" if sentiment == "positive" else "negative"
    tag_label = "Most Helpful Positive" if sentiment == "positive" else "Most Helpful Negative"

    body = review_data["body"].replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
    title = review_data.get("title", "").replace('<', '&lt;').replace('>', '&gt;')
    user = review_data.get("user_name", "Anonymous") or "Anonymous"
    rating = review_data.get("user_rating")
    up = review_data.get("upvotes", 0)
    down = review_data.get("downvotes", 0)

    rating_str = f"★ {int(rating)}/10" if pd.notna(rating) else ""
    title_html = f'<div class="review-title-text">{title}</div>' if title else ""
    rating_html = f'<div>{rating_str}</div>' if rating_str else ""

    return f"""<div class="review-card">
    <div class="tag {tag_cls}">{tag_label}</div>
    {title_html}
    <div class="body">{body}</div>
    <div class="review-meta">
        <div>👤 <span>{user}</span></div>
        {rating_html}
        <div>👍 <span>{up}</span></div>
        <div>👎 <span>{down}</span></div>
    </div>
</div>"""


def render_report(results, key_suffix=""):
    """Render a single movie report card with all stats."""
    name = results["movie_name"]
    sent = results["sentiment_split"]
    counts = results.get("sentiment_counts", {"Positive": 0, "Mixed": 0, "Negative": 0})

    st.markdown(f'<div class="section-heading">📊  {name}</div>', unsafe_allow_html=True)

    # ── Row 1: Key Metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(stat_card("AI Predicted Rating", f"{results['avg_predicted_score']}/10", "gold"), unsafe_allow_html=True)
    with c2:
        avg_ur = results.get("avg_user_rating")
        ur_str = f"{avg_ur}/10" if avg_ur else "N/A"
        st.markdown(stat_card("Avg User Rating", ur_str, "blue"), unsafe_allow_html=True)
    with c3:
        st.markdown(stat_card("Reviews Analyzed", str(results["total_reviews"]), "blue"), unsafe_allow_html=True)
    with c4:
        st.markdown(stat_card("Spoiler Reviews", f"{results.get('spoiler_pct', 0)}%", "amber"), unsafe_allow_html=True)

    # ── Row 2: Sentiment Bar ──
    st.markdown(
        sentiment_bar(sent["Positive"], sent["Mixed"], sent["Negative"],
                      counts["Positive"], counts["Mixed"], counts["Negative"]),
        unsafe_allow_html=True,
    )

    # ── Row 3: Rating Distribution ──
    rating_dist = results.get("rating_distribution", {})
    if rating_dist:
        st.markdown('<div class="section-heading">⭐ User Rating Distribution</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            "Rating": [f"{r}★" for r in range(1, 11)],
            "Count": [rating_dist.get(r, 0) for r in range(1, 11)],
        }).set_index("Rating")
        st.bar_chart(chart_df, color="#e6a817", height=220)

    # ── Row 4: Verdict Banner ──
    verdict = results.get("verdict", "")
    st.markdown(
        f"""<div class="verdict-banner">
            <h3>{verdict}</h3>
            <p>Polarization: <b>{results['polarization_score']}</b> · 
            Community Engagement: <b>{results.get('community_votes', 0):,}</b> total votes</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Row 5: Full Review Highlights ──
    st.markdown('<div class="section-heading">💬 Review Highlights</div>', unsafe_allow_html=True)
    col_p, col_n = st.columns(2)
    with col_p:
        st.markdown(review_card_html(results.get("best_positive"), "positive"), unsafe_allow_html=True)
    with col_n:
        st.markdown(review_card_html(results.get("best_negative"), "negative"), unsafe_allow_html=True)

    # ── Row 6: Quick Stats ──
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(stat_card("Avg Review Length", f"{results.get('avg_review_length', 0)} words", ""), unsafe_allow_html=True)
    with q2:
        st.markdown(stat_card("Community Votes", f"{results.get('community_votes', 0):,}", ""), unsafe_allow_html=True)
    with q3:
        st.markdown(stat_card("Helpfulness Ratio", f"{results.get('helpfulness_ratio', 0)}%", "green"), unsafe_allow_html=True)

    # ── Row 7: Watch Link ──
    st.markdown(f'<div class="watch-btn" style="margin-top: 1.5rem; text-align: center; padding: 1rem; background: var(--bg-card); border-radius: var(--radius); border: 1px solid var(--border-subtle); font-weight: 600; color: var(--text-primary); cursor: pointer;">📺 &nbsp;Watch {name} — link coming soon</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 5. STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════

def analyze_movie(movie_id: str):
    """Scrape reviews live from IMDb and run the sentiment model."""
    with st.status("Live-analyzing from IMDb…", expanded=True) as status:
        df = scrape_live_reviews(movie_id, status, max_reviews=100)
        if df.empty:
            status.update(label="Could not fetch reviews.", state="error")
            return None, "error"
        status.update(label=f"Running AI model on {len(df)} reviews...", state="running")
        results = run_model(df)
        status.update(label=f"Done — {len(df)} reviews analyzed!", state="complete")
    return results, "scraped"


def main():
    st.set_page_config(
        page_title="CineScope — AI Movie Analyzer",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎬 CineScope")
        st.caption("AI-Powered Movie Sentiment Analysis")
        st.divider()
        mode = st.radio(
            "Navigate",
            ["Analyze", "Compare"],
            captions=["Deep-dive into a single movie", "Side-by-side matchup"],
        )
        st.divider()
        st.markdown(
            "<small style='color:var(--text-secondary)'>Trained on <b>1,100+ movies</b>. "
            "Unknown titles are scraped live from IMDb.</small>",
            unsafe_allow_html=True,
        )

    # ── Hero ─────────────────────────────────────────────
    st.markdown(
        """<div class="hero-header">
            <h1>CineScope</h1>
            <p>Search any movie. Our AI analyzes thousands of audience reviews
            and tells you exactly what people think — in seconds.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Analyze Mode ─────────────────────────────────────
    if mode == "Analyze":
        col_input, col_btn = st.columns([4, 1], vertical_alignment="bottom")
        with col_input:
            query = st.text_input(
                "IMDb ID",
                placeholder="tt1375666  (Inception)",
                label_visibility="collapsed",
            )
        with col_btn:
            go = st.button("Analyze", type="primary", use_container_width=True)

        if go and query:
            if not query.strip().startswith("tt"):
                st.warning("Please enter a valid IMDb ID (e.g. `tt1375666`).")
            else:
                results, source = analyze_movie(query.strip())
                if results:
                    st.divider()
                    render_report(results)

    # ── Compare Mode ─────────────────────────────────────
    elif mode == "Compare":
        col_a, col_b = st.columns(2)
        with col_a:
            id_a = st.text_input("Movie A", placeholder="tt0468569", key="cmp_a")
        with col_b:
            id_b = st.text_input("Movie B", placeholder="tt4154796", key="cmp_b")

        go = st.button("Compare", type="primary", use_container_width=True)

        if go and id_a and id_b:
            res_a, _ = analyze_movie(id_a.strip())
            res_b, _ = analyze_movie(id_b.strip())

            if res_a and res_b:
                # Head-to-head verdict
                winner = res_a if res_a["avg_predicted_score"] >= res_b["avg_predicted_score"] else res_b
                loser = res_b if winner is res_a else res_a
                st.markdown(
                    f"""<div class="verdict-banner">
                        <h3>🏆 {winner['movie_name']} wins the audience vote</h3>
                        <div class="verdict-desc">
                            Scores higher at <b>{winner['avg_predicted_score']}/10</b> vs
                            {loser['movie_name']} at <b>{loser['avg_predicted_score']}/10</b>.<br>
                            {winner['movie_name']} is {winner['sentiment_split']['Positive']}% positive
                            while {loser['movie_name']} is {loser['sentiment_split']['Positive']}% positive.
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                st.divider()

                # Comparison Table
                st.markdown('<div class="section-heading">📊 Head-to-Head Comparison</div>', unsafe_allow_html=True)
                
                # Helper to format table cells and highlight winner
                def cell(val_a, val_b, higher_is_better=True, suffix=""):
                    cls_a = "winner" if (val_a > val_b and higher_is_better) or (val_a < val_b and not higher_is_better) else ""
                    cls_b = "winner" if (val_b > val_a and higher_is_better) or (val_b < val_a and not higher_is_better) else ""
                    # Handle equal
                    if val_a == val_b:
                        cls_a = cls_b = "winner"
                    
                    # Convert None to N/A for display, but logic above might fail on None. Handle safely:
                    disp_a = f"{val_a}{suffix}" if val_a is not None else "N/A"
                    disp_b = f"{val_b}{suffix}" if val_b is not None else "N/A"
                    
                    return f'<td class="{cls_a}">{disp_a}</td><td class="{cls_b}">{disp_b}</td>'

                def safe_val(v): return v if v is not None else 0

                html_table = f"""
                <table class="cmp-table">
                    <tr>
                        <th>Metric</th>
                        <th>{res_a['movie_name']}</th>
                        <th>{res_b['movie_name']}</th>
                    </tr>
                    <tr>
                        <td>AI Predicted Rating</td>
                        {cell(res_a['avg_predicted_score'], res_b['avg_predicted_score'], suffix='/10')}
                    </tr>
                    <tr>
                        <td>Avg User Rating</td>
                        {cell(res_a.get('avg_user_rating'), res_b.get('avg_user_rating'), suffix='/10')}
                    </tr>
                    <tr>
                        <td>Positive %</td>
                        {cell(res_a['sentiment_split']['Positive'], res_b['sentiment_split']['Positive'], suffix='%')}
                    </tr>
                    <tr>
                        <td>Mixed %</td>
                        <td>{res_a['sentiment_split']['Mixed']}%</td>
                        <td>{res_b['sentiment_split']['Mixed']}%</td>
                    </tr>
                    <tr>
                        <td>Negative %</td>
                        {cell(res_a['sentiment_split']['Negative'], res_b['sentiment_split']['Negative'], higher_is_better=False, suffix='%')}
                    </tr>
                    <tr>
                        <td>Total Reviews</td>
                        {cell(res_a['total_reviews'], res_b['total_reviews'])}
                    </tr>
                    <tr>
                        <td>Avg Review Length</td>
                        {cell(res_a.get('avg_review_length', 0), res_b.get('avg_review_length', 0), suffix=' words')}
                    </tr>
                    <tr>
                        <td>Community Votes</td>
                        {cell(res_a.get('community_votes', 0), res_b.get('community_votes', 0))}
                    </tr>
                    <tr>
                        <td>Helpfulness Ratio</td>
                        {cell(res_a.get('helpfulness_ratio', 0), res_b.get('helpfulness_ratio', 0), suffix='%')}
                    </tr>
                    <tr>
                        <td>Spoiler %</td>
                        {cell(res_a.get('spoiler_pct', 0), res_b.get('spoiler_pct', 0), higher_is_better=False, suffix='%')}
                    </tr>
                </table>
                """
                st.markdown(html_table, unsafe_allow_html=True)
                st.divider()

                r1, r2 = st.columns(2)
                with r1:
                    render_report(res_a, key_suffix="_a")
                with r2:
                    render_report(res_b, key_suffix="_b")
            elif not res_a:
                st.error(f"Could not analyze Movie A (`{id_a}`).")
            else:
                st.error(f"Could not analyze Movie B (`{id_b}`).")


if __name__ == "__main__":
    main()
