import streamlit as st
import pandas as pd
import time
import json
import html
import re
import os
import urllib.parse
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

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
        if os.path.exists(SESSION_FILE):
            context = browser.new_context(storage_state=SESSION_FILE)
        else:
            context = browser.new_context()

        page = context.new_page()
        url = f"https://www.imdb.com/title/{movie_id}/reviews"

        try:
            status_container.update(label="Connecting to IMDb...", state="running")

            # Robust retry for page navigation
            for attempt in range(5):
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    try:
                        page.wait_for_selector("script#__NEXT_DATA__", state="attached", timeout=10000)
                    except Exception:
                        pass
                    break
                except Exception as e:
                    if attempt == 4:
                        raise e
                    time.sleep(3)

            # Get movie name from page title
            movie_name = movie_id
            for attempt in range(3):
                try:
                    title_str = page.title()
                    movie_name = title_str.split(" - ")[0].strip() if title_str else movie_id
                    break
                except Exception:
                    if attempt == 2:
                        break
                    time.sleep(3)

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

            seen_ids = {r["review_id"] for r in reviews}

            # GraphQL pagination
            empty_count = 0
            while cursor and len(reviews) < max_reviews:
                status_container.update(
                    label=f"Fetching reviews... {len(reviews)}/{max_reviews}",
                    state="running",
                )
                time.sleep(0.5)
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
# 2. ML MODEL PLACEHOLDER
#    Replace the body of run_model() with your real trained model.
# ═══════════════════════════════════════════════════════════════════

def run_model(reviews_df):
    """
    Placeholder — returns mock stats so the UI has something to render.
    Swap this out with your real model inference.
    """
    if reviews_df is None or reviews_df.empty:
        return None

    movie_name = (
        reviews_df.iloc[0]["movie_name"]
        if "movie_name" in reviews_df.columns
        else "Unknown Movie"
    )

    # Truncate review text for display cards
    pos_body = reviews_df.iloc[0]["review_body"] if "review_body" in reviews_df.columns else ""
    neg_body = reviews_df.iloc[-1]["review_body"] if len(reviews_df) > 1 and "review_body" in reviews_df.columns else ""
    pos_text = (pos_body[:300] + "…") if len(pos_body) > 300 else pos_body
    neg_text = (neg_body[:300] + "…") if len(neg_body) > 300 else neg_body

    return {
        "movie_name": movie_name,
        "total_reviews": len(reviews_df),
        "sentiment_split": {"Positive": 65, "Mixed": 20, "Negative": 15},
        "avg_predicted_score": 7.8,
        "polarization_score": "Medium",
        "top_positive": pos_text or "Great movie!",
        "top_negative": neg_text or "Not my favorite.",
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

/* ── Review quote card ───────────────────────────────── */
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
.review-card .body {
    font-size: .92rem;
    line-height: 1.6;
    color: var(--text-primary);
    font-style: italic;
}

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

/* ── Hide default Streamlit chrome ────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
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


def sentiment_bar(pos: int, mix: int, neg: int):
    return f"""<div class="sentiment-bar-wrapper">
        <div class="bar-title">Sentiment Distribution</div>
        <div class="sentiment-bar">
            <div class="seg-pos" style="width:{pos}%"></div>
            <div class="seg-mix" style="width:{mix}%"></div>
            <div class="seg-neg" style="width:{neg}%"></div>
        </div>
        <div class="sentiment-legend">
            <div><span class="dot" style="background:var(--accent-green)"></span><span>{pos}%</span> Positive</div>
            <div><span class="dot" style="background:var(--accent-amber)"></span><span>{mix}%</span> Mixed</div>
            <div><span class="dot" style="background:var(--accent-red)"></span><span>{neg}%</span> Negative</div>
        </div>
    </div>"""


def review_card(text: str, sentiment: str):
    tag_cls = "positive" if sentiment == "positive" else "negative"
    tag_label = "Most Helpful Positive" if sentiment == "positive" else "Most Helpful Negative"
    escaped = text.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
    return f"""<div class="review-card">
        <div class="tag {tag_cls}">{tag_label}</div>
        <div class="body">"{escaped}"</div>
    </div>"""


def render_report(results, key_suffix=""):
    """Render a single movie report card with custom HTML components."""
    name = results["movie_name"]
    sent = results["sentiment_split"]

    st.markdown(f'<div class="section-heading">📊  {name}</div>', unsafe_allow_html=True)

    # Top metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(stat_card("Predicted Rating", f"{results['avg_predicted_score']}/10", "gold"), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card("Polarization", results["polarization_score"], "amber"), unsafe_allow_html=True)
    with c3:
        st.markdown(stat_card("Reviews Analyzed", str(results["total_reviews"]), "blue"), unsafe_allow_html=True)
    with c4:
        st.markdown(stat_card("Positive Rate", f"{sent['Positive']}%", "green"), unsafe_allow_html=True)

    # Sentiment bar
    st.markdown(sentiment_bar(sent["Positive"], sent["Mixed"], sent["Negative"]), unsafe_allow_html=True)

    # Review highlights
    col_p, col_n = st.columns(2)
    with col_p:
        st.markdown(review_card(results["top_positive"], "positive"), unsafe_allow_html=True)
    with col_n:
        st.markdown(review_card(results["top_negative"], "negative"), unsafe_allow_html=True)

    # Watch link (blank for now)
    st.markdown(f'<div class="watch-btn">📺 &nbsp;Watch {name} — link coming soon</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 5. STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════

def analyze_movie(movie_id: str):
    """Shared logic: check trained DB → fall back to live scrape → run model."""
    trained_movie_ids = ["tt0371746", "tt0800080"]  # Placeholder — load your real list here

    if movie_id in trained_movie_ids:
        # Instant lookup from pre-computed data
        df = pd.DataFrame([{"movie_name": "Trained Movie", "review_body": "Placeholder"}])
        return run_model(df), "trained"
    else:
        with st.status("Live-analyzing from IMDb…", expanded=True) as status:
            df = scrape_live_reviews(movie_id, status, max_reviews=100)
            if df.empty:
                status.update(label="Could not fetch reviews.", state="error")
                return None, "error"
            status.update(label=f"Done — {len(df)} reviews extracted!", state="complete")
        return run_model(df), "scraped"


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
                        <h3>🏆 {winner['movie_name']}</h3>
                        <p>Scores higher at <b>{winner['avg_predicted_score']}/10</b> vs
                        {loser['movie_name']} at <b>{loser['avg_predicted_score']}/10</b>.
                        {winner['movie_name']} is {winner['sentiment_split']['Positive']}% positive
                        while {loser['movie_name']} is {loser['sentiment_split']['Positive']}%.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

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
