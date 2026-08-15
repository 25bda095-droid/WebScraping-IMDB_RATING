That's the overall shape of the app. Now let's break down each piece properly.

## 1. Model I/O — the actual contract

**Input to the model (per movie, batch-computed, not typed by a user):**
- Raw `review_body` text (+ `review_title`)
- Metadata: `franchise`, `format`, `spoiler`, `review_length`, `upvotes`/`downvotes`

**Output (per movie, stored, not recomputed live):**
- `sentiment_split` — % positive / mixed / negative across all its reviews
- `avg_predicted_score` — model's own numeric estimate, separate from the raw `user_rating` average
- `rating_deviation` / `polarization_score` — how much reviewers disagree
- `sentiment_trend_by_year` — array for a line chart
- `top_positive_review` / `top_negative_review` — highest-upvoted real text on each side
- `helpfulness_prediction` — used only if you build that secondary model
- `mismatch_count` — reviews where sentiment ≠ rating (from the mismatch detector idea)

The user never sends text into the model live for the main flow — they just pick a movie, and you serve pre-computed JSON.

## 2. Page-by-page breakdown

**Landing page** — don't drop the user on an empty search bar. Show 3-4 curated rows: "Most positively reviewed," "Most polarizing," "Trending searches." Gives people something to click into immediately.

**Search bar** — fuzzy-match autocomplete against your ~1,100 movie list as they type. If nothing matches, show "not in our trained set — we'll fetch live reviews now" and trigger your scraper as a fallback (with a small loading state, since scraping takes a few seconds).

**Movie report page** (the core screen, shown in the diagram):
- Header: poster, title, franchise/format badge, IMDb link
- Sentiment donut + avg rating vs. franchise average
- Rating/sentiment trend line by year
- Two review cards: top upvoted positive, top upvoted negative
- **Watch link button** — a "Watch now" button linking out. Since this is your own curated link, keep it simple: a per-movie or per-franchise URL you set yourself (or a JustWatch-style "where to watch" search link built from the movie title, which works for any movie without you manually mapping 1,100 links one by one)

**Compare page:**
- Two movie search boxes side by side
- Renders both report cards next to each other
- Add one extra thing a single view can't show: a **head-to-head verdict line** — "Movie A is more positively reviewed (78% vs 61%) but Movie B is less polarizing" — this is the payoff of comparison mode, not just two cards stacked

**Optional "test your own review" page** — demoted to a small secondary feature, not the homepage, per what we discussed earlier.

## 3. Watch link — practical way to do it without manual work

Rather than storing 1,100 individual links yourself, generate a dynamic search URL per movie, e.g. a JustWatch search link built from the movie title (`https://www.justwatch.com/us/search?q=<movie_name>`). One line of code, works for any movie including scraped ones outside your original 1,100, no manual link-curation needed. If you want it to feel more "curated," you can hardcode a handful of your own picks for the flagship movies you demo most, and fall back to the dynamic search link for everything else.

## 4. What makes this "usable by a normal user," concretely

- Zero typing/pasting required for the core flow — search and click only
- Fast — because everything for your 1,100 movies is pre-computed, not live-inferred
- Graceful fallback for unknown movies (scrape once, cache forever) instead of failing
- Comparison mode gives a reason to explore more than one movie per visit
- Watch link closes the loop — the user came wanting to decide whether to watch something, and you end their journey with exactly that action

Want me to scaffold the actual Streamlit `app.py` with these pages (Home / Search / Report / Compare) and placeholder functions for the pre-computed lookup + scrape-fallback + watch-link generator, so you have working skeleton code to fill in with your real model?