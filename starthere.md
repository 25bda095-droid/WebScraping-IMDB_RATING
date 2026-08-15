# 🕸️ The Modern Web Scraping Guide: Inside CineScope

Welcome to the **Scraping Architecture Guide**. If you are looking at the CineScope source code and wondering how we manage to scrape hundreds of IMDb reviews in seconds without getting blocked or dealing with complex HTML parsing, this guide is for you.

We are going to break down exactly how the scraper works, why we use the tools we do, and how you can apply these techniques to other modern React-based websites.

---

## 1. The Problem with Traditional Scraping
In the old days of the internet, you could simply use Python's `requests` and `BeautifulSoup` to download a website's HTML, find `<div class="review">`, and extract the text.

**This no longer works on modern websites like IMDb.**
Modern sites use frontend frameworks like React or Next.js (often called SPA — Single Page Applications). When you request the page, the server returns a nearly empty HTML shell and heavily obfuscated JavaScript. The actual data is dynamically loaded via APIs (like GraphQL) after the page loads in the user's browser. 

If you try to use `BeautifulSoup`, you will only scrape empty `<div>` tags. 

---

## 2. The Solution: Playwright + Data Hydration
To scrape modern websites, we need a real browser that can execute JavaScript. That's why we use **[Playwright](https://playwright.dev/)**. Playwright boots up an invisible (headless) Chromium browser, navigates to IMDb, and waits for the JavaScript to execute and populate the page.

But we *still* don't parse the HTML directly. Instead, we use two advanced techniques: **State Hydration** and **GraphQL Interception**.

### Technique A: Intercepting `__NEXT_DATA__`
When a Next.js website (like IMDb) loads, it embeds all the initial data needed for the page inside a hidden `<script>` tag with the ID `__NEXT_DATA__`. This is called "hydration data".

Instead of parsing HTML elements, we just grab this hidden JSON blob. It contains perfectly structured, clean data for the first batch of reviews!

**Here is the exact code from `app.py`:**
```javascript
// We execute this JavaScript INSIDE the headless browser:
const el = document.querySelector("script#__NEXT_DATA__");
const data = JSON.parse(el.textContent);
const reviews = data?.props?.pageProps?.contentData?.data?.title?.reviews;
```
*Why do we do this inside `page.evaluate()`?* Because Python has strict memory limits for string transfers. By parsing the massive JSON string directly inside the browser's JavaScript engine, we avoid memory crashes and return only the clean, extracted review nodes back to Python.

### Technique B: GraphQL Pagination
Once we have the first batch of reviews from `__NEXT_DATA__`, we need to get the rest. When a user scrolls down on IMDb, the site doesn't load a new page. Instead, it fires a hidden API request to a **GraphQL** server to get more reviews.

We mimic this exact API call directly in our code:

```python
# From fetch_reviews_graphql_sync() in app.py
gql_url = (
    f"https://caching.graphql.imdb.com/?operationName=TitleReviewsRefine"
    f"&variables={{'after': '{cursor}', 'const': '{title_id}'}}"
    f"&extensions={{'persistedQuery': {{'sha256Hash': '{PERSISTED_HASH}'}}}}"
)
```
By taking the `cursor` (the ID of the last review we scraped) and passing it to this endpoint, we trick IMDb into giving us the next 50 reviews in clean JSON format instantly. We run this in a fast loop until we hit our target.

---

## 3. Code Breakdown: The Lifecycle of a Scrape

Here is exactly what happens when you type a movie ID into the CineScope UI:

1. **Booting the Browser:** 
   We launch Playwright using a context manager (`with sync_playwright()`). We pass a custom `user_agent` to make our bot look like a normal Google Chrome user on Linux.
2. **Session Persistence (Cookies):**
   If `imdb_session.json` exists, we load it. This file stores our cookies. It speeds up page loads because IMDb doesn't have to re-verify our session or show cookie consent banners.
3. **Wait for Network Idle:**
   We call `page.goto(url)` and explicitly tell Playwright to wait until the `__NEXT_DATA__` script is attached to the DOM.
4. **Scrape & Aggregate:**
   We execute our JSON extraction logic, convert the reviews into a flat list of dictionaries, and close the browser.
5. **Pandas Conversion:**
   The raw dictionaries are loaded into a `pandas.DataFrame`. Pandas allows us to effortlessly drop duplicates, fill missing values, calculate string lengths, and format the data before passing it to the ML model.

---

## 4. Why this approach is robust
- **Speed:** We aren't scrolling and waiting for DOM updates. We query the GraphQL API directly for bulk data.
- **Accuracy:** Scraping JSON guarantees that we don't accidentally miss a review because an HTML class name changed slightly (a common issue with BeautifulSoup).
- **Stealth:** By using Playwright and saving cookies to `imdb_session.json`, we avoid triggering anti-bot mechanisms. 

---

## 📚 Summary of Required Libraries
If you want to build a similar scraper for another site, this is the stack you need:

| Library | Purpose in our Scraper |
| :--- | :--- |
| **`playwright`** | Boots the headless Chromium browser and executes JavaScript contexts. |
| **`pandas`** | Organizes the scraped JSON into a fast, manipulatable table (DataFrame). |
| **`json`** | Standard library used for encoding variables to send to GraphQL. |
| **`urllib.parse`** | Standard library used to properly encode the URL strings for the API calls. |

By combining Playwright's browser automation with direct API interception, you get the best of both worlds: the stealth of a real browser, and the blazing speed of an API integration!