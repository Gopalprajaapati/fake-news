import time
import pandas as pd
import requests
from googlesearch import search
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


# Function to fetch news article links using Google Search
def google_news_search(query, days_ago, num_results=10):
    start_date = (datetime.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")

    search_query = f"{query} after:{start_date} before:{end_date} site:indiatimes.com OR site:ndtv.com OR site:thehindu.com OR site:hindustantimes.com OR site:bbc.com"

    print(f"🔍 Searching Google for: {search_query}")

    news_links = []
    for url in search(search_query, num_results=num_results, stop=num_results, lang="en"):
        if "google.com" not in url:
            news_links.append(url)

    return news_links


# Function to scrape article content
def scrape_article(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to fetch {url}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("title").text if soup.find("title") else "No Title"
        paragraphs = soup.find_all("p")
        content = " ".join([p.text.strip() for p in paragraphs if len(p.text) > 50])[:2000]

        return {"url": url, "title": title, "content": content}

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None


# User input
query = input("Enter search keywords: ")
days_ago = int(input("Enter number of days back to search: "))

# Search for News Articles
news_links = google_news_search(query, days_ago, num_results=15)

if not news_links:
    print("No articles found.")
else:
    print(f"✅ Found {len(news_links)} articles. Scraping content...\n")

# Scrape Articles
news_data = []
for link in news_links:
    print(f"Scraping: {link}")
    article = scrape_article(link)
    if article:
        news_data.append(article)
    time.sleep(2)  # Delay to avoid blocking

# Save to CSV
df = pd.DataFrame(news_data)
df.to_csv("news_data.csv", index=False)

print("\n✅ Scraping completed! News saved as 'news_data.csv'.")
df.head()
