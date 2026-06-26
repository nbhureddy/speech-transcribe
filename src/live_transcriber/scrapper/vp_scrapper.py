import json
import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://forum.valuepickr.com"
SEARCH_URL = f"{BASE_URL}/search.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

OUTPUT_DIR = Path("valuepickr_downloads")
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text("\n", strip=True)

# ------------------------------------------------------
# Search topics
# ------------------------------------------------------
def search_topics(company_name):
    params = {
        "q": company_name
    }

    response = requests.get(
        SEARCH_URL,
        params=params,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    topics = data.get("topics", [])

    unique_topics = []
    seen = set()

    for topic in topics:
        topic_id = topic.get("id")

        if topic_id in seen:
            continue

        seen.add(topic_id)

        unique_topics.append({
            "id": topic_id,
            "title": topic.get("title"),
            "slug": topic.get("slug"),
            "posts_count": topic.get("posts_count"),
            "views": topic.get("views"),
            "last_posted_at": topic.get("last_posted_at")
        })

    return unique_topics


# ------------------------------------------------------
# Display topics
# ------------------------------------------------------
def display_topics(topics):
    print("\nMatching Topics:\n")

    for idx, topic in enumerate(topics, start=1):
        print(f"[{idx}] {topic['title']}")
        print(f"     Topic ID : {topic['id']}")
        print(f"     Posts    : {topic['posts_count']}")
        print(f"     Views    : {topic['views']}")
        print(f"     Activity : {topic['last_posted_at']}")
        print()


# ------------------------------------------------------
# Download topic JSON
# ------------------------------------------------------
def download_topic(topic):
    topic_url = (
        f"{BASE_URL}/t/{topic['slug']}/{topic['id']}.json"
    )

    response = requests.get(
        topic_url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    return response.json()

# ------------------------------------------------------
# Save topic
# ------------------------------------------------------
def save_topic(topic, topic_json):
    safe_title = re.sub(r'[^a-zA-Z0-9_-]+', '_', topic['title'])

    topic_dir = OUTPUT_DIR / safe_title
    topic_dir.mkdir(exist_ok=True)

    # Save raw JSON
    json_path = topic_dir / "thread.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(topic_json, f, indent=2, ensure_ascii=False)

    # Save cleaned TXT
    txt_path = topic_dir / "thread.txt"

    posts = topic_json.get("post_stream", {}).get("posts", [])

    with open(txt_path, "w", encoding="utf-8") as f:
        for post in posts:
            author = post.get("username")
            created_at = post.get("created_at")
            cooked = post.get("cooked", "")

            text = clean_html(cooked)

            f.write("=" * 80 + "\n")
            f.write(f"Author    : {author}\n")
            f.write(f"CreatedAt : {created_at}\n")
            f.write("-" * 80 + "\n")
            f.write(text + "\n\n")

    # Optional markdown
    md_path = topic_dir / "thread.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {topic['title']}\n\n")

        for post in posts:
            author = post.get("username")
            created_at = post.get("created_at")
            cooked = post.get("cooked", "")

            text = clean_html(cooked)

            f.write(f"## {author} ({created_at})\n\n")
            f.write(text)
            f.write("\n\n")

    print(f"Saved: {topic['title']}")
    print(f"Folder: {topic_dir}\n")


# ------------------------------------------------------
# Main flow
# ------------------------------------------------------
def main():
    company_name = input("Enter company name: ").strip()

    topics = search_topics(company_name)

    if not topics:
        print("No matching topics found.")
        return

    display_topics(topics)

    selection = input(
        "Select topics (comma separated indexes, e.g. 1,2,5): "
    )

    indexes = []

    for part in selection.split(","):
        part = part.strip()

        if not part:
            continue

        indexes.append(int(part) - 1)

    selected_topics = [topics[i] for i in indexes]

    print("\nDownloading selected topics...\n")

    for topic in selected_topics:
        try:
            topic_json = download_topic(topic)
            save_topic(topic, topic_json)
        except Exception as e:
            print(f"Failed: {topic['title']}")
            print(str(e))
            print()

    print("Done.")


if __name__ == "__main__":
    main()