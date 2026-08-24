import os

from utils import make_dir_if_not_exists, extract_text_from_html

DATA_DIR = "./data"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_news_articles",
            "description": "Get today's news articles.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }
]


class Tools:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_news_articles() -> list:
        """Get news articles using the tool."""
        # load all txt files from ./data
        make_dir_if_not_exists(path=DATA_DIR)

        articles = []
        # Get all HTML files in the folder
        html_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".html")]

        for file in html_files:
            file_path = os.path.join(DATA_DIR, file)

            try:
                # Read HTML file
                with open(file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                # Extract clean text from HTML
                extracted_text = extract_text_from_html(html_content)
                article_name = file.replace(".html", "").replace("_", " ").title()
                # Create article dictionary
                article = {
                    "title": article_name,
                    "file_path": file_path,
                    "date": "Today",
                    "content": extracted_text,  # Clean text without HTML/CSS
                }

                articles.append(article)
                print(f"Loaded: {article['title']}")

            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue

        return articles
