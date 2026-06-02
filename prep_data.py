import json
import re
from bs4 import BeautifulSoup
from pathlib import Path


def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove tags which carry no doc content    
    for tag in soup(["script", "style", "meta", "noscript", "head"]):
        tag.decompose()

    # Remove inline XBRL tags 
    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
        tag.unwrap()

    text = soup.get_text(separator="\n", strip=True)

    # Normalise unicode whitespace and spacing
    text = re.sub(r"[\u200b-\u200f\ufeff\xa0]", " ", text)  
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def clean_and_prepare():
    with open("data/QA_pairs.json", "r") as f:
        qa_pairs = json.load(f)

    doc_cache: dict[str, str] = {}
    unified_input = []

    for item in qa_pairs:
        doc_name = item["document"]

        if doc_name not in doc_cache:
            html_path = Path("data/documents") / doc_name
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            cleaned = clean_html(raw)
            doc_cache[doc_name] = cleaned

            wc = len(cleaned.split())
            print(f"  {doc_name}: {wc:,} words")

        unified_input.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "document": doc_cache[doc_name], # embed full doc per Q
        })

    with open("input.json", "w", encoding="utf-8") as f:
        json.dump(unified_input, f, indent=2, ensure_ascii=False)
        
if __name__ == "__main__":
    clean_and_prepare()