import json
from pathlib import Path

BENCHMARK_PATH = Path("data/benchmark/eval_questions.json")
PROCESSED_DIR = Path("data/processed")

KEYWORD_FIXES = {
    "Q03": ["SEBI"],
    "Q10": ["school"],
    "Q15": ["grievance"],
    "Q16": ["group"]
}

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

docs = {}
for json_file in PROCESSED_DIR.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        doc_id = data.get("doc_id", json_file.stem)
        docs[doc_id] = data.get("pages", [])

for q in questions:
    if q["category"] == "unanswerable":
        continue
    if q["id"] in KEYWORD_FIXES:
        q["search_keywords"] = KEYWORD_FIXES[q["id"]]

    matched_pages = []
    for doc_id in q["expected_docs"]:
        pages = docs.get(doc_id, [])
        for page in pages:
            page_num = page.get("page_number") or page.get("page")
            text = page.get("text", "").lower()
            if all(kw.lower() in text for kw in q["search_keywords"]):
                matched_pages.append({"doc_id": doc_id, "page": page_num})
    q["expected_pages"] = matched_pages[:5]
    print(f"{q['id']} ({q['category']}): {len(q['expected_pages'])} pages -> {q['expected_pages']}")

with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)