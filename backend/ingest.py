import json
from pathlib import Path
import fitz  # PyMuPDF

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_PATH = DATA_DIR / "metadata.json"


def parse_documents():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    for doc_meta in registry["documents"]:
        pdf_path = DATA_DIR / doc_meta["relative_path"]
        if not pdf_path.exists():
            print(f"[MISSING] {pdf_path}")
            continue

        doc = fitz.open(pdf_path)
        toc = doc.get_toc()  # Useful for PageIndex hierarchy construction
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            pages.append({
                "page_number": page_num + 1,
                "char_count": len(text),
                "text": text
            })

        output_payload = {
            **doc_meta,
            "extracted_page_count": len(pages),
            "toc": toc,
            "pages": pages
        }

        out_file = PROCESSED_DIR / f"{doc_meta['doc_id']}.json"
        with open(out_file, "w", encoding="utf-8") as out:
            json.dump(output_payload, out, ensure_ascii=False, indent=2)

        empty_pages = sum(1 for p in pages if p["char_count"] < 50)
        print(
            f"[OK] {doc_meta['doc_id']} | Pages: {len(pages)} | "
            f"TOC entries: {len(toc)} | Low-text pages: {empty_pages}"
        )


if __name__ == "__main__":
    parse_documents()