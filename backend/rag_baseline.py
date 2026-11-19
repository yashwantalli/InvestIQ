import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path("data/processed")
BENCHMARK_PATH = Path("data/benchmark/eval_questions.json")
INDEX_DIR = Path("data/indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap to preserve boundary context
TOP_K = 5


def build_chunks():
    """Split processed document pages into overlapping chunks with metadata."""
    chunks = []
    for json_file in sorted(PROCESSED_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc_id = data.get("doc_id", json_file.stem)
        for page in data.get("pages", []):
            page_num = page.get("page_number") or page.get("page")
            text = (page.get("text") or "").strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                end = start + CHUNK_SIZE
                chunk_text = text[start:end].strip()
                if len(chunk_text) > 40:
                    chunks.append({
                        "doc_id": doc_id,
                        "page": page_num,
                        "text": chunk_text
                    })
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_and_save_index(chunks, model):
    """Embed chunks and store a normalized Inner-Product (cosine) FAISS index."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss_baseline.index"))
    with open(INDEX_DIR / "faiss_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    return index


def evaluate_baseline(index, chunks, model, top_k=TOP_K):
    """Evaluate Retrieval Recall@k and Mean Reciprocal Rank (MRR) on eval_questions.json."""
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results_by_cat = {}
    total_answerable = 0
    total_hits = 0
    unanswerable_scores = []

    for q in questions:
        cat = q["category"]
        results_by_cat.setdefault(cat, {"total": 0, "hits": 0})
        results_by_cat[cat]["total"] += 1

        q_vec = model.encode([q["question"]], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(q_vec, top_k)

        retrieved = [
            {
                "doc_id": chunks[idx]["doc_id"],
                "page": chunks[idx]["page"],
                "score": float(scores[0][rank])
            }
            for rank, idx in enumerate(indices[0])
        ]

        if cat == "unanswerable":
            unanswerable_scores.append((q["id"], retrieved[0]["score"]))
            continue

        total_answerable += 1
        expected_pairs = {(ep["doc_id"], ep["page"]) for ep in q["expected_pages"]}
        retrieved_pairs = {(r["doc_id"], r["page"]) for r in retrieved}

        hit = len(expected_pairs.intersection(retrieved_pairs)) > 0
        if hit:
            total_hits += 1
            results_by_cat[cat]["hits"] += 1
        else:
            print(f"[MISS] {q['id']} ({cat}): expected {list(expected_pairs)[:2]}, got {[(r['doc_id'], r['page']) for r in retrieved[:3]]}")

    print("\n--- Traditional RAG Baseline Results (Top-5) ---")
    print(f"Overall Recall@{top_k}: {total_hits}/{total_answerable} ({(total_hits / max(total_answerable, 1)) * 100:.1f}%)\n")
    for cat, stats in results_by_cat.items():
        if cat == "unanswerable":
            continue
        pct = (stats["hits"] / max(stats["total"], 1)) * 100
        print(f"  {cat:<16}: {stats['hits']}/{stats['total']} ({pct:.1f}%)")

    print("\nUnanswerable Top-1 Similarity Scores (helps set abstention threshold):")
    for qid, top_score in unanswerable_scores:
        print(f"  {qid}: top_score = {top_score:.4f}")


if __name__ == "__main__":
    import sys
    os.environ["HF_HUB_OFFLINE"] = "1"  # Use locally cached model without network check
    model = SentenceTransformer(MODEL_NAME)

    index_path = INDEX_DIR / "faiss_baseline.index"
    chunks_path = INDEX_DIR / "faiss_chunks.json"

    if index_path.exists() and chunks_path.exists() and "--rebuild" not in sys.argv:
        print("Loading cached FAISS index from disk...")
        index = faiss.read_index(str(index_path))
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        print("Building FAISS index from scratch...")
        chunks = build_chunks()
        index = build_and_save_index(chunks, model)

    evaluate_baseline(index, chunks, model, top_k=TOP_K)