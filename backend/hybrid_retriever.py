import json
import re
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path("data/processed")
BENCHMARK_PATH = Path("data/benchmark/eval_questions.json")
INDEX_DIR = Path("data/indexes")

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
RRF_K = 60  # Standard Reciprocal Rank Fusion constant

# Lightweight entity/document routing hints from metadata
DOC_HINTS = {
    "infosys_ar_2025_26": ["infosys", "annual report", "fy 2025-26", "fy26", "consolidated", "dividend per share"],
    "sebi_nsfe_2020_2025": ["nsfe", "national strategy", "5 cs", "2020-2025", "curriculum", "ncert"],
    "amfi_group_companies": ["amfi", "group companies", "sbi mutual fund", "hdfc asset management", "sponsor", "amc"],
    "sebi_fin_edu_booklet": ["demat", "ipo", "primary market", "secondary market", "scores", "sip"]
}


def detect_target_docs(question: str):
    """Identify if the query explicitly targets one or more specific documents."""
    q_lower = question.lower()
    matched = []
    for doc_id, keywords in DOC_HINTS.items():
        if any(kw in q_lower for kw in keywords):
            matched.append(doc_id)
    return matched


class HybridRetriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.faiss_index = faiss.read_index(str(INDEX_DIR / "faiss_baseline.index"))
        with open(INDEX_DIR / "faiss_chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(INDEX_DIR / "page_index_trees.json", "r", encoding="utf-8") as f:
            trees = json.load(f)

        self.tree_nodes = [node for doc_nodes in trees.values() for node in doc_nodes]
        node_texts = [n["summary_text"] for n in self.tree_nodes]
        self.node_vecs = self.model.encode(node_texts, show_progress_bar=False, normalize_embeddings=True).astype("float32")

        # Load full page text lookup for returning grounded passages
        self.page_lookup = {}
        for json_file in PROCESSED_DIR.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc_id = data.get("doc_id", json_file.stem)
            for p in data.get("pages", []):
                pnum = p.get("page_number") or p.get("page")
                self.page_lookup[(doc_id, pnum)] = p.get("text", "")

    def search(self, question: str, top_k: int = TOP_K):
        q_vec = self.model.encode([question], normalize_embeddings=True).astype("float32")
        target_docs = detect_target_docs(question)

        # 1. Traditional RAG (FAISS) candidate pool (top 15 chunks)
        rag_scores, rag_indices = self.faiss_index.search(q_vec, 15)
        rag_ranked_pages = []
        seen_rag = set()
        for rank, idx in enumerate(rag_indices[0]):
            c = self.chunks[idx]
            key = (c["doc_id"], c["page"])
            if key not in seen_rag:
                seen_rag.add(key)
                rag_ranked_pages.append((key, float(rag_scores[0][rank]), c["text"]))

        # 2. PageIndex Tree Navigation candidate pool (top 8 sections -> pages)
        node_scores = np.dot(self.node_vecs, q_vec[0])
        top_node_idx = np.argsort(node_scores)[::-1][:8]
        tree_ranked_pages = []
        seen_tree = set()
        for idx in top_node_idx:
            node = self.tree_nodes[idx]
            # Take up to first 3 pages of the matched section
            for p in range(node["start_page"], min(node["end_page"] + 1, node["start_page"] + 3)):
                key = (node["doc_id"], p)
                if key not in seen_tree:
                    seen_tree.add(key)
                    tree_ranked_pages.append((key, float(node_scores[idx]), node["title"]))

        # 3. Reciprocal Rank Fusion (RRF) + Document Hint Boost
        fused_scores = {}
        evidence_meta = {}

        for rank, (key, raw_score, snippet) in enumerate(rag_ranked_pages):
            rrf = 1.0 / (RRF_K + rank + 1)
            if target_docs and key[0] in target_docs:
                rrf *= 1.35  # Boost explicitly referenced documents
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf
            evidence_meta[key] = {"source": "FAISS", "raw_score": raw_score, "section": None}

        for rank, (key, raw_score, section_title) in enumerate(tree_ranked_pages):
            rrf = 1.0 / (RRF_K + rank + 1)
            if target_docs and key[0] in target_docs:
                rrf *= 1.35
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf
            if key in evidence_meta:
                evidence_meta[key]["source"] = "HYBRID (FAISS + PageIndex)"
                evidence_meta[key]["section"] = section_title
                evidence_meta[key]["raw_score"] = max(evidence_meta[key]["raw_score"], raw_score)
            else:
                evidence_meta[key] = {"source": "PageIndex", "raw_score": raw_score, "section": section_title}

        # Sort by fused RRF score
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)[:top_k]

        results = []
        for key in sorted_keys:
            doc_id, page = key
            meta = evidence_meta[key]
            results.append({
                "doc_id": doc_id,
                "page": page,
                "rrf_score": round(fused_scores[key], 5),
                "raw_similarity": round(meta["raw_score"], 4),
                "retriever": meta["source"],
                "section": meta["section"],
                "passage": self.page_lookup.get(key, "")[:600].strip()
            })
        return results


def evaluate_hybrid():
    retriever = HybridRetriever()
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results_by_cat = {}
    total_answerable = 0
    total_hits = 0
    unanswerable_max_sim = []

    for q in questions:
        cat = q["category"]
        retrieved = retriever.search(q["question"], top_k=TOP_K)

        if cat == "unanswerable":
            unanswerable_max_sim.append((q["id"], retrieved[0]["raw_similarity"]))
            continue

        results_by_cat.setdefault(cat, {"total": 0, "hits": 0})
        results_by_cat[cat]["total"] += 1
        total_answerable += 1

        expected_pairs = {(ep["doc_id"], ep["page"]) for ep in q["expected_pages"]}
        retrieved_pairs = {(r["doc_id"], r["page"]) for r in retrieved}

        if expected_pairs.intersection(retrieved_pairs):
            total_hits += 1
            results_by_cat[cat]["hits"] += 1
        else:
            print(f"[MISS] {q['id']} ({cat}): expected {list(expected_pairs)[:2]}, got {[(r['doc_id'], r['page'], r['retriever']) for r in retrieved[:3]]}")

    print("\n--- Hybrid Fusion (PageIndex + FAISS RRF) Results (Top-5) ---")
    print(f"Overall Recall@{TOP_K}: {total_hits}/{total_answerable} ({(total_hits / max(total_answerable, 1)) * 100:.1f}%)\n")
    for cat, stats in results_by_cat.items():
        pct = (stats["hits"] / max(stats["total"], 1)) * 100
        print(f"  {cat:<16}: {stats['hits']}/{stats['total']} ({pct:.1f}%)")

    print("\nUnanswerable Top-1 Raw Similarity:")
    for qid, sim in unanswerable_max_sim:
        print(f"  {qid}: max_sim = {sim:.4f}")


if __name__ == "__main__":
    evaluate_hybrid()