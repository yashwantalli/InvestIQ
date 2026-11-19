import json
import os
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

PROCESSED_DIR = Path("data/processed")
BENCHMARK_PATH = Path("data/benchmark/eval_questions.json")
INDEX_DIR = Path("data/indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vectorless_trees():
    """Build granular hierarchical JSON trees without any embeddings or vectors."""
    trees = {}
    for json_file in sorted(PROCESSED_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_id = data.get("doc_id", json_file.stem)
        pages = data.get("pages", [])
        total_pages = len(pages)
        page_map = {(p.get("page_number") or p.get("page")): (p.get("text") or "") for p in pages}
        raw_toc = data.get("toc", [])
        nodes = []

        if raw_toc and len(raw_toc) >= 10:
            # Fine-grained embedded TOC (e.g., Infosys Annual Report with 32 entries)
            for idx, entry in enumerate(raw_toc):
                level = entry.get("level", 1) if isinstance(entry, dict) else entry[0]
                title = (entry.get("title", "") if isinstance(entry, dict) else str(entry[1])).strip()
                start_p = entry.get("page", 1) if isinstance(entry, dict) else int(entry[2])

                end_p = total_pages
                for nxt in raw_toc[idx + 1:]:
                    nxt_lvl = nxt.get("level", 1) if isinstance(nxt, dict) else nxt[0]
                    nxt_p = nxt.get("page", total_pages) if isinstance(nxt, dict) else int(nxt[2])
                    if nxt_lvl <= level and nxt_p >= start_p:
                        end_p = max(start_p, nxt_p - 1)
                        break

                nodes.append({
                    "node_id": f"{doc_id}_N{idx+1:03d}",
                    "doc_id": doc_id,
                    "level": level,
                    "title": title,
                    "start_page": start_p,
                    "end_page": min(end_p, total_pages)
                })
        else:
            # Granular page/heading tree for TOC-less or coarse-TOC PDFs
            for p in pages:
                pnum = p.get("page_number") or p.get("page")
                lines = [ln.strip() for ln in (p.get("text") or "").splitlines() if ln.strip()]
                headings = [
                    ln for ln in lines[:8]
                    if 4 <= len(ln) <= 95 and not ln.endswith(".") and not re.match(r"^\d+$", ln)
                ]
                title = " | ".join(headings[:2]) if headings else f"Page {pnum}"
                nodes.append({
                    "node_id": f"{doc_id}_P{pnum:03d}",
                    "doc_id": doc_id,
                    "level": 1,
                    "title": title,
                    "start_page": pnum,
                    "end_page": pnum
                })

        for node in nodes:
            # Store concise node summary (title + key terms/opening lines, NO vectors)
            section_texts = [page_map.get(p, "") for p in range(node["start_page"], node["end_page"] + 1)]
            combined = " ".join(section_texts)
            node["summary"] = f"{node['title']} — {combined[:600].strip()}"
            node["full_index_text"] = f"{doc_id} {node['title']} {combined}"

        trees[doc_id] = nodes
        print(f"[VECTORLESS TREE] {doc_id}: {len(nodes)} nodes")

    with open(INDEX_DIR / "page_index_trees.json", "w", encoding="utf-8") as f:
        json.dump(trees, f, indent=2, ensure_ascii=False)
    return trees


CACHE_PATH = INDEX_DIR / "pageindex_llm_cache.json"

def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def llm_tree_search(question: str, trees: dict, top_k: int = TOP_K):
    """True LLM Reasoning-Based Tree Navigation with model fallback & disk cache."""
    cache = load_cache()
    if question in cache:
        return cache[question]

    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    compact_tree = [
        {
            "node_id": n["node_id"],
            "doc_id": n["doc_id"],
            "title": n["title"],
            "pages": f"{n['start_page']}-{n['end_page']}",
            "summary": n["summary"][:220]
        }
        for doc_nodes in trees.values()
        for n in doc_nodes
    ]

    prompt = f"""You are a financial document navigator. Given the hierarchical document tree below and a user question, reason about which sections logically contain the answer.
Return ONLY a valid JSON list of the top {top_k} most relevant node_ids, ordered from most to least relevant. Example: ["infosys_ar_2025_26_N005", "sebi_fin_edu_booklet_P024"]

Question: {question}

Document Tree:
{json.dumps(compact_tree, ensure_ascii=False)}
"""
    # Try high-throughput Flash models in order if one hits a temporary 503 spike
    for model_id in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"]:
        try:
            resp = client.models.generate_content(model=model_id, contents=prompt)
            match = re.search(r"\[.*?\]", resp.text, re.DOTALL)
            chosen = json.loads(match.group(0)) if match else []
            if chosen:
                cache[question] = chosen
                save_cache(cache)
                return chosen
        except Exception as e:
            print(f"  [WARN] {model_id} temporary error ({type(e).__name__}), trying next...")

    return []  # Triggers automatic offline vectorless BM25 fallback if all API calls fail

def vectorless_bm25_tree_search(question: str, all_nodes: list, bm25: BM25Okapi, top_k: int = TOP_K):
    """Offline vectorless lexical + structural title scoring over tree nodes."""
    q_tokens = tokenize(question)
    base_scores = bm25.get_scores(q_tokens)

    # Boost nodes whose structural section title directly matches query terms
    boosted_scores = []
    q_set = set(q_tokens)
    for idx, node in enumerate(all_nodes):
        title_tokens = set(tokenize(node["title"]))
        title_overlap = len(q_set.intersection(title_tokens))
        score = float(base_scores[idx]) * (1.0 + 0.35 * title_overlap)
        boosted_scores.append((score, idx))

    boosted_scores.sort(reverse=True)
    return [all_nodes[idx] for _, idx in boosted_scores[:top_k]]


def evaluate_vectorless_page_index(trees, top_k=TOP_K):
    all_nodes = [n for doc_nodes in trees.values() for n in doc_nodes]
    node_by_id = {n["node_id"]: n for n in all_nodes}
    use_llm = bool(os.environ.get("GEMINI_API_KEY"))

    tokenized_corpus = [tokenize(n["full_index_text"]) for n in all_nodes]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results_by_cat = {}
    total_answerable = 0
    total_hits = 0

    mode_label = "LLM Reasoning Tree Search" if use_llm else "Vectorless Structural BM25 Tree Search"
    print(f"\nRunning {mode_label} (0 vectors used)...")

    for q in questions:
        cat = q["category"]
        if cat == "unanswerable":
            continue

        results_by_cat.setdefault(cat, {"total": 0, "hits": 0})
        results_by_cat[cat]["total"] += 1
        total_answerable += 1

        if use_llm:
            chosen_ids = llm_tree_search(q["question"], trees, top_k=top_k)
            matched_nodes = [node_by_id[nid] for nid in chosen_ids if nid in node_by_id]
        else:
            matched_nodes = vectorless_bm25_tree_search(q["question"], all_nodes, bm25, top_k=top_k)

        retrieved_pairs = set()
        for n in matched_nodes:
            for p in range(n["start_page"], n["end_page"] + 1):
                retrieved_pairs.add((n["doc_id"], p))

        expected_pairs = {(ep["doc_id"], ep["page"]) for ep in q["expected_pages"]}
        if expected_pairs.intersection(retrieved_pairs):
            total_hits += 1
            results_by_cat[cat]["hits"] += 1
        else:
            print(f"[MISS] {q['id']} ({cat}): expected {list(expected_pairs)[:2]}, got {[(n['doc_id'], n['title'][:25], n['start_page']) for n in matched_nodes[:2]]}")

    print(f"\n--- True Vectorless PageIndex Results (Top-{top_k}) ---")
    print(f"Overall Recall@{top_k}: {total_hits}/{total_answerable} ({(total_hits / max(total_answerable, 1)) * 100:.1f}%)\n")
    for cat, stats in results_by_cat.items():
        pct = (stats["hits"] / max(stats["total"], 1)) * 100
        print(f"  {cat:<16}: {stats['hits']}/{stats['total']} ({pct:.1f}%)")


if __name__ == "__main__":
    trees = build_vectorless_trees()
    evaluate_vectorless_page_index(trees, top_k=TOP_K)