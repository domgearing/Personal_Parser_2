import json
import pathlib
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = pathlib.Path("models/pos_change_log.json")
OUT_DIR = pathlib.Path("models/pos_change_plots")


def load_changes(path: pathlib.Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Change log not found - {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def count_bigrams(changes: List[Dict]) -> Tuple[Counter, Counter]:
    """
    count added/rem POS bigrams across all sentences.

    bigrams stored as lists: ["DT", "NN"],
    convert back to tuples ("DT", "NN") to use as keys in Counter
    """
    added_counter = Counter()
    removed_counter = Counter()

    for rec in changes:
        for a in rec.get("added", []):
            if isinstance(a, list):
                a = tuple(a)
            added_counter[a] += 1

        for r in rec.get("removed", []):
            if isinstance(r, list):
                r = tuple(r)
            removed_counter[r] += 1

    return added_counter, removed_counter


def count_unigram_pos(changes: List[Dict]) -> Tuple[Counter, Counter]:
    """
    count how often a POS tag was added or rem.,
    based on bigram changes.
    """
    added_pos = Counter()
    removed_pos = Counter()

    for rec in changes:
        for a in rec.get("added", []):
            if isinstance(a, list):
                a = tuple(a)
            if len(a) == 2:
                added_pos[a[0]] += 1
                added_pos[a[1]] += 1

        for r in rec.get("removed", []):
            if isinstance(r, list):
                r = tuple(r)
            if len(r) == 2:
                removed_pos[r[0]] += 1
                removed_pos[r[1]] += 1

    return added_pos, removed_pos


def plot_top(counter: Counter, title: str, out_path: pathlib.Path, top_k: int = 15):
    """
    bar plot of top-k items in counter
    """
    if not counter:
        print(f"[WARN] No data for {title}")
        return

    items = counter.most_common(top_k)
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


# HEATMAPS OF POS TRANSITIONS 

def build_transition_matrix(counter: Counter, top_tags: List[str]) -> np.ndarray:
    """
    |top_tags| x |top_tags| matrix of bigram counts
    rows = preceding POS, cols = following POS
    """
    idx = {tag: i for i, tag in enumerate(top_tags)}
    mat = np.zeros((len(top_tags), len(top_tags)), dtype=float)

    for (p1, p2), c in counter.items():
        if p1 in idx and p2 in idx:
            mat[idx[p1], idx[p2]] += c

    return mat


def plot_heatmap(mat: np.ndarray, labels: List[str], title: str, out_path: pathlib.Path):
    if mat.size == 0:
        print(f"[WARN] Empty matrix for heatmap: {title}")
        return

    plt.figure(figsize=(10, 8))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Count")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved heatmap: {out_path}")


#POS TRANSITION DIAGRAMS 

def plot_pos_transition_diagram(counter: Counter, title: str, out_path: pathlib.Path, min_count: int = 3):
    """
    plot POS transition diagram 
    nodes are POS tags, edges weighted by bigram counts >= min_count
    """
    try:
        import networkx as nx
    except ImportError:
        print(f"[WARN] skipping transition diagram: {title}")
        return

    #build graph
    G = nx.DiGraph()
    for (p1, p2), c in counter.items():
        if c >= min_count:
            G.add_edge(p1, p2, weight=c)

    if not G.edges:
        print(f"[WARN] No edges above threshold {min_count} for {title}")
        return

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.8, iterations=100)

    #edge widths scaled by counts
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights)
    widths = [1 + 4 * (w / max_w) for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightgray")
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowstyle="->", arrowsize=12)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved transition diagram: {out_path}")


#SANKEY FLOWS (GPT -> PERSONAL)

def build_bigram_flow(changes: List[Dict], max_flows: int = 50):
    """
    build flow table between removed bigrams (GPT-ish patterns)
    added bigrams (personal-ish patterns).

    For each sentence-level record:
    for each rem bigram r
    for each added bigram a
    increment flow (r -> a)

    return: sources: List[str], targets: List[str], values : List[int]
    """
    flow_counter = Counter()

    for rec in changes:
        removed = rec.get("removed", [])
        added = rec.get("added", [])
        #convert to tuples
        removed = [tuple(r) if isinstance(r, list) else r for r in removed]
        added = [tuple(a) if isinstance(a, list) else a for a in added]

        for r in removed:
            for a in added:
                src = f"{r[0]}→{r[1]}"
                tgt = f"{a[0]}→{a[1]}"
                flow_counter[(src, tgt)] += 1

    if not flow_counter:
        return [], [], []

    top_flows = flow_counter.most_common(max_flows)
    sources = [s for (s, _), _ in top_flows]
    targets = [t for (_, t), _ in top_flows]
    values = [v for _, v in top_flows]

    return sources, targets, values


def plot_sankey_flows(sources: List[str], targets: List[str], values: List[int], out_html: pathlib.Path, out_csv: pathlib.Path):
    """
    plot sankey diagram with plotly
    save flows to CSV
    """
    if not sources:
        print("[WARN] No flows to visualize for diagram.")
        return

    #save CSV
    import csv
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_bigram", "target_bigram", "count"])
        for s, t, v in zip(sources, targets, values):
            writer.writerow([s, t, v])
    print(f"[INFO] Saved POS bigram flow table (for Sankey) to {out_csv}")

    #try plotly
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly not installed; skipping Sankey HTML plot.")
        return

    #make node index
    labels = sorted(set(sources) | set(targets))
    idx = {lab: i for i, lab in enumerate(labels)}

    src_idx = [idx[s] for s in sources]
    tgt_idx = [idx[t] for t in targets]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=labels,
                    pad=15,
                    thickness=20,
                ),
                link=dict(
                    source=src_idx,
                    target=tgt_idx,
                    value=values,
                ),
            )
        ]
    )

    fig.update_layout(title_text="GPT → YOU POS Bigram Flow (Rewrites)", font_size=10)
    fig.write_html(str(out_html))
    print(f"[INFO] Saved Sankey diagram HTML to {out_html}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    changes = load_changes(LOG_PATH)
    print(f"[INFO] Loaded {len(changes)} sentence-level change records from {LOG_PATH}")

    #bigram-lvl changes
    added_bigrams, removed_bigrams = count_bigrams(changes)

    print("\n=== TOP ADDED POS BIGRAMS (structural patterns personal used more) ===")
    for (b, cnt) in added_bigrams.most_common(20):
        print(f"  {b[0]} → {b[1]} : {cnt}")

    print("\n=== TOP REMOVED POS BIGRAMS (patterns GPT used more) ===")
    for (b, cnt) in removed_bigrams.most_common(20):
        print(f"  {b[0]} → {b[1]} : {cnt}")

    #unigram POS counts (POS tags that are added/rem most)
    added_pos, removed_pos = count_unigram_pos(changes)

    print("\n=== TOP ADDED POS TAGS ===")
    for pos, cnt in added_pos.most_common(20):
        print(f"  {pos}: {cnt}")

    print("\n=== TOP REMOVED POS TAGS ===")
    for pos, cnt in removed_pos.most_common(20):
        print(f"  {pos}: {cnt}")

    #barplots
    plot_top(
        added_bigrams,
        "Top Added POS Bigrams (Rewritten vs Original)",
        OUT_DIR / "added_bigrams.png",
    )
    plot_top(
        removed_bigrams,
        "Top Removed POS Bigrams (Rewritten vs Original)",
        OUT_DIR / "removed_bigrams.png",
    )
    plot_top(
        added_pos,
        "Top Added POS Tags",
        OUT_DIR / "added_pos.png",
    )
    plot_top(
        removed_pos,
        "Top Removed POS Tags",
        OUT_DIR / "removed_pos.png",
    )

    #heatmaps for added/rem transitions (use top POS tags overall)
    #use comb. POS freq. to pick top tags
    total_pos = added_pos + removed_pos
    top_pos_labels = [pos for pos, _ in total_pos.most_common(20)]
    if top_pos_labels:
        mat_added = build_transition_matrix(added_bigrams, top_pos_labels)
        mat_removed = build_transition_matrix(removed_bigrams, top_pos_labels)

        plot_heatmap(
            mat_added,
            top_pos_labels,
            "Added POS Bigram Transition Heatmap",
            OUT_DIR / "added_bigrams_heatmap.png",
        )
        plot_heatmap(
            mat_removed,
            top_pos_labels,
            "Removed POS Bigram Transition Heatmap",
            OUT_DIR / "removed_bigrams_heatmap.png",
        )
    else:
        print("[WARN] No POS tags found for heatmaps.")

    #POS transition diagrams
    plot_pos_transition_diagram(
        added_bigrams,
        "POS Transition Diagram – Added Bigrams (YOU-style)",
        OUT_DIR / "added_bigrams_graph.png",
        min_count=3,
    )
    plot_pos_transition_diagram(
        removed_bigrams,
        "POS Transition Diagram – Removed Bigrams (GPT-style)",
        OUT_DIR / "removed_bigrams_graph.png",
        min_count=3,
    )

    #sankey flow between GPT-ish and PERSONAL-ish POS bigramss
    sources, targets, values = build_bigram_flow(changes, max_flows=40)
    plot_sankey_flows(
        sources,
        targets,
        values,
        OUT_DIR / "pos_bigram_sankey.html",
        OUT_DIR / "pos_bigram_flows.csv",
    )


if __name__ == "__main__":
    main()