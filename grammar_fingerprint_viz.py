import json
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go

#helper funcs: loading + basic counting

def load_change_log(path="models/pos_change_log.json"):
    p = pathlib.Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data


def count_bigrams(changes):
    """
    From STRUCTURAL_CHANGES entries
    "added": [[POS1, POS2], ...]
    "removed": [[POS1, POS2], ...]
    return dicts mapping (POS1, POS2) -> count
    """
    added = defaultdict(int)
    removed = defaultdict(int)

    for entry in changes:
        for a in entry.get("added", []):
            tup = tuple(a)
            added[tup] += 1
        for r in entry.get("removed", []):
            tup = tuple(r)
            removed[tup] += 1

    return dict(added), dict(removed)


def all_pos_from_bigrams(added, removed):
    tags = set()
    for (a, b) in list(added.keys()) + list(removed.keys()):
        tags.add(a)
        tags.add(b)
    return sorted(tags)


def bigram_matrix(counts, pos_list):
    """
    Turn dict[(POS1,POS2)->freq] into |POS|x|POS| matrix
    rows: source POS, cols: target POS
    """
    idx = {p: i for i, p in enumerate(pos_list)}
    mat = np.zeros((len(pos_list), len(pos_list)), dtype=float)
    for (a, b), v in counts.items():
        if a in idx and b in idx:
            mat[idx[a], idx[b]] = v
    return mat

# heatmaps

def plot_heatmaps(added, removed, out_prefix="pos_bigram"):
    pos_list = all_pos_from_bigrams(added, removed)
    if not pos_list:
        print("[WARN] No POS tags found for heatmaps.")
        return

    A = bigram_matrix(added, pos_list)
    G = bigram_matrix(removed, pos_list)

    #avoid log(0) just vis raw counts
    #YOU-style heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    im0 = axes[0].imshow(A, aspect="auto")
    axes[0].set_title("YOU-style POS bigrams (added)")
    axes[0].set_xticks(range(len(pos_list)))
    axes[0].set_xticklabels(pos_list, rotation=90)
    axes[0].set_yticks(range(len(pos_list)))
    axes[0].set_yticklabels(pos_list)

    im1 = axes[1].imshow(G, aspect="auto")
    axes[1].set_title("GPT-style POS bigrams (removed)")
    axes[1].set_xticks(range(len(pos_list)))
    axes[1].set_xticklabels(pos_list, rotation=90)
    axes[1].set_yticks(range(len(pos_list)))
    axes[1].set_yticklabels(pos_list)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}_you_vs_gpt_heatmaps.png", dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out_prefix}_you_vs_gpt_heatmaps.png")

    #delta heatmap: YOU - GPT
    D = A - G
    plt.figure(figsize=(8, 7))
    vmax = np.max(np.abs(D)) if np.any(D) else 1.0
    im = plt.imshow(D, aspect="auto", vmin=-vmax, vmax=vmax, cmap="bwr")
    plt.title("Delta POS bigram frequency (YOU - GPT)")
    plt.xticks(range(len(pos_list)), pos_list, rotation=90)
    plt.yticks(range(len(pos_list)), pos_list)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_delta_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out_prefix}_delta_heatmap.png")


#POS entropy / variability scatterplot

def outgoing_entropy(counts):
    """
    counts: dict[(POS1,POS2)->freq]
    returns: dict[POS1] -> entropy over outgoing transitions
    """
    by_src = defaultdict(lambda: defaultdict(float))
    for (a, b), v in counts.items():
        by_src[a][b] += v

    ent = {}
    for src, dests in by_src.items():
        total = sum(dests.values())
        if total <= 0:
            continue
        probs = np.array(list(dests.values()), dtype=float) / total
        #avoid log(0)
        probs = probs[probs > 0]
        H = -np.sum(probs * np.log2(probs))
        ent[src] = H
    return ent


def plot_entropy_scatter(added, removed, out_path="pos_entropy_scatter.png"):
    ent_you = outgoing_entropy(added)
    ent_gpt = outgoing_entropy(removed)

    #union of POS with >= 1 outgoing edge in either
    tags = sorted(set(ent_you.keys()) | set(ent_gpt.keys()))
    if not tags:
        print("[WARN] No POS entropy data to plot")
        return

    xs = []
    ys = []
    labels = []
    for tag in tags:
        x = ent_you.get(tag, 0.0)
        y = ent_gpt.get(tag, 0.0)
        xs.append(x)
        ys.append(y)
        labels.append(tag)

    plt.figure(figsize=(7, 7))
    plt.scatter(xs, ys, alpha=0.8)
    max_val = max(xs + ys) if xs and ys else 1.0
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Entropy of outgoing POS transitions (YOU)")
    plt.ylabel("Entropy of outgoing POS transitions (GPT)")
    plt.title("POS Entropy / Variability (YOU vs GPT)")

    #annotate biggest diffs
    diffs = [(abs(ent_you.get(t, 0) - ent_gpt.get(t, 0)), i, t)
             for i, t in enumerate(tags)]
    diffs.sort(reverse=True)
    for _, i, t in diffs[:10]:
        plt.annotate(labels[i], (xs[i], ys[i]), fontsize=8,
                     xytext=(5, 5), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out_path}")


#radial grammar fingerprint

def plot_radial_fingerprint(added, removed, out_path="pos_radial_fingerprint.png", top_k=12):
    """
    For each POS tag, get net pref:
    net = outgoing_added - outgoing_removed
    plot top_k by absolute net in radial plot
    """
    pos_all = all_pos_from_bigrams(added, removed)
    if not pos_all:
        print("[WARN] No POS tags for fingerprint")
        return

    out_you = defaultdict(float)
    out_gpt = defaultdict(float)
    for (a, b), v in added.items():
        out_you[a] += v
    for (a, b), v in removed.items():
        out_gpt[a] += v

    net = {}
    for p in pos_all:
        net[p] = out_you.get(p, 0.0) - out_gpt.get(p, 0.0)

    #pick top_k by abs diff
    top = sorted(net.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    labels = [k for k, _ in top]
    values = [net[k] for k, _ in top]
    N = len(labels)

    #angles for each POS 
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    #close polygon for plotting
    angles_closed = np.concatenate([angles, [angles[0]]])
    values_closed = values + [values[0]]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, values_closed, marker="o")
    ax.fill(angles_closed, values_closed, alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title("POS Grammar Fingerprint (net outgoing preference YOU − GPT)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out_path}")

#grammar level Sankey (PCFG rules)

def load_pcfg(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def build_rule_delta_map(pcfg_you, pcfg_gpt):
    """
    For each binary rule (LHS -> RHS...) get delta log-prob
    delta = logP_you - logP_gpt
    missing rules get default low log-prob
    return dict[(LHS, RHS_str)] -> delta
    """
    you_map = {}
    gpt_map = {}

    for lhs, rules in pcfg_you.items():
        for r in rules:
            rhs = tuple(r["rhs"])
            you_map[(lhs, rhs)] = r["log_prob"]

    for lhs, rules in pcfg_gpt.items():
        for r in rules:
            rhs = tuple(r["rhs"])
            gpt_map[(lhs, rhs)] = r["log_prob"]

    all_keys = set(you_map.keys()) | set(gpt_map.keys())
    default_lp = -20.0
    deltas = {}
    for k in all_keys:
        lp_y = you_map.get(k, default_lp)
        lp_g = gpt_map.get(k, default_lp)
        deltas[k] = lp_y - lp_g

    return deltas


def draw_grammar_sankey(pcfg_you_path="models/pcfg_personal_pos.json",
                         pcfg_gpt_path="models/pcfg_gpt_pos.json",
                         top_n=30,
                         out_path="grammar_rule_sankey.html"):
    pcfg_you = load_pcfg(pcfg_you_path)
    pcfg_gpt = load_pcfg(pcfg_gpt_path)
    deltas = build_rule_delta_map(pcfg_you, pcfg_gpt)

    #take top_n rules that are YOU-pref
    sorted_rules = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)
    top_rules = sorted_rules[:top_n]

    #nodes: all LHS and RHS pattern labs
    lhs_nodes = set()
    rhs_nodes = set()
    for (lhs, rhs), d in top_rules:
        lhs_nodes.add(lhs)
        rhs_nodes.add(" ".join(rhs))

    node_labels = list(lhs_nodes) + list(rhs_nodes)
    node_index = {lab: i for i, lab in enumerate(node_labels)}

    sources = []
    targets = []
    values = []

    for (lhs, rhs), d in top_rules:
        s = node_index[lhs]
        t = node_index[" ".join(rhs)]
        sources.append(s)
        targets.append(t)
        #use pos magnitude of delta as “flow” strength
        values.append(max(d, 0.0))

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=node_labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(title_text="Grammar-level Preference (YOU-favored rules)", font_size=10)
    fig.write_html(out_path)
    print(f"[SAVED] Grammar-level Sankey to {out_path}")

#main

def main():
    changes = load_change_log("models/pos_change_log.json")
    added, removed = count_bigrams(changes)

    #heatmap
    plot_heatmaps(added, removed, out_prefix="viz/pos_bigram")

    #entropy scatter
    plot_entropy_scatter(added, removed, out_path="viz/pos_entropy_scatter.png")

    #radial grammar fingerprint
    plot_radial_fingerprint(added, removed, out_path="viz/pos_radial_fingerprint.png")

    #Grammar-level Sankey (PCFG rules)
    draw_grammar_sankey(
        pcfg_you_path="models/pcfg_personal_pos.json",
        pcfg_gpt_path="models/pcfg_gpt_pos.json",
        top_n=30,
        out_path="viz/grammar_rule_sankey.html",
    )


if __name__ == "__main__":
    main()