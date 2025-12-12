import argparse
import math
from typing import Dict, List, Tuple, Set, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from nltk.tokenize import word_tokenize
from nltk.tree import Tree

from cky_viterbi import PCFG as LexPCFG, cky_viterbi
from cky_POS import PCFG as PosPCFG, cky_viterbi_with_rules
import nltk

LOG_ZERO = -1e9


def build_cky_chart(tokens: List[str], grammar: Any) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    run CKY and return full chart:
    chart[(i, j)] = { A: best_logprob_for_A_covering_tokens[i:j] }
    """

    n = len(tokens)
    chart: Dict[Tuple[int, int], Dict[str, float]] = {}

    #init diagl with lexical rules
    for i, w in enumerate(tokens):
        cell: Dict[str, float] = {}

        if w in grammar.lexical_rules:
            for A, logp in grammar.lexical_rules[w]:
                cell[A] = max(cell.get(A, LOG_ZERO), logp)

        #step back to UNK if needed
        if not cell and "UNK" in grammar.lexical_rules:
            for A, logp in grammar.lexical_rules["UNK"]:
                cell[A] = max(cell.get(A, LOG_ZERO), logp)

        if not cell:
            print(f"[DEBUG] No lexical entries for token {i}: {w!r}")

        chart[(i, i + 1)] = cell

    # programming for longer spans
    for span in range(2, n + 1):      # span length
        for i in range(0, n - span + 1):
            j = i + span
            best_for_span: Dict[str, float] = {}

            for k in range(i + 1, j):
                left_cell = chart.get((i, k), {})
                right_cell = chart.get((k, j), {})
                if not left_cell or not right_cell:
                    continue

                for B, logpB in left_cell.items():
                    for C, logpC in right_cell.items():
                        key = (B, C)
                        if key not in grammar.binary_rules:
                            continue
                        for A, logp_rule in grammar.binary_rules[key]:
                            cand = logp_rule + logpB + logpC
                            if cand > best_for_span.get(A, LOG_ZERO):
                                best_for_span[A] = cand

            chart[(i, j)] = best_for_span

    return chart


def prepare_chart_matrix(tokens: List[str],
                         chart: Dict[Tuple[int, int], Dict[str, float]]) -> np.ndarray:
    """
    Build an n x n matrix M where:
      M[i, j] = # nonterminals that span tokens[i:j+1]
    for j >= i; lower triangle and diagl are NaN
    """
    n = len(tokens)
    M = np.full((n, n), np.nan)

    for (i, j), cell in chart.items():
        if j > i:
            col = j - 1  #j is end index, align to column j-1
            M[i, col] = len(cell)  #size of cell = how many NT's

    return M


def best_nt_label(cell: Dict[str, float]) -> str:
    """
    return single highest-prob NT in this cell
    """
    if not cell:
        return ""
    #sort by logprob desc
    best_nt = max(cell.items(), key=lambda kv: kv[1])[0]
    return best_nt


def collect_spans_from_tree(tree: Tree) -> Set[Tuple[int, int]]:
    """
    input: NLTK Tree
    collect all (start, end) spans in leaf indices for all constituents in tree
    returns: set of (i, j), j is exclusive
    """

    spans: Set[Tuple[int, int]] = set()

    def helper(t: Tree, start_idx: int) -> int:
        """
        recursively get spans
        returns end index after subtree.
        """
        if isinstance(t[0], str):
            #preterminal: one word
            end_idx = start_idx + 1
            spans.add((start_idx, end_idx))
            return end_idx

        idx = start_idx
        for child in t:
            idx = helper(child, idx)
        #span for this subtree
        spans.add((start_idx, idx))
        return idx

    helper(tree, 0)
    return spans


def visualize_cky_chart_png(tokens: List[str],
                            chart: Dict[Tuple[int, int], Dict[str, float]],
                            parse_tree: Tree,
                            out_path: str = "cky_chart.png",
                            max_span_for_labels: int = 4,
                            figsize_scale: float = 0.6) -> None:
    """
    make triangular CKY chart visualization, save as PNG.

    - Heatmap color is # of NT's in each cell
    - For spans with length <= max_span_for_labels, use only highest-probability NT label
    - use red rectangles for all spans used in Viterbi parse
    """

    n = len(tokens)
    M = prepare_chart_matrix(tokens, chart)

    #get parse path spans
    parse_spans = collect_spans_from_tree(parse_tree) if parse_tree is not None else set()

    #figure size scales with sentence length - capped reasonably
    fig_w = max(6, n * figsize_scale)
    fig_h = max(6, n * figsize_scale * 0.7)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    #use masked array - hide NaNs (lower triangle)
    masked = np.ma.masked_invalid(M)
    cmap = plt.cm.Blues
    cmap.set_bad(color="white")

    im = ax.imshow(masked, interpolation='nearest', cmap=cmap, origin='upper')

    #colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("# nonterminals in cell", rotation=90)

    #tick labs: indices along axes
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], rotation=90, fontsize=8)
    ax.set_yticklabels([str(i) for i in range(n)], fontsize=8)

    ax.set_xlabel("End index (j-1)")
    ax.set_ylabel("Start index (i)")
    ax.set_title("CKY Chart with Viterbi Parse Path")

    #grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False)

    #annotate cells with single NT label on short spans
    for (i, j), cell in chart.items():
        if j <= i:
            continue
        span_len = j - i
        if span_len > max_span_for_labels:
            continue  #don't label long spans -avoids clutter

        val = M[i, j - 1]
        if math.isnan(val) or val == 0:
            continue

        label = best_nt_label(cell)
        if not label:
            continue

        #place text at (col=j-1, row=i)
        ax.text(
            j - 1,
            i,
            label,
            ha="center",
            va="center",
            fontsize=6,
            color="black"
        )

    # emphasize Viterbi parse spans - red rectangles
    for (i, j) in parse_spans:
        if j <= i:
            continue
        col = j - 1
        row = i
        #draw rectangle around cell
        rect = Rectangle(
            (col - 0.5, row - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="red",
            linewidth=1.5
        )
        ax.add_patch(rect)

    #add tokens along bottom as sep axis
    fig.subplots_adjust(bottom=0.22)
    token_ax = fig.add_axes([0.1, 0.05, 0.8, 0.1])
    token_ax.set_axis_off()
    token_ax.set_xlim(0, n)
    token_ax.set_ylim(0, 1)

    for idx, tok in enumerate(tokens):
        token_ax.text(
            idx + 0.0,
            0.5,
            tok,
            ha="center",
            va="center",
            fontsize=8,
            rotation=45
        )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] CKY chart visualization saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Vis CKY chart as PNG with Viterbi path"
    )
    parser.add_argument(
        "--grammar",
        type=str,
        required=True,
        help="Path to PCFG JSON file."
    )
    parser.add_argument(
        "--sent",
        type=str,
        required=True,
        help="sent to parse and visualize"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="cky_chart.png",
        help="Output PNG filename."
    )
    parser.add_argument(
        "--max-span-labels",
        type=int,
        default=4,
        help="Max span length to draw NT labels in cells"
    )
    parser.add_argument(
        "--pos-grammar",
        action="store_true",
        help="parse over POS tags instead of words",
    )
    args = parser.parse_args()

    # load grammar
    if args.pos_grammar:
        grammar = PosPCFG(start_symbol="S", grammar_path=args.grammar)
    else:
        grammar = LexPCFG(start_symbol="S", grammar_path=args.grammar)

    # tok or POS-tag depending on grammar type
    if args.pos_grammar:
        words = word_tokenize(args.sent)
        tokens = [tag for _, tag in nltk.pos_tag(words)]
        print("Sentence:", args.sent)
        print("POS TOKENS:", tokens)
    else:
        tokens = word_tokenize(args.sent)
        print("Sentence:", args.sent)
        print("TOKENS  :", tokens)

    #build CKY chart
    chart = build_cky_chart(tokens, grammar)

    #get Viterbi parse tree for path overlay
    if args.pos_grammar:
        logp, tree, _ = cky_viterbi_with_rules(tokens, grammar)
    else:
        logp, tree = cky_viterbi(tokens, grammar)
    if tree is None:
        print("[WARN] No parse found for this sentence under the grammar.")
    else:
        print("Viterbi log-prob:", logp)
        # tree.pretty_print() 

    #viz - save PNG
    visualize_cky_chart_png(
        tokens,
        chart,
        parse_tree=tree,
        out_path=args.out,
        max_span_for_labels=args.max_span_labels,
    )


if __name__ == "__main__":
    main()
