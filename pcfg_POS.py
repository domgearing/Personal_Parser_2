import argparse
import json
import math
import pathlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

import nltk
from nltk import Tree, Nonterminal, Production


def strip_functional(tag: str) -> str:
    return tag.split("-")[0].split("=")[0]


def base_label(sym: str) -> str:
    s = sym
    if "|" in s:
        s = s.split("|")[0]
    if "-" in s:
        s = s.split("-")[0]
    if "=" in s:
        s = s.split("=")[0]
    return s


def load_trees(jsonl_path: str) -> List[Tree]:
    trees: List[Tree] = []
    path = pathlib.Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tree_str = rec.get("tree")
            if not tree_str:
                continue
            try:
                t = Tree.fromstring(tree_str)
            except Exception as e:
                print(f"[WARN] Could not parse tree string: {e}")
                continue
            trees.append(t)
    print(f"Loaded {len(trees)} POS trees from {jsonl_path}")
    return trees


def tree_to_cnf_productions(t: Tree) -> List[Production]:
    t = t.copy(deep=True)

    #normalize NT labs (S-SBJ -> S)
    for subtree in t.subtrees():
        if isinstance(subtree.label(), str):
            subtree.set_label(strip_functional(subtree.label()))

    #keep unary preterminals, collapse other unary rules
    t.collapse_unary(collapsePOS=False, collapseRoot=False)

    # convert to CNF
    t.chomsky_normal_form(factor="left", horzMarkov=1)

    return list(t.productions())


def collect_productions(trees: List[Tree]) -> List[Production]:
    all_prods: List[Production] = []
    for i, t in enumerate(trees, start=1):
        prods = tree_to_cnf_productions(t)
        all_prods.extend(prods)
        if i % 100 == 0:
            print(f"Processed {i} trees... total productions so far: {len(all_prods)}")
    print(f"Collected total of {len(all_prods)} productions from {len(trees)} trees")
    return all_prods


def estimate_pcfg_pos(prods: List[Production], top_k: int = 999999):
    counts: Dict[str, Counter] = defaultdict(Counter)

    for p in prods:
        lhs_label = base_label(str(p.lhs()))

        rhs_labels = []
        for sym in p.rhs():
            if isinstance(sym, Nonterminal):
                rhs_labels.append(base_label(str(sym)))
            else:
                rhs_labels.append(str(sym))

        counts[lhs_label][tuple(rhs_labels)] += 1

    pcfg: Dict[str, List[Dict[str, Any]]] = {}

    for lhs_str, rhs_counter in counts.items():
        total = sum(rhs_counter.values())
        rule_list = []

        for rhs, c in rhs_counter.items():
            prob = c / total
            rule_list.append((rhs, prob))

        rule_list.sort(key=lambda x: x[1], reverse=True)
        rule_list = rule_list[:top_k]

        serialized = [
            {"rhs": list(rhs), "prob": prob, "log_prob": math.log(prob)}
            for rhs, prob in rule_list
        ]

        pcfg[lhs_str] = serialized

    print(f"Built POS-based PCFG with {len(pcfg)} LHS nonterminals")
    return pcfg


def save_pcfg(pcfg, out_path: str):
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pcfg, indent=2), encoding="utf-8")
    print(f"Saved POS PCFG with {len(pcfg)} LHS symbols to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build POS-based PCFG from parse trees")
    parser.add_argument(
        "--trees",
        type=str,
        required=True,
        help="Path to input POS-tree JSONL file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to save output PCFG JSON file",
    )

    args = parser.parse_args()

    trees = load_trees(args.trees)
    prods = collect_productions(trees)
    pcfg = estimate_pcfg_pos(prods)
    save_pcfg(pcfg, args.out)


if __name__ == "__main__":
    main()