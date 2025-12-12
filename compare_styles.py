import argparse
import json
import math
import pathlib
from collections import defaultdict

LOG_FLOOR = -20.0  #floor for unseen rules so log prob diffs not inf

#ignore LHS symbols as mostly parser artifacts / not interesting
ARTIFACT_LHS = {
    "PDT", "LST", "SYM", "FRAG", "UCP", "HYPH", "INTJ", "RBS", "PRN", "NNP"
}

#RHS symbols - ignore if appear anywhere in a rule
#this just helps us clean up parse trees and cfg later on 
# to extract only most interesting, useful rules
NOISY_RHS_SYMBOLS = {
    "UNK", "-LRB-", "-RRB-", "``", "''"
}


def load_pcfg(path):
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def is_nonterminal(sym: str) -> bool:
    """
    Treat symbols that look like PTB style NT's as structural:
      - all caps letters (e.g., NP, VP, PP, S, SBAR, ADJP, ADVP, QP, CONJP, etc)
    """
    return sym.isalpha() and sym.upper() == sym


def is_structural_rule(lhs: str, rhs: tuple[str, ...]) -> bool:
    """
    Keep struct rules, not lexical artifacts.
    Rule struct:
      - LHS is NT we care about (and not in ARTIFACT_LHS)
      - RHS has >= one NT
      - RHS has no noisy symbols like UNK, -LRB-, etc
    """
    if lhs in ARTIFACT_LHS:
        return False

    if not is_nonterminal(lhs):
        return False
    if not rhs:
        return False

    #skip rules with noisy/placeholder symbols
    if any(sym in NOISY_RHS_SYMBOLS for sym in rhs):
        return False

    # must have >= 1 nonterminal-looking symbol on RHS
    if not any(is_nonterminal(sym) for sym in rhs):
        return False

    return True


def compute_style_diffs(personal_path, gpt_path, out_top_k=10):
    pcfg_you = load_pcfg(personal_path)
    pcfg_gpt = load_pcfg(gpt_path)

    #for quick lookup of log probs
    logP_you = defaultdict(dict)
    logP_gpt = defaultdict(dict)

    for lhs, rules in pcfg_you.items():
        for r in rules:
            rhs = tuple(r["rhs"])
            logP_you[lhs][rhs] = r.get("log_prob", math.log(r["prob"]))

    for lhs, rules in pcfg_gpt.items():
        for r in rules:
            rhs = tuple(r["rhs"])
            logP_gpt[lhs][rhs] = r.get("log_prob", math.log(r["prob"]))

    diffs = []

    #union of rules seen in either grammar
    all_lhs = set(logP_you.keys()) | set(logP_gpt.keys())
    for lhs in all_lhs:
        rhs_set = set(logP_you[lhs].keys()) | set(logP_gpt[lhs].keys())
        for rhs in rhs_set:
            #syntactic/struct rules only
            if not is_structural_rule(lhs, rhs):
                continue

            ly = logP_you[lhs].get(rhs, LOG_FLOOR)
            lg = logP_gpt[lhs].get(rhs, LOG_FLOOR)
            style_diff = ly - lg
            diffs.append((lhs, rhs, style_diff))

    if not diffs:
        print("No structural rules to compare - check your PCFGs or filters")
        return

    # sort by how much more personal like it than GPT
    diffs.sort(key=lambda x: x[2], reverse=True)

    print("\n=== Structural rules that are MOST 'you-ish' (you >> GPT) ===")
    for lhs, rhs, d in diffs[:out_top_k]:
        rhs_str = " ".join(rhs)
        print(f"{lhs:5} -> {rhs_str:25}   Δ(you-GPT)={d:6.2f}")

    print("\n=== Structural rules that are MOST 'GPT-ish' (GPT >> you) ===")
    diffs.sort(key=lambda x: x[2])  # most negative first
    for lhs, rhs, d in diffs[:out_top_k]:
        rhs_str = " ".join(rhs)
        print(f"{lhs:5} -> {rhs_str:25}   Δ(you-GPT)={d:6.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 2 PCFGs, report most personal-like vs GPT-like struct rules"
    )
    parser.add_argument(
        "--personal-grammar",
        type=str,
        required=True,
        help="Path to personal PCFG JSON.",
    )
    parser.add_argument(
        "--gpt-grammar",
        type=str,
        required=True,
        help="Path to GPT PCFG JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="# top rules to list per cat",
    )
    args = parser.parse_args()

    compute_style_diffs(
        args.personal_grammar,
        args.gpt_grammar,
        out_top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
