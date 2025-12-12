import argparse
import json
import pathlib
import math

def analyze_top_rules(grammar_path: str, top_k: int = 5):
    path = pathlib.Path(grammar_path)
    with path.open("r", encoding="utf-8") as f:
        pcfg = json.load(f)

    #focus on important categories from what I've seen
    interesting_LHS = ["S", "NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]

    for lhs in interesting_LHS:
        if lhs not in pcfg:
            continue
        rules = pcfg[lhs]
        #rules have probs, so only must sort and keep top_k
        rules_sorted = sorted(rules, key=lambda r: r["prob"], reverse=True)[:top_k]
        print(f"\n== {lhs} top {top_k} rules ==")
        
        #get entropy - how rigid each cat. is
        probs = [r["prob"] for r in pcfg[lhs]]
        entropy = -sum(p * math.log(p) for p in probs)
        print(f"Entropy H({lhs}) = {entropy:.3f}")
            
        for r in rules_sorted:
            print(f"{lhs} -> {' '.join(r['rhs'])}   (p={r['prob']:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="print entropy, top rules for imp. NT's in a PCFG."
    )
    parser.add_argument(
        "--grammar",
        type=str,
        required=True,
        help="path to PCFG JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="# of highest-prob rules to show per NT",
    )
    args = parser.parse_args()
    analyze_top_rules(args.grammar, args.top_k)


if __name__ == "__main__":
    main()
