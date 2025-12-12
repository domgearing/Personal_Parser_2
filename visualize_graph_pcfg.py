import argparse
import json
import pathlib
import re
from typing import List, Optional
from graphviz import Digraph


def load_pcfg(path: str) -> dict:
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_focus_nonterminals(pcfg: dict, focus_lhs: Optional[List[str]] = None, max_symbols: int = 10) -> List[str]:
    """
    pick which nonterminals to visualize
    if focus_lhs is given, use that intersection
    otherwise, pick the first max_symbols in sorted order
    """
    all_lhs = sorted(pcfg.keys())

    if focus_lhs is not None:
        # keep ones that exist in pcfg
        chosen = [lhs for lhs in focus_lhs if lhs in pcfg]
    else:
        chosen = all_lhs[:max_symbols]

    return chosen


def build_grammar_graph(
    pcfg: dict,
    focus_lhs: Optional[List[str]] = None,
    top_n_rules: int = 5,
    out_path: str = "outputs/pcfg_fragment"
) -> None:
    """
    Graphviz graph for fragment of PCFG
    - pcfg: grammar dict (LHS -> list of rules with 'rhs', 'prob')
    - focus_lhs: list of nonterminals to center the graph on (e.g. ["S", "NP", "VP"])
                 If None, pick up to 10 automatically.
    - top_n_rules: number of highest-prob rules per LHS / include
    - out_path: output out_path + ".pdf"
    """
    #which NT's to show 
    lhs_list = pick_focus_nonterminals(pcfg, focus_lhs, max_symbols=10)

    dot = Digraph(comment="PCFG fragment")
    dot.attr("graph", rankdir="LR")  # left-to-right
    dot.attr("node", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica")

    #detect nonterminal-looking symbols (all caps)
    def is_nonterminal(sym: str) -> bool:
        return bool(re.match(r"^[A-Z]+$", sym))

    #add nodes for chosen LHS symbols
    for lhs in lhs_list:
        dot.node(lhs, lhs, shape="ellipse", style="filled", fillcolor="lightblue")

    #add rule nodes and connect them
    for lhs in lhs_list:
        rules = sorted(pcfg[lhs], key=lambda r: r["prob"], reverse=True)[:top_n_rules]

        for idx, rule in enumerate(rules):
            rhs_syms = rule["rhs"]
            prob = rule.get("prob", 0.0)

            #create unique ID for rule node
            rule_id = f"{lhs}_rule_{idx}"

            #label for rule node
            rhs_str = " ".join(rhs_syms)
            rule_label = f"{lhs} - {rhs_str}\n p={prob:.3f}"

            #rule node - box
            dot.node(rule_id, label=rule_label, shape="box", style="rounded,filled", fillcolor="white")

            #edge from LHS nonterminal to rule node
            dot.edge(lhs, rule_id)

            #edges from rule node to any NT children
            for sym in rhs_syms:
                if is_nonterminal(sym) and sym in pcfg:
                    #ensure child NT exists as node 
                    dot.node(sym, sym, shape="ellipse", style="filled", fillcolor="lightgrey")
                    dot.edge(rule_id, sym)

    #make sure output directory is good
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    #render graph 
    dot.render(str(out_path), format="png", cleanup=True)
    print(f"Graph written to {out_path.with_suffix('.pdf')}")
    

def main():
    parser = argparse.ArgumentParser(
        description="Graphviz fragment of PCFG (top rules / selected nonterminal)"
    )
    parser.add_argument(
        "--grammar",
        type=str,
        required=True,
        help="Path to PCFG JSON file.",
    )
    parser.add_argument(
        "--focus-lhs",
        type=str,
        default=None,
        help="list of LHS symbols to vis (default: first 10)",
    )
    parser.add_argument(
        "--top-n-rules",
        type=int,
        default=5,
        help="Number of top-prob rules per NT to include",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/pcfg_fragment",
        help="Output path prefix",
    )
    args = parser.parse_args()

    pcfg = load_pcfg(args.grammar)
    focus = None
    if args.focus_lhs:
        focus = [s.strip() for s in args.focus_lhs.split(",") if s.strip()]

    build_grammar_graph(
        pcfg,
        focus_lhs=focus,
        top_n_rules=args.top_n_rules,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
