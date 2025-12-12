import argparse
import json
import pathlib
from typing import List, Optional

from graphviz import Digraph
from nltk import Tree


def load_tree_records(jsonl_path: str) -> List[dict]:
    """
    Load all records from mine_trees.jsonl
    each line JSON object with 
    - "text": sentence
    - "tree": bracketed tree string 
    """
    path = pathlib.Path(jsonl_path)
    records: List[dict] = []
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "tree" in rec and "text" in rec:
                records.append(rec)
    
    print(f"Loaded {len(records)} tree records from {jsonl_path}")
    return records

def visualize_tree_record(
    rec: dict,
    out_base: pathlib.Path,
    pretty_print: bool = True,
) -> None:
    """
    Pretty-print parse tree 
    save to PDF with Graphviz
    """
    sent_text = rec["text"]
    tree_str = rec["tree"]

    #parse tree string ->  NLTK Tree
    try:
        t: Tree = Tree.fromstring(tree_str)
    except Exception as e:
        print(f"[WARN] Could not parse tree: {e}")
        return

    #pretty-print in terminal
    if pretty_print:
        print("=" * 80)
        print("Sentence:")
        print(sent_text)
        print("\nTree (pretty_print):")
        t.pretty_print()
        print("=" * 80)

    #build graphviz Digraph from NLTK Tree
    dot = Digraph(comment=sent_text)
    dot.attr("graph", rankdir="TB")  #top-bottom layout
    dot.attr("node", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="8")

    node_counter = 0

    def add_subtree(subtree) -> str:
        nonlocal node_counter
        node_id = f"n{node_counter}"
        node_counter += 1

        if isinstance(subtree, Tree):
            #NT node: use label
            dot.node(node_id, label=subtree.label(), shape="oval")
            for child in subtree:
                child_id = add_subtree(child)
                dot.edge(node_id, child_id)
        else:
            #Leaf / terminal: draw as box with word
            dot.node(node_id, label=str(subtree), shape="box",
                     style="filled", fillcolor="lightgrey")

        return node_id

    add_subtree(t)

    #output directory exists
    out_base.parent.mkdir(parents=True, exist_ok=True)

    #render to pdf - graphviz
    pdf_path = out_base.with_suffix(".pdf")
    dot.render(str(out_base), format="pdf", cleanup=True)
    print(f"Saved tree visualization to {pdf_path}")
    

    
def visualize_example_trees(
    jsonl_path: str,
    out_dir: str,
    indices: Optional[List[int]] = None,
    num_examples: int = 3,
) -> None:
    
    """
    Vis ex trees from parsed writing
    args:
    jsonl_path: path to mine_trees.jsonl
    out_dir: directory to save tree images
    indices: indices of sentences to vis (0 based)
    num_examples: how many trees to vis
    """
    records = load_tree_records(jsonl_path)
    n = len(records)
    if n == 0:
        print("No records found, nothing to visualize.")
        return
    
    #which indices to show
    if indices is not None:
        chosen_indices = [i for i in indices if 0 <= i <n]
    else:
        chosen_indices = list(range(min(num_examples, n)))
    
    out_dir_path = pathlib.Path(out_dir)
    
    for i in chosen_indices:
        rec = records[i]
        out_path = out_dir_path / f"tree_{i}.ps"
        visualize_tree_record(rec, out_path, pretty_print=True)

def main():
    parser = argparse.ArgumentParser(
        description="Render parse trees from JSONL of bracketed trees using Graphviz"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to JSONL with 'text' and 'tree' fields",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to write rendered tree PDFs.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="List of 0-based sent idxs to vis(default: first N)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="If --indices not set, # first ex to render",
    )
    args = parser.parse_args()

    idx_list = None
    if args.indices:
        idx_list = []
        for tok in args.indices.split(","):
            tok = tok.strip()
            if tok.isdigit():
                idx_list.append(int(tok))

    visualize_example_trees(
        jsonl_path=args.jsonl,
        out_dir=args.out_dir,
        indices=idx_list,
        num_examples=args.num_examples,
    )

if __name__ == "__main__":
    main()
    
