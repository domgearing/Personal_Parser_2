import json
import math
import pathlib
import argparse
from typing import Dict, List, Tuple, Any, Optional

import nltk
from nltk import Tree, Nonterminal
from nltk.tokenize import word_tokenize, sent_tokenize

LOG_ZERO = -1e9  # approx log(0)
LOG_OOV_LEX = -15.0  #backup log-prob for unseen lexical POS tags


class PCFG:
    """
    Loads learned PCFG from JSON, exposes binary + lexical rule tables 
    for CKY, rule-level scoring.
    TERMINALS are POS TAGS (ex: 'PRP', 'MD', 'NN', ',', '.')
    """

    def __init__(
        self,
        start_symbol: str = "S",
        grammar_path: str = None,
        allow_oov_lexical: bool = True,
        oov_logp: float = LOG_OOV_LEX,
    ):
        if grammar_path is None:
            raise ValueError("PCFG requires grammar_path")
        self.start_symbol = start_symbol
        self.allow_oov_lexical = allow_oov_lexical
        self.oov_logp = oov_logp
        self.binary_rules: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        self.lexical_rules: Dict[str, List[Tuple[str, float]]] = {}
        self.nonterminals: set[str] = set()
        self._load(grammar_path)

    def _load(self, path: str) -> None:
        p = pathlib.Path(path)
        with p.open("r", encoding="utf-8") as f:
            pcfg = json.load(f)

        self.pcfg_raw = pcfg  #save for later analysis

        for lhs, rules in pcfg.items():
            self.nonterminals.add(lhs)
            for r in rules:
                rhs = r["rhs"]
                logp = r.get("log_prob", math.log(r["prob"]))
                if len(rhs) == 1:
                    #lexical rule: A -> POS_TAG
                    tok = rhs[0]
                    self.lexical_rules.setdefault(tok, []).append((lhs, logp))
                elif len(rhs) == 2:
                    #binary rule: A -> B C
                    key = (rhs[0], rhs[1])
                    self.binary_rules.setdefault(key, []).append((lhs, logp))
                else:
                    #should not happen in good CNF
                    continue


def cky_viterbi_with_rules(
    tokens: List[str], grammar: PCFG
) -> Tuple[Optional[float], Optional[Tree], List[float]]:
    """
    CKY with Viterbi backpointers over POS-tag tokens.

    Returns:
        best_logprob: float or None
        best_tree: nltk.Tree or None (tree over POS tags)
        rule_logps: list of log-probs for each production in best tree
        (empty list if no parse)
    """
    n = len(tokens)
    if n == 0:
        return None, None, []

    #chart: pi[(i, j)] = {A: best logprob for A covering tokens[i:j]}
    pi: Dict[Tuple[int, int], Dict[str, float]] = {}

    #backpointers: bp[(i, j, A)] = (k, B, C)
    bp: Dict[Tuple[int, int, str], Tuple[int, str, str]] = {}

    #init diagonal with lexical rules
    for i, tok in enumerate(tokens):
        cell: Dict[str, float] = {}

        if tok in grammar.lexical_rules:
            for A, logp in grammar.lexical_rules[tok]:
                cell[A] = max(cell.get(A, LOG_ZERO), logp)
        elif grammar.allow_oov_lexical:
            #backup: allow POS tag as own preterminal with low prob
            cell[tok] = grammar.oov_logp
            #keep record so later calls reuse this entry
            grammar.lexical_rules.setdefault(tok, []).append((tok, grammar.oov_logp))
        else:
            #shouldn't happen with POS-based grammar
            print(f"[DEBUG] No lexical entries for token {i}: {tok!r}")

        pi[(i, i + 1)] = cell

    #dynamic prog. for longer spans
    for span in range(2, n + 1):  #span length
        for i in range(0, n - span + 1):
            j = i + span
            best_for_span: Dict[str, float] = {}

            #try all split points
            for k in range(i + 1, j):
                left_cell = pi.get((i, k), {})
                right_cell = pi.get((k, j), {})
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
                                bp[(i, j, A)] = (k, B, C)

            pi[(i, j)] = best_for_span

    #get best root symbol for full span
    root_cell = pi.get((0, n), {})
    if not root_cell:
        return None, None, []

    if grammar.start_symbol in root_cell:
        root_sym = grammar.start_symbol
    elif "ROOT" in root_cell:
        root_sym = "ROOT"
    else:
        #backup - best nonterminal
        root_sym = max(root_cell.items(), key=lambda kv: kv[1])[0]

    best_logprob = root_cell[root_sym]

    #reconstruct best tree over POS tags
    def build_tree(i: int, j: int, A: str) -> Tree:
        if j == i + 1:
            return Tree(A, [tokens[i]])
        k, B, C = bp[(i, j, A)]
        left = build_tree(i, k, B)
        right = build_tree(k, j, C)
        return Tree(A, [left, right])

    best_tree = build_tree(0, n, root_sym)

    #extract rule log-probs from best tree
    rule_logps: List[float] = []

    for prod in best_tree.productions():
        lhs = str(prod.lhs())
        rhs_syms = prod.rhs()

        #lexical A -> POS
        if len(rhs_syms) == 1 and not isinstance(rhs_syms[0], Nonterminal):
            tok = rhs_syms[0]
            if tok in grammar.lexical_rules:
                for A, logp in grammar.lexical_rules[tok]:
                    if A == lhs:
                        rule_logps.append(logp)
                        break

        #binary A -> B C
        elif len(rhs_syms) == 2 and all(isinstance(s, Nonterminal) for s in rhs_syms):
            B = str(rhs_syms[0])
            C = str(rhs_syms[1])
            key = (B, C)
            if key in grammar.binary_rules:
                for A, logp in grammar.binary_rules[key]:
                    if A == lhs:
                        rule_logps.append(logp)
                        break

    return best_logprob, best_tree, rule_logps


def compute_style_score(rule_logps: List[float]) -> Optional[float]:
    """
    Style score = average log-probability / rule.
    Higher (less negative) means closer to learned style.
    """
    if not rule_logps:
        return None
    return sum(rule_logps) / len(rule_logps)


def parse_and_style_score(
    sent: str, grammar: PCFG
) -> Tuple[Optional[float], Optional[Tree], Optional[float], List[str]]:
    """
    Tokenize, POS-tag, CKY-parse over POS tags, compute style score.
    Returns: best_logprob, best_tree (over POS tags), style_score, pos_tokens
    """

    def normalize_pos_tag(tag: str) -> str:
        #map parentheses to PTB-style bracket tags
        if tag in {"(", "-LRB-"}:
            return "-LRB-"
        if tag in {")", "-RRB-"}:
            return "-RRB-"
        return tag

    words = word_tokenize(sent)
    raw_pos = nltk.pos_tag(words)
    pos_tokens = [normalize_pos_tag(tag) for _, tag in raw_pos]

    logp, tree, rule_logps = cky_viterbi_with_rules(pos_tokens, grammar)
    if logp is None or tree is None:
        return None, None, None, pos_tokens
    style_score = compute_style_score(rule_logps)
    return logp, tree, style_score, pos_tokens


def rule_style_gaps(tree: Tree, grammar: PCFG) -> List[Tuple[Tree, float]]:
    """
    For each node in tree, get 'style gap' for its production:
       style_gap = best_logp_for_LHS - logp_for_this_rule
    Tree is over POS tags in the leaves
    """
    gaps: List[Tuple[Tree, float]] = []

    #best log-prob / LHS from grammar
    best_logp_for_lhs: Dict[str, float] = {}
    for lhs, rules in grammar.pcfg_raw.items():
        best_logp_for_lhs[lhs] = max(r["log_prob"] for r in rules)

    for node in tree.subtrees():
        if isinstance(node, Tree) and len(node) > 0:
            lhs = node.label()
            #build RHS sequence of child labels/tags
            rhs = []
            for child in node:
                if isinstance(child, Tree):
                    rhs.append(child.label())
                else:
                    rhs.append(child)  # POS tag string

            #find logp of this rule
            logp_rule = None
            for r in grammar.pcfg_raw.get(lhs, []):
                if r["rhs"] == rhs:
                    logp_rule = r["log_prob"]
                    break

            if logp_rule is None:
                continue

            best = best_logp_for_lhs.get(lhs, logp_rule)
            gap = best - logp_rule
            gaps.append((node, gap))

    return gaps


def load_sentences_from_file(path: str) -> List[str]:
    """
    Load raw text file and split into sentences with NLTK
    """
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    return sents


def score_text(sentences: List[str], grammar: PCFG) -> Tuple[float, float, int]:
    """
    Score full text under POS-based grammar
    Returns:
    total_logprob_parsed : sum of sent log-probs over parsable sents
    avg_style_score : mean style_score over parsable sents
    num_parsed : # sents successfully parsed
    """
    total_logprob = 0.0
    style_scores: List[float] = []
    num_parsed = 0

    for sent in sentences:
        logp, tree, style, pos_tokens = parse_and_style_score(sent, grammar)
        if logp is None:
            #skip unparsable sentences
            continue
        num_parsed += 1
        total_logprob += logp
        if style is not None:
            style_scores.append(style)

    avg_style = sum(style_scores) / len(style_scores) if style_scores else float("nan")
    return total_logprob, avg_style, num_parsed


def main():
    parser = argparse.ArgumentParser(description="POS-based CKY + style scoring")
    parser.add_argument(
        "--grammar",
        type=str,
        required=True,
        help="Path to POS-based PCFG JSON (e.g., models/pcfg_personal_pos.json)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Path to raw text file to score",
    )
    parser.add_argument(
        "--sent-idx",
        type=int,
        default=0,
        help="0-based index of sentence in the text for detailed debug parse",
    )
    args = parser.parse_args()

    #load grammar via arg
    g = PCFG(start_symbol="S", grammar_path=args.grammar)

    #load sents
    sentences = load_sentences_from_file(args.text)
    print(f"Loaded {len(sentences)} sentences from {args.text}")

    #score whole text
    total_logp, avg_style, num_parsed = score_text(sentences, g)
    print("\n=== WHOLE-TEXT SCORES (POS-based PCFG) ===")
    print(f"Parsed sentences: {num_parsed} / {len(sentences)}")
    print(f"Total log-prob (sum over parsed sentences): {total_logp}")
    print(f"Average style score (avg logP per rule across sentences): {avg_style}")

    #pick one sent by index for debug
    if not sentences:
        print("\n[WARN] No sentences found in text; skipping debug")
        return

    idx = max(0, min(args.sent_idx, len(sentences) - 1))
    sent = sentences[idx]

    print(f"\n=== DETAILED DEBUG FOR SENTENCE INDEX {idx} ===")
    print("Sentence:", sent)

    logp, tree, style, pos_tokens = parse_and_style_score(sent, g)

    print("POS TOKENS:", pos_tokens)
    for t in pos_tokens:
        print(t, "→", "OK" if t in g.lexical_rules else "MISSING")

    if tree is None:
        print("No parse for sentence under this POS-based grammar")
        return

    gaps = rule_style_gaps(tree, g)
    gaps_sorted = sorted(gaps, key=lambda x: x[1], reverse=True)

    print("\nTop 5 biggest style gaps (POS-based):")
    for subtree, gap in gaps_sorted[:5]:
        print("LHS:", subtree.label(), "gap:", gap)
        print("Yield (POS):", " ".join(subtree.leaves()))
        print()

    print("Log-prob:", logp)
    print("Style score (avg rule log-prob):", style)
    tree.pretty_print()


if __name__ == "__main__":
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    main()
