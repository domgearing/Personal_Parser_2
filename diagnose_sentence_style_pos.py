# diagnose_sentence_style_pos.py

import argparse
import json
import math
import pathlib
from typing import Dict, Tuple, List

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

from cky_POS import (
    PCFG,
    cky_viterbi_with_rules,
    compute_style_score,
    rule_style_gaps,
)


def load_style_stats(path: str):
    p = pathlib.Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def lexical_weirdness(word: str, pos: str, stats: Dict, eps: float = 1e-9) -> float:
    """
    log(P_gpt(word|pos)) - log(P_you(word|pos)).
    Positive => more GPT-ish than you-ish.
    """
    Pw_you = stats["you"]["Pw_given_pos"]
    Pw_gpt = stats["gpt"]["Pw_given_pos"]
    w = word.lower()
    p_you = Pw_you.get(pos, {}).get(w, eps)
    p_gpt = Pw_gpt.get(pos, {}).get(w, eps)
    return math.log(p_gpt) - math.log(p_you)


def bigram_weirdness(
    pos1: str, pos2: str, stats: Dict, eps: float = 1e-9
) -> float:
    """
    log(P_gpt(pos2|pos1)) - log(P_you(pos2|pos1)).
    Positive => GPT uses this bigram more
    """
    Pn_you = stats["you"]["P_next"]
    Pn_gpt = stats["gpt"]["P_next"]
    p_you = Pn_you.get(pos1, {}).get(pos2, eps)
    p_gpt = Pn_gpt.get(pos1, {}).get(pos2, eps)
    return math.log(p_gpt) - math.log(p_you)


def pick_sentence(text_path: str, idx: int) -> str:
    p = pathlib.Path(text_path)
    text = p.read_text(encoding="utf-8")
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if idx < 0 or idx >= len(sents):
        raise IndexError(
            f"Sentence index {idx} out of range (0..{len(sents)-1}) for {text_path}"
        )
    return sents[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose how sent deviates from personal POS-level style"
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default=None,
        help="Sentence string to diagnose, if not given, use --text and --sent-idx",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Txt file to select a sent from (cleaned .txt)",
    )
    parser.add_argument(
        "--sent-idx",
        type=int,
        default=0,
        help="Sent idx (0-based) in --text",
    )
    parser.add_argument(
        "--style-stats",
        type=str,
        default="models/style_stats_pos.json",
        help="Path to style_stats_pos.json",
    )
    parser.add_argument(
        "--grammar",
        type=str,
        default="models/pcfg_personal_pos.json",
        help="Your POS-based PCFG JSON",
    )
    args = parser.parse_args()

    # choose sent
    if args.sentence is not None:
        sent = args.sentence.strip()
    else:
        if args.text is None:
            raise ValueError("Either --sentence or (--text and --sent-idx) must be given")
        sent = pick_sentence(args.text, args.sent_idx)

    print("Sentence:")
    print(sent)
    print("-" * 80)

    #load stats and grammar
    stats = load_style_stats(args.style_stats)
    g_you = PCFG(start_symbol="S", grammar_path=args.grammar)

    #tag POS
    words = word_tokenize(sent)
    tagged = pos_tag(words)
    pos_seq = [pos for _, pos in tagged]

    print("WORDS:", words)
    print("POS:", pos_seq)
    print()

    #parse under your POS-PCFG
    logp, tree, rule_logps = cky_viterbi_with_rules(pos_seq, g_you)
    if logp is None or tree is None:
        print("No POS-based parse under your grammar (can still inspect lexical and bigram style)")
        can_parse = False
        style_score = None
    else:
        can_parse = True
        style_score = compute_style_score(rule_logps)
        print(f"CKY parse log-prob: {logp}")
        print(f"Style score (avg rule logP): {style_score}")
        print()
        tree.pretty_print()

    # lexical weirdness
    lex_weird = []
    for (w, pos) in tagged:
        score = lexical_weirdness(w, pos, stats)
        lex_weird.append((w, pos, score))

    lex_weird_sorted = sorted(lex_weird, key=lambda x: x[2], reverse=True)

    print("\nTop lexical GPT-ish words (log P_gpt - log P_you):")
    for w, pos, sc in lex_weird_sorted[:10]:
        print(f"  {w!r:15} POS={pos:5}  weirdness={sc:.3f}")

    #POS bigram weirdness
    bigrams = []
    for a, b in zip(pos_seq, pos_seq[1:]):
        sc = bigram_weirdness(a, b, stats)
        bigrams.append((a, b, sc))

    bigrams_sorted = sorted(bigrams, key=lambda x: x[2], reverse=True)

    print("\nTop POS bigrams where GPT > YOU (log P_gpt - log P_you):")
    for a, b, sc in bigrams_sorted[:10]:
        print(f"  {a:5} -> {b:5}   weirdness={sc:.3f}")

    #rule lvl gaps (only if parsed)
    if can_parse:
        gaps = rule_style_gaps(tree, g_you)
        gaps_sorted = sorted(gaps, key=lambda x: x[1], reverse=True)

        print("\nTop 5 rule-level style gaps (your PCFG):")
        for subtree, gap in gaps_sorted[:5]:
            print("LHS:", subtree.label(), " gap:", f"{gap:.3f}")
            print("Yield (POS):", " ".join(subtree.leaves()))
            print()

    print("\nDone.")


if __name__ == "__main__":
    main()