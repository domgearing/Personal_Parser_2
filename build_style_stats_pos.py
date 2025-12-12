#build_style_stats_pos.py

import argparse
import json
import math
import pathlib
from collections import Counter, defaultdict
from typing import Dict, Tuple

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag


def build_pos_and_lex_stats(text_path: str):
    """
    From clean .txt file, get:
      -POS unigrams (counts)
      -POS bigrams (P(next_pos | pos))
      -Lexical P(word | POS)
    """

    path = pathlib.Path(text_path)
    text = path.read_text(encoding="utf-8")

    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]

    pos_unigrams = Counter()
    pos_bigram_counts: Counter[Tuple[str, str]] = Counter()
    pos_word_counts: Counter[Tuple[str, str]] = Counter()

    for s in sents:
        words = word_tokenize(s)
        tagged = pos_tag(words)  # -> list[(word, POS)]

        if not tagged:
            continue

        #POS seq
        pos_seq = [pos for _, pos in tagged]

        #unigrams
        for pos in pos_seq:
            pos_unigrams[pos] += 1

        #bigrams
        for a, b in zip(pos_seq, pos_seq[1:]):
            pos_bigram_counts[(a, b)] += 1

        #lexical counts
        for w, pos in tagged:
            pos_word_counts[(pos, w.lower())] += 1

    #build P(next_pos | pos)
    pos_next_probs: Dict[str, Dict[str, float]] = {}
    tmp_next = defaultdict(Counter)
    for (a, b), c in pos_bigram_counts.items():
        tmp_next[a][b] += c
    for a, counter in tmp_next.items():
        total = sum(counter.values())
        pos_next_probs[a] = {b: c / total for b, c in counter.items()}

    #build P(word | POS)
    Pw_given_pos: Dict[str, Dict[str, float]] = {}
    tmp_pw = defaultdict(Counter)
    for (pos, w), c in pos_word_counts.items():
        tmp_pw[pos][w] += c
    for pos, counter in tmp_pw.items():
        total = sum(counter.values())
        Pw_given_pos[pos] = {w: c / total for w, c in counter.items()}

    return {
        "pos_unigrams": dict(pos_unigrams),
        "P_next": pos_next_probs,
        "Pw_given_pos": Pw_given_pos,
    }


def compute_deltas(stats_you, stats_gpt, eps: float = 1e-9):
    """
    get log-odds deltas between YOU and GPT for
        POS bigrams
        lexical P(word | POS)
    """

    #POS bigrams
    deltas_pos_bigrams = []
    Pn_you = stats_you["P_next"]
    Pn_gpt = stats_gpt["P_next"]

    #union all bigrams
    all_bigrams = set()
    for a, d in Pn_you.items():
        for b in d.keys():
            all_bigrams.add((a, b))
    for a, d in Pn_gpt.items():
        for b in d.keys():
            all_bigrams.add((a, b))

    for (a, b) in all_bigrams:
        p_you = Pn_you.get(a, {}).get(b, eps)
        p_gpt = Pn_gpt.get(a, {}).get(b, eps)
        log_you = math.log(p_you)
        log_gpt = math.log(p_gpt)
        delta = log_you - log_gpt  #positive => more "you-ish"
        deltas_pos_bigrams.append(
            {
                "pos1": a,
                "pos2": b,
                "logP_you": log_you,
                "logP_gpt": log_gpt,
                "delta": delta,
            }
        )

    #lexical deltas
    deltas_lexical = []
    Pw_you = stats_you["Pw_given_pos"]
    Pw_gpt = stats_gpt["Pw_given_pos"]

    all_pos_word = set()
    for pos, d in Pw_you.items():
        for w in d.keys():
            all_pos_word.add((pos, w))
    for pos, d in Pw_gpt.items():
        for w in d.keys():
            all_pos_word.add((pos, w))

    for (pos, w) in all_pos_word:
        p_you = Pw_you.get(pos, {}).get(w, eps)
        p_gpt = Pw_gpt.get(pos, {}).get(w, eps)
        log_you = math.log(p_you)
        log_gpt = math.log(p_gpt)
        delta = log_you - log_gpt  #positive => more "you-ish"
        deltas_lexical.append(
            {
                "pos": pos,
                "word": w,
                "logP_you": log_you,
                "logP_gpt": log_gpt,
                "delta": delta,
            }
        )

    #sort 
    deltas_pos_bigrams.sort(key=lambda x: x["delta"], reverse=True)
    deltas_lexical.sort(key=lambda x: x["delta"], reverse=True)

    return {
        "pos_bigrams": deltas_pos_bigrams,
        "lexical": deltas_lexical,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build POS style stats for YOU vs GPT"
    )
    parser.add_argument(
        "--you-text",
        type=str,
        default="personal_data/personal_dbb_clean.txt",
        help="Clean txt file for your writing",
    )
    parser.add_argument(
        "--gpt-text",
        type=str,
        default="gpt_data/gpt_dbb_clean.txt",
        help="Clean txt file for GPT writing",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/style_stats_pos.json",
        help="Output JSON for style stats",
    )
    args = parser.parse_args()

    print(f"Building stats for YOU from {args.you_text} ...")
    stats_you = build_pos_and_lex_stats(args.you_text)
    print(f"Building stats for GPT from {args.gpt_text} ...")
    stats_gpt = build_pos_and_lex_stats(args.gpt_text)

    print("Computing deltas (YOU vs GPT)...")
    deltas = compute_deltas(stats_you, stats_gpt)

    out = {
        "you": stats_you,
        "gpt": stats_gpt,
        "deltas": deltas,
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved style stats to {out_path}")


if __name__ == "__main__":
    main()