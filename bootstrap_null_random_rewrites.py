"""
bootstrap_null_random_rewrites.py

Generate null distribution for rewrite metrics by randomly selecting
candidate rewrites per sentence, recomputing style deltas / coverage

Outputs CSV with per draw, per run rand metrics
run, draw_idx, style_delta, prop_changed, n_sentences

command to run:
    python bootstrap_null_random_rewrites.py \
        --runs-root runs \
        --draws 1000 \
        --seed 13 \
        --out runs_summary/null_bootstrap.csv
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cky_POS import PCFG, parse_and_style_score
from build_style_stats_pos import compute_deltas
from nltk import pos_tag, word_tokenize, sent_tokenize


def pick_gpt_essay(run_dir: Path) -> Optional[Path]:
    """Pick the held-out GPT essay text path."""
    gpt_dir = run_dir / "gpt_data"
    if not gpt_dir.exists():
        return None
    #prefer gpt_<topic>.txt thats not train
    candidates = [
        p for p in gpt_dir.glob("gpt_*.txt") if "train" not in p.stem and "clean" not in p.stem
    ]
    if candidates:
        return candidates[0]
    #fallback any gpt*.txt
    fallback = sorted(gpt_dir.glob("*.txt"))
    return fallback[0] if fallback else None


def pick_candidate_file(run_dir: Path) -> Optional[Path]:
    data_dir = run_dir / "data"
    if not data_dir.exists():
        return None
    cands = sorted(data_dir.glob("gpt_candidates*.jsonl"))
    return cands[0] if cands else None


def load_candidates(path: Path, orig_sents: List[str]) -> Dict[int, List[str]]:
    """
    Build a mapping: sent_idx -> list[candidates] 
    record with keys
        sent_idx: int, cands: list[str]
        orig: str, cands: list[str] 
    """
    mapping: Dict[int, List[str]] = {}
    sent_to_idx: Dict[str, int] = {}
    for i, s in enumerate(orig_sents):
        sent_to_idx.setdefault(s, i)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = None
            if "sent_idx" in rec:
                idx = rec["sent_idx"]
            elif "orig" in rec:
                idx = sent_to_idx.get(rec["orig"].strip())
            if idx is None:
                continue
            cands = rec.get("cands", [])
            if not isinstance(cands, list):
                continue
            mapping.setdefault(idx, []).extend(cands)
    return mapping


def random_rewrite(orig_sents: List[str], cand_map: Dict[int, List[str]], rng: random.Random) -> List[str]:
    new_sents = []
    for i, s in enumerate(orig_sents):
        cands = cand_map.get(i, [])
        if cands:
            new_sents.append(rng.choice(cands))
        else:
            new_sents.append(s)
    return new_sents


def _simple_split_sentences(text: str) -> List[str]:
    import re
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def compute_style_delta(text: str, g_you: PCFG, g_gpt: PCFG, mu_you: float, mu_gpt: float) -> Optional[float]:
    """
    Style delta = mean over sents of (style_you_norm - style_gpt_norm)
    """
    sents = [s.strip() for s in _simple_split_sentences(text) if s.strip()]
    if not sents:
        return None
    rels: List[float] = []
    for s in sents:
        _, _, sy, _ = parse_and_style_score(s, g_you)
        _, _, sg, _ = parse_and_style_score(s, g_gpt)
        if sy is None or sg is None:
            continue
        sy_norm = sy - mu_you
        sg_norm = sg - mu_gpt
        rels.append(sy_norm - sg_norm)
    if not rels:
        return None
    return float(sum(rels) / len(rels))


def compute_delta_means(random_text: str, gpt_stats: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute mean delta for POS bigrams and lexical P(word|POS) comparing
    random rewritten essay vs. GPT baseline stats
    """
    try:
        stats_rand = build_pos_and_lex_stats_text(random_text)
    except Exception as e:
        return None, None
    try:
        deltas = compute_deltas(stats_rand, gpt_stats)
    except Exception:
        return None, None

    def mean_delta(list_of_dicts: List[dict]) -> Optional[float]:
        vals = [d.get("delta") for d in list_of_dicts if isinstance(d, dict) and isinstance(d.get("delta"), (int, float))]
        return float(np.mean(vals)) if vals else None

    pos_mean = mean_delta(deltas.get("pos_bigrams", []))
    lex_mean = mean_delta(deltas.get("lexical", []))
    return pos_mean, lex_mean


def build_pos_and_lex_stats_text(text: str):
    """
    Inline version of build_pos_and_lex_stats but for raw text 
    """
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]

    pos_unigrams = {}
    pos_bigram_counts = {}
    pos_word_counts = {}

    for s in sents:
        words = word_tokenize(s)
        tagged = pos_tag(words)
        if not tagged:
            continue
        pos_seq = [pos for _, pos in tagged]

        for pos in pos_seq:
            pos_unigrams[pos] = pos_unigrams.get(pos, 0) + 1

        for a, b in zip(pos_seq, pos_seq[1:]):
            pos_bigram_counts[(a, b)] = pos_bigram_counts.get((a, b), 0) + 1

        for w, pos in tagged:
            pos_word_counts[(pos, w.lower())] = pos_word_counts.get((pos, w.lower()), 0) + 1

    # build P_next
    pos_next_probs: Dict[str, Dict[str, float]] = {}
    tmp_next = {}
    for (a, b), c in pos_bigram_counts.items():
        tmp_next.setdefault(a, {})
        tmp_next[a][b] = tmp_next[a].get(b, 0) + c
    for a, counter in tmp_next.items():
        total = sum(counter.values())
        pos_next_probs[a] = {b: c / total for b, c in counter.items()} if total else {}

    # build Pw_given_pos
    Pw_given_pos: Dict[str, Dict[str, float]] = {}
    tmp_pw = {}
    for (pos, w), c in pos_word_counts.items():
        tmp_pw.setdefault(pos, {})
        tmp_pw[pos][w] = tmp_pw[pos].get(w, 0) + c
    for pos, counter in tmp_pw.items():
        total = sum(counter.values())
        Pw_given_pos[pos] = {w: c / total for w, c in counter.items()} if total else {}

    return {
        "pos_unigrams": pos_unigrams,
        "P_next": pos_next_probs,
        "Pw_given_pos": Pw_given_pos,
    }


def process_run(run_dir: Path, draws: int, rng: random.Random) -> Optional[pd.DataFrame]:
    personal_pcfg = run_dir / "models" / "pcfg_personal_pos.json"
    gpt_pcfg = run_dir / "models" / "pcfg_gpt_pos.json"
    baselines = run_dir / "models" / "style_baselines.json"
    style_stats_path = run_dir / "models" / "style_stats_pos.json"
    cand_file = pick_candidate_file(run_dir)
    gpt_essay_path = pick_gpt_essay(run_dir)

    if not (personal_pcfg.exists() and gpt_pcfg.exists() and baselines.exists() and style_stats_path.exists() and cand_file and gpt_essay_path):
        return None

    g_you = PCFG(grammar_path=str(personal_pcfg))
    g_gpt = PCFG(grammar_path=str(gpt_pcfg))
    base = json.loads(baselines.read_text(encoding="utf-8"))
    mu_you = float(base.get("mu_you_on_you", 0.0))
    mu_gpt = float(base.get("mu_gpt_on_gpt", 0.0))
    try:
        style_stats = json.loads(style_stats_path.read_text(encoding="utf-8"))
        gpt_stats = style_stats.get("gpt", {})
    except Exception:
        gpt_stats = {}

    orig_text = gpt_essay_path.read_text(encoding="utf-8")
    orig_sents = [s.strip() for s in _simple_split_sentences(orig_text) if s.strip()]
    if not orig_sents:
        return None

    cand_map = load_candidates(cand_file, orig_sents)

    rows = []
    for d in range(draws):
        new_sents = random_rewrite(orig_sents, cand_map, rng)
        new_text = " ".join(new_sents)
        rel = compute_style_delta(new_text, g_you, g_gpt, mu_you, mu_gpt)
        if rel is None:
            continue
        pos_delta, lex_delta = compute_delta_means(new_text, gpt_stats) if gpt_stats else (None, None)
        if pos_delta is None:
            pos_delta = 0.0
        if lex_delta is None:
            lex_delta = 0.0
        changed = sum(1 for o, n in zip(orig_sents, new_sents) if o != n)
        prop_changed = changed / len(orig_sents)
        rows.append(
            {
                "draw_idx": d,
                "style_delta": rel,
                "prop_changed": prop_changed,
                "deltas_pos_bigrams_mean": pos_delta,
                "deltas_lexical_mean": lex_delta,
                "n_sentences": len(orig_sents),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap null by randomizing rewrite candidates.")
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=str, default="runs_summary/null_bootstrap.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    runs_root = Path(args.runs_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        df = process_run(run_dir, args.draws, rng)
        if df is None:
            continue
        df.insert(0, "run", run_dir.name)
        all_rows.append(df)

    if not all_rows:
        raise SystemExit("No runs processed; missing candidates/PCFGs?")

    final = pd.concat(all_rows, ignore_index=True)
    final.to_csv(out_path, index=False)
    print(f"[SAVED] Null bootstrap samples -> {out_path}")


if __name__ == "__main__":
    main()
